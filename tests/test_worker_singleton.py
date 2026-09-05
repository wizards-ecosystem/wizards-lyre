"""The one-GPU-occupant guard (SPEC.md sec 4.3).

`worker/run_worker.py` refuses to initialize a backend or claim a job until it
holds two things: an OS-level exclusive file lock, and the SQLite worker lease.
Those two exist for different reasons -- the file lock has no staleness window
and dies with the process, while the lease is what `/api/health` and the server
can actually observe -- and between them they are what stops two workers ever
loading models onto the same GPU.

None of this needs a GPU to exercise, and none of it was covered.
"""

from __future__ import annotations

import threading
import time

import pytest

from server import config, jobs
from worker import run_worker


@pytest.fixture()
def db(lyre_env, monkeypatch: pytest.MonkeyPatch):
    jobs.init_db()
    return config.db_path()


def test_os_lock_is_exclusive_and_released_on_close(db) -> None:
    """A second acquirer must not get the lock while the first holds it, and
    must get it immediately once the first closes."""
    first_stop = threading.Event()
    first = run_worker._acquire_os_singleton_lock(first_stop)
    assert first is not None

    # A rival that gives up quickly gets nothing while we hold the lock.
    rival_stop = threading.Event()
    rival_result: list = []

    def rival() -> None:
        rival_result.append(run_worker._acquire_os_singleton_lock(rival_stop))

    thread = threading.Thread(target=rival, daemon=True)
    thread.start()
    time.sleep(0.3)
    assert rival_result == [], "a second worker acquired the GPU lock while it was held"

    rival_stop.set()
    thread.join(timeout=5)
    assert rival_result == [None], "waiting worker must return None once asked to stop"

    # Closing the handle releases it, exactly as process exit would.
    first.close()
    second = run_worker._acquire_os_singleton_lock(threading.Event())
    assert second is not None
    second.close()


def test_os_lock_returns_none_when_stopped_before_acquiring(db) -> None:
    held = run_worker._acquire_os_singleton_lock(threading.Event())
    assert held is not None
    try:
        stop = threading.Event()
        stop.set()
        assert run_worker._acquire_os_singleton_lock(stop) is None
    finally:
        held.close()


def test_lease_is_acquired_when_free_and_waits_out_a_live_rival(db) -> None:
    rival = "rival-worker"
    assert jobs.acquire_worker_lease(rival) is True

    # A live rival holds the lease, so this must not return True.
    stop = threading.Event()
    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(run_worker._acquire_lease_or_wait("mine", stop)), daemon=True
    )
    thread.start()
    time.sleep(0.2)
    assert result == [], "acquired the lease while a live rival held it"

    stop.set()
    thread.join(timeout=10)
    assert result == [False], "must report failure rather than proceeding onto the GPU"

    jobs.release_worker_lease(rival)
    assert run_worker._acquire_lease_or_wait("mine", threading.Event()) is True


def test_heartbeat_stops_the_worker_when_the_lease_is_lost(
    db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If another process takes the lease over, the heartbeat must set
    stop_event so the main loop stops claiming jobs -- not keep running."""
    monkeypatch.setattr(jobs, "WORKER_LEASE_HEARTBEAT_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(run_worker.jobs, "renew_worker_lease", lambda _owner: False)

    stop = threading.Event()
    thread = threading.Thread(
        target=run_worker._lease_heartbeat_loop, args=("mine", stop), daemon=True
    )
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert stop.is_set(), "losing the lease must stop the worker loop"


def test_heartbeat_survives_a_transient_renewal_error(db, monkeypatch: pytest.MonkeyPatch) -> None:
    """A momentarily locked SQLite file must not kill the heartbeat thread
    silently -- that would let the lease go stale while this process is still
    alive and possibly mid-job."""
    monkeypatch.setattr(jobs, "WORKER_LEASE_HEARTBEAT_INTERVAL_SEC", 0.01)
    calls: list[int] = []

    def flaky(_owner: str) -> bool:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return True

    monkeypatch.setattr(run_worker.jobs, "renew_worker_lease", flaky)

    stop = threading.Event()
    thread = threading.Thread(
        target=run_worker._lease_heartbeat_loop, args=("mine", stop), daemon=True
    )
    thread.start()
    deadline = time.time() + 5
    while len(calls) < 4 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5)

    assert len(calls) >= 4, "heartbeat thread died on a transient error instead of retrying"


def test_run_loop_does_not_start_a_backend_without_the_os_lock(
    db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordering that matters: nothing touches the backend or the queue
    until the singleton lock is held. With the lock already taken and the stop
    event set, run_loop must return having done neither."""
    monkeypatch.setattr(
        run_worker,
        "_startup_readiness",
        lambda: pytest.fail("backend initialized without the lock"),
    )
    monkeypatch.setattr(
        jobs, "claim_next_queued_job", lambda: pytest.fail("claimed a job without the lock")
    )

    held = run_worker._acquire_os_singleton_lock(threading.Event())
    assert held is not None
    try:
        stop = threading.Event()
        stop.set()
        run_worker.run_loop(stop, poll_interval=0.01)
    finally:
        held.close()
