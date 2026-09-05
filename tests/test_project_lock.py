"""The cross-process project lock (server/storage/locks.py).

Every metadata mutation for a project -- from the HTTP server and from the
worker process alike -- serializes on a lock file, because the two are
separate OS processes and an in-process lock would not see across that
boundary (SPEC.md sec 5/10). The happy path is exercised constantly by the
rest of the suite; the contention paths that make the lock safe rather than
merely present were not covered at all:

  - a lock abandoned by a crashed process must be reclaimed, or that project
    is unwritable forever
  - a lock held by a live process must time out rather than block indefinitely
  - a long-running holder must not have its own lock reclaimed underneath it
  - re-entering the lock on the same thread must not deadlock, since
    update_take_annotations and friends nest locked calls

No GPU, no server, no worker.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from server import storage
from server.storage import locks


@pytest.fixture()
def project(lyre_env) -> str:
    return storage.create_project(title="Lock Test")["id"]


def _lock_file(project_id: str):
    return storage.jailed_path(".project-locks", f"{project_id}.lock")


def test_lock_is_exclusive_between_threads(project: str) -> None:
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with locks._project_lock(project):
            order.append("holder-in")
            inside.set()
            release.wait(5)
            order.append("holder-out")

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert inside.wait(5)

    waiter_done = threading.Event()

    def waiter() -> None:
        with locks._project_lock(project):
            order.append("waiter-in")
        waiter_done.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.3)
    assert "waiter-in" not in order, "a second holder entered while the lock was held"

    release.set()
    thread.join(timeout=5)
    assert waiter_done.wait(5)
    assert order == ["holder-in", "holder-out", "waiter-in"]


def test_a_stale_lock_from_a_crashed_process_is_reclaimed(project: str) -> None:
    """A process that dies mid-write leaves its lock file behind. Nothing ever
    releases it, so unless it is reclaimed the project is permanently
    unwritable."""
    lock_path = _lock_file(project)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    # Backdate it past the staleness window, as a crash hours ago would look.
    old = time.time() - (locks._PROJECT_LOCK_STALE_SEC * 10)
    os.utime(lock_path, (old, old))

    with locks._project_lock(project):
        pass  # reclaimed rather than blocking until the timeout


def test_a_live_lock_times_out_instead_of_blocking_forever(
    project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock held by a process that is still alive must not be stolen, but the
    waiter must not hang forever either."""
    monkeypatch.setattr(locks, "_PROJECT_LOCK_TIMEOUT_SEC", 0.3)
    monkeypatch.setattr(locks, "_PROJECT_LOCK_POLL_SEC", 0.02)

    inside = threading.Event()
    release = threading.Event()
    failure: list[BaseException] = []

    def holder() -> None:
        with locks._project_lock(project):
            inside.set()
            release.wait(10)

    def waiter() -> None:
        try:
            with locks._project_lock(project):
                pass
        except BaseException as exc:
            failure.append(exc)

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    assert inside.wait(5)
    try:
        waiter_thread = threading.Thread(target=waiter, daemon=True)
        waiter_thread.start()
        waiter_thread.join(timeout=10)
        assert not waiter_thread.is_alive(), "waiting on a held lock never returned"
        assert failure and isinstance(failure[0], TimeoutError)
        assert project in str(failure[0])
    finally:
        release.set()
        holder_thread.join(timeout=5)


def test_a_long_holder_refreshes_its_own_lock(
    project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The holder touches its lock file on a timer. Without that, a legitimate
    operation slower than the staleness window would have its own lock
    reclaimed by a rival while it was still working."""
    monkeypatch.setattr(locks, "_PROJECT_LOCK_STALE_SEC", 0.3)
    lock_path = _lock_file(project)

    with locks._project_lock(project):
        first = lock_path.stat().st_mtime
        # Hold for longer than the staleness window and let the refresh fire.
        time.sleep(0.5)
        refreshed = lock_path.stat().st_mtime

    assert refreshed > first, "a long-held lock went stale under its own holder"


def test_the_lock_is_reentrant_on_one_thread(project: str) -> None:
    """storage.update_take_annotations takes the lock and then calls
    write_take_meta, which takes it again. A non-reentrant lock would deadlock
    on that path."""
    with locks._project_lock(project):
        with locks._project_lock(project):
            pass
        # The inner exit must not have released the outer hold.
        assert locks._PROJECT_LOCK_STATE.held[project] == 1


def test_the_lock_file_is_removed_after_release(project: str) -> None:
    lock_path = _lock_file(project)
    with locks._project_lock(project):
        assert lock_path.exists()
    assert not lock_path.exists(), "a released lock must not look held to the next process"


def test_the_lock_is_released_even_when_the_body_raises(project: str) -> None:
    lock_path = _lock_file(project)
    with pytest.raises(ValueError), locks._project_lock(project):
        raise ValueError("boom")
    assert not lock_path.exists()
    # And the project is still writable afterwards.
    with locks._project_lock(project):
        pass


def test_the_lock_lives_outside_the_project_directory(project: str) -> None:
    """A lock inside projects/<id> disappears during delete_project's rmtree,
    which would let a waiting writer create a fresh lock -- and a fresh project
    directory -- while the deletion still owns the old, unlinked file."""
    lock_path = _lock_file(project)
    assert storage.project_dir(project) not in lock_path.parents
