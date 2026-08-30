"""Dedicated ACE-Step worker process (SPEC.md sec 5, sec 10).

Run as its own OS process, separate from the FastAPI server, so a long
generation never blocks HTTP and a native GPU crash never touches it:

    python -m worker.run_worker

Polls `server.jobs`' SQLite queue for `queued` rows, claims and runs one at
a time against the backend selected by `LYRE_WORKER` (default: `acestep`;
set `LYRE_WORKER=mock` for local dev/tests without a GPU). Never imports
FastAPI or binds a port (SPEC.md sec 10 point 4).

On startup and on every poll, it also reclaims jobs a previous, now-dead
worker process left stuck `running` (see `server.jobs.reclaim_stale_jobs`)
-- that's what makes a killed process or a native CUDA crash recoverable
instead of leaving the job `running` forever.

Before polling, it runs the backend's startup readiness check (SPEC.md sec
10 point 1: detect CUDA, log VRAM, preload the default `iterate` DiT + LM)
via `<backend>.initialize_worker()`. It then publishes:

- overall readiness + which DiT (if any) is loaded, via
  `server.jobs.publish_worker_status` -- this is what `/api/health` reports
  (SPEC.md sec 8 `dit_loaded`)
- per-profile capability, via `server.jobs.publish_worker_capability` -- so
  `quality` (XL) CPU-offload support (SPEC.md sec 4.1/8.1) is reported
  honestly

That publish is *not* a one-time startup snapshot: after every claimed job
(success or failure), it republishes from the backend's current live state
(`get_loaded_dit_profile` / `supports_dit_profile`), not the frozen startup
result. Otherwise a job that switches the loaded DiT profile would leave
`/api/health` reporting the startup profile forever, and a worker that
initially failed to load but later recovers (a job succeeds anyway) would
stay reported unavailable with its capabilities still rejecting new jobs.
It's also republished periodically while idle (not just after a job), so a
genuinely-alive-but-idle worker doesn't fall foul of `server.jobs`'
freshness check (`WORKER_STATUS_STALE_AFTER_SEC`) and start reading as dead.
This worker process is the only one allowed to import acestep to find any
of this out; the FastAPI server just reads what got published (SPEC.md sec
10 point 4).

Before any of that, it acquires a cross-process singleton lease
(`server.jobs.acquire_worker_lease`) so at most one worker process ever
initializes ACE-Step or claims a job at a time (SPEC.md sec 4.3: one GPU
occupant, jobs serialize). A second worker process waits out the first
rather than loading a model alongside it -- it never touches the backend
until it actually holds the lease.

That SQLite lease is a *heartbeat* lease: it can be stolen from a still-live
owner if its heartbeat merely looks stale (see
`server.jobs.WORKER_LEASE_STALE_AFTER_SEC`), and once a synchronous job is
running on the main thread nothing here can interrupt it just because the
heartbeat thread stopped renewing. So before touching the SQLite lease at
all, this process also grabs a plain OS-level exclusive lock on a file next
to the jobs DB and holds it open for as long as the process runs. That kind
of lock has no staleness window: the OS holds it exactly as long as this
process is alive and drops it the instant the process exits or crashes, so
a rival worker can never get past `_acquire_os_singleton_lock` while this
one is still up, no matter how long a job takes or how the heartbeat thread
behaves.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import uuid
from typing import BinaryIO

from server import config, jobs, storage

if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle: BinaryIO) -> bool:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(handle: BinaryIO) -> None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _try_lock(handle: BinaryIO) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(handle: BinaryIO) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


DEFAULT_POLL_INTERVAL_SEC = 0.5
OS_LOCK_POLL_INTERVAL_SEC = 1.0


def _startup_readiness() -> tuple[bool, str]:
    module = jobs.resolve_worker_module()
    init_fn = getattr(module, "initialize_worker", None)
    if init_fn is None:
        return True, f"{module.__name__}: no startup initialization required"
    return init_fn()


def _current_readiness(startup_ready: bool, startup_message: str) -> tuple[bool, str]:
    """Recompute readiness from what's actually loaded right now, via cheap
    read-only getters -- never re-runs the (possibly expensive, model
    -swapping) startup load. Falls back to the last known startup result
    only when nothing has ever loaded successfully, so a genuine startup
    failure still reads as one instead of a generic "nothing loaded"."""
    module = jobs.resolve_worker_module()
    get_loaded = getattr(module, "get_loaded_dit_profile", None)
    loaded_dit_profile = get_loaded() if get_loaded is not None else None
    if loaded_dit_profile is not None:
        return True, f"worker: '{loaded_dit_profile}' DiT + LM currently loaded"
    return startup_ready, startup_message


def _publish_status(ready: bool, message: str) -> None:
    module = jobs.resolve_worker_module()
    get_loaded = getattr(module, "get_loaded_dit_profile", None)
    loaded_dit_profile = get_loaded() if get_loaded is not None else None
    jobs.publish_worker_status(ready, message, loaded_dit_profile)


def _publish_capabilities(ready: bool, message: str) -> None:
    module = jobs.resolve_worker_module()
    check = getattr(module, "supports_dit_profile", None)
    for dit_profile in storage.VALID_DIT_PROFILES:
        if not ready:
            # Nothing has loaded (startup failed and nothing since has
            # recovered) -- publish that honestly instead of letting a
            # per-profile check (which may not import acestep at all, e.g.
            # non-quality profiles) report "supported" only to fail on the
            # first job.
            jobs.publish_worker_capability(dit_profile, False, message)
        elif check is not None:
            supported, reason = check(dit_profile)
            jobs.publish_worker_capability(dit_profile, supported, reason)
        else:
            jobs.publish_worker_capability(dit_profile, True, None)
    _publish_train_lora_capability(module, ready, message)


def _publish_train_lora_capability(module, ready: bool, message: str) -> None:
    """`train_lora` (SPEC.md sec 4.4 style pack) isn't a `dit_profile`, so it
    doesn't fit the per-profile loop above -- it's published under its own
    key in the same `worker_capabilities` table, read by
    `server.jobs.enqueue_job`'s `_check_worker_capability("train_lora")`.
    Both backends implement `train_lora` (mock for tests; acestep wraps the
    real DatasetBuilder/LoRATrainer pipeline)."""
    if not ready:
        jobs.publish_worker_capability("train_lora", False, message)
        return
    supported = hasattr(module, "train_lora")
    reason = (
        None
        if supported
        else (f"worker backend '{module.__name__}' does not implement train_lora (SPEC.md sec 4.4)")
    )
    jobs.publish_worker_capability("train_lora", supported, reason)


def _refresh_published_state(startup_ready: bool, startup_message: str) -> None:
    ready, message = _current_readiness(startup_ready, startup_message)
    _publish_status(ready, message)
    _publish_capabilities(ready, message)


def _acquire_os_singleton_lock(stop_event: threading.Event) -> BinaryIO | None:
    """Block until this process holds an OS-level exclusive lock on a file
    next to the jobs DB (SPEC.md sec 4.3: one GPU occupant). Unlike the
    SQLite `worker_lease` heartbeat, this has no staleness window: the OS
    keeps it held exactly as long as this process is alive and releases it
    automatically the moment the process exits or crashes, so a rival
    worker can never proceed past this point while a live process still
    holds it -- even if that process is deep in a synchronous GPU call and
    its heartbeat thread has died or fallen behind.

    Returns the open, locked file handle -- keep it open for the process
    lifetime; closing it releases the lock -- or None if `stop_event` fired
    before the lock was ever acquired."""
    db_path = config.db_path()
    lock_path = db_path.parent / f"{db_path.name}.worker.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(fd, "r+b")
    if handle.read(1) == b"":
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
    handle.seek(0)

    announced = False
    while not stop_event.is_set():
        if _try_lock(handle):
            return handle
        if not announced:
            print(
                "Wizard's Lyre worker: another worker process holds the OS-level "
                "GPU lock; waiting for it to exit..."
            )
            announced = True
        stop_event.wait(OS_LOCK_POLL_INTERVAL_SEC)
    handle.close()
    return None


def _acquire_lease_or_wait(owner_id: str, stop_event: threading.Event) -> bool:
    """Block until we hold the single cross-process worker lease (SPEC.md
    sec 4.3: one GPU occupant), waiting out any live rival worker instead of
    initializing ACE-Step alongside it. Returns False only if `stop_event`
    fires (e.g. Ctrl+C) before the lease was ever acquired."""
    announced = False
    while not stop_event.is_set():
        if jobs.acquire_worker_lease(owner_id):
            return True
        if not announced:
            print(
                "Wizard's Lyre worker: another worker process already holds the "
                "GPU lease; waiting for it to finish or go stale..."
            )
            announced = True
        stop_event.wait(jobs.WORKER_LEASE_HEARTBEAT_INTERVAL_SEC)
    return False


def _lease_heartbeat_loop(owner_id: str, stop_event: threading.Event) -> None:
    """Keep the lease alive while this process runs. If renewal ever fails
    (our heartbeat went stale and another process claimed the lease first --
    e.g. this process hung longer than WORKER_LEASE_STALE_AFTER_SEC), signal
    `stop_event` so the main loop stops claiming/running jobs immediately
    instead of risking two workers on the GPU at once.

    A transient error from `renew_worker_lease` itself (e.g. a momentarily
    locked SQLite file) must not silently kill this thread -- an unhandled
    exception here would end the thread without ever setting `stop_event`,
    so the lease would quietly go stale while this process is still alive
    (and possibly still mid-job). Log it and retry on the next heartbeat
    tick instead. The OS-level lock acquired in `run_loop` before this
    thread ever starts is what actually keeps a rival worker off the GPU in
    the meantime; this loop's job is status, not that final guarantee."""
    while not stop_event.wait(jobs.WORKER_LEASE_HEARTBEAT_INTERVAL_SEC):
        try:
            renewed = jobs.renew_worker_lease(owner_id)
        except Exception:
            traceback.print_exc()
            continue
        if not renewed:
            print("Wizard's Lyre worker: lost the GPU lease to another worker; stopping.")
            stop_event.set()
            return


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set.

    Acquires the OS-level singleton lock first (waiting out any live rival
    -- see `_acquire_os_singleton_lock`), then the SQLite worker lease, and
    never initializes the backend or touches the job queue until it holds
    both."""
    jobs.init_db()
    os_lock = _acquire_os_singleton_lock(stop_event)
    if os_lock is None:
        return
    try:
        owner_id = uuid.uuid4().hex
        if not _acquire_lease_or_wait(owner_id, stop_event):
            return

        lease_thread = threading.Thread(
            target=_lease_heartbeat_loop, args=(owner_id, stop_event), daemon=True
        )
        lease_thread.start()
        try:
            jobs.reclaim_stale_jobs()  # recover anything a previous, now-dead worker left running
            startup_ready, startup_message = _startup_readiness()
            _refresh_published_state(startup_ready, startup_message)
            last_status_refresh = time.monotonic()
            while not stop_event.is_set():
                jobs.reclaim_stale_jobs()
                did_work = jobs.process_one_queued_job()
                if did_work:
                    # A job just ran (successfully or not) -- republish from live
                    # state so a DiT profile switch or a recovery from an earlier
                    # startup failure is visible immediately, not just at process
                    # startup.
                    _refresh_published_state(startup_ready, startup_message)
                    last_status_refresh = time.monotonic()
                else:
                    if (
                        time.monotonic() - last_status_refresh
                        >= jobs.WORKER_STATUS_HEARTBEAT_INTERVAL_SEC
                    ):
                        # Idle, but still alive -- keep touching worker_status so
                        # server.jobs' freshness check doesn't start reporting a
                        # perfectly healthy worker as dead (see module docstring).
                        _refresh_published_state(startup_ready, startup_message)
                        last_status_refresh = time.monotonic()
                    stop_event.wait(poll_interval)
        finally:
            stop_event.set()
            lease_thread.join(timeout=jobs.WORKER_LEASE_HEARTBEAT_INTERVAL_SEC)
            jobs.release_worker_lease(owner_id)
    finally:
        _unlock(os_lock)
        os_lock.close()


def main() -> None:
    stop_event = threading.Event()
    print("Wizard's Lyre worker: polling for queued jobs (Ctrl+C to stop)...")
    try:
        run_loop(stop_event)
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    main()
