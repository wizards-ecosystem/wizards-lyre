"""Dedicated ACE-Step worker process (SPEC.md sec 5, sec 10).

Run as its own OS process, separate from the FastAPI server, so a long
generation never blocks HTTP and a native GPU crash never touches it:

    python -m worker.run_worker

Polls `server.jobs`' SQLite queue for `queued` rows, claims and runs one at
a time against the backend selected by `BARD_WORKER` (default: `acestep`;
set `BARD_WORKER=mock` for local dev/tests without a GPU). Never imports
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
"""

from __future__ import annotations

import threading
import time
import uuid

from server import jobs, storage

DEFAULT_POLL_INTERVAL_SEC = 0.5


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


def _refresh_published_state(startup_ready: bool, startup_message: str) -> None:
    ready, message = _current_readiness(startup_ready, startup_message)
    _publish_status(ready, message)
    _publish_capabilities(ready, message)


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
                "Wizard's Bard worker: another worker process already holds the "
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
    instead of risking two workers on the GPU at once."""
    while not stop_event.wait(jobs.WORKER_LEASE_HEARTBEAT_INTERVAL_SEC):
        if not jobs.renew_worker_lease(owner_id):
            print("Wizard's Bard worker: lost the GPU lease to another worker; stopping.")
            stop_event.set()
            return


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set.

    Acquires the cross-process worker lease first (waiting out any live
    rival) and never initializes the backend or touches the job queue until
    it holds it."""
    jobs.init_db()
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
                if time.monotonic() - last_status_refresh >= jobs.WORKER_STATUS_HEARTBEAT_INTERVAL_SEC:
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


def main() -> None:
    stop_event = threading.Event()
    print("Wizard's Bard worker: polling for queued jobs (Ctrl+C to stop)...")
    try:
        run_loop(stop_event)
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    main()
