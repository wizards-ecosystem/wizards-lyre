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
This worker process is the only one allowed to import acestep to find any
of this out; the FastAPI server just reads what got published (SPEC.md sec
10 point 4).
"""

from __future__ import annotations

import threading

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


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set."""
    jobs.init_db()
    jobs.reclaim_stale_jobs()  # recover anything a previous, now-dead worker left running
    startup_ready, startup_message = _startup_readiness()
    _refresh_published_state(startup_ready, startup_message)
    while not stop_event.is_set():
        jobs.reclaim_stale_jobs()
        did_work = jobs.process_one_queued_job()
        if did_work:
            # A job just ran (successfully or not) -- republish from live
            # state so a DiT profile switch or a recovery from an earlier
            # startup failure is visible immediately, not just at process
            # startup.
            _refresh_published_state(startup_ready, startup_message)
        else:
            stop_event.wait(poll_interval)


def main() -> None:
    stop_event = threading.Event()
    print("Wizard's Bard worker: polling for queued jobs (Ctrl+C to stop)...")
    try:
        run_loop(stop_event)
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    main()
