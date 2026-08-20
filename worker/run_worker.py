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
via `<backend>.initialize_worker()`, then publishes what it can currently
load (see `server.jobs.publish_worker_capability`) for every `dit_profile`
based on that *actual* result -- not an optimistic guess -- so `quality`
(XL) CPU-offload support (SPEC.md sec 4.1/8.1) is reported honestly, and a
startup failure (missing ACE-Step/CUDA/weights) is visible immediately
instead of only surfacing on the first job. This worker process is the only
one allowed to import acestep to find any of this out; the FastAPI server
just reads what got published (SPEC.md sec 10 point 4).
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


def _publish_capabilities(startup_ready: bool, startup_message: str) -> None:
    module = jobs.resolve_worker_module()
    check = getattr(module, "supports_dit_profile", None)
    for dit_profile in storage.VALID_DIT_PROFILES:
        if not startup_ready:
            # Startup couldn't even load the default profile -- ACE-Step,
            # CUDA, or weights are unavailable, so nothing else will load
            # either. Publish that honestly instead of letting a per-profile
            # check (which may not import acestep at all, e.g. non-quality
            # profiles) report "supported" only to fail on the first job.
            jobs.publish_worker_capability(dit_profile, False, startup_message)
        elif check is not None:
            supported, reason = check(dit_profile)
            jobs.publish_worker_capability(dit_profile, supported, reason)
        else:
            jobs.publish_worker_capability(dit_profile, True, None)


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set."""
    jobs.init_db()
    jobs.reclaim_stale_jobs()  # recover anything a previous, now-dead worker left running
    startup_ready, startup_message = _startup_readiness()
    _publish_capabilities(startup_ready, startup_message)
    while not stop_event.is_set():
        jobs.reclaim_stale_jobs()
        did_work = jobs.process_one_queued_job()
        if not did_work:
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
