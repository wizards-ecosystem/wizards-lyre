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

On startup it also publishes what it can currently load (see
`server.jobs.publish_worker_capability`) for every `dit_profile`, e.g.
whether `quality` (XL) has CPU-offload support (SPEC.md sec 4.1/8.1). This
worker process is the only one allowed to import acestep to find that out;
the FastAPI server just reads what got published (SPEC.md sec 10 point 4).
"""

from __future__ import annotations

import threading

from server import jobs, storage

DEFAULT_POLL_INTERVAL_SEC = 0.5


def _publish_capabilities() -> None:
    module = jobs.resolve_worker_module()
    check = getattr(module, "supports_dit_profile", None)
    if check is None:
        return
    for dit_profile in storage.VALID_DIT_PROFILES:
        supported, reason = check(dit_profile)
        jobs.publish_worker_capability(dit_profile, supported, reason)


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set."""
    jobs.init_db()
    jobs.reclaim_stale_jobs()  # recover anything a previous, now-dead worker left running
    _publish_capabilities()
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
