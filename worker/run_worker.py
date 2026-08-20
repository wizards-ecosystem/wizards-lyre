"""Dedicated ACE-Step worker process (SPEC.md sec 5, sec 10).

Run as its own OS process, separate from the FastAPI server, so a long
generation never blocks HTTP and a native GPU crash never touches it:

    python -m worker.run_worker

Polls `server.jobs`' SQLite queue for `queued` rows, claims and runs one at
a time against the backend selected by `BARD_WORKER` (default: `acestep`;
set `BARD_WORKER=mock` for local dev/tests without a GPU). Never imports
FastAPI or binds a port (SPEC.md sec 10 point 4).
"""

from __future__ import annotations

import threading

from server import jobs

DEFAULT_POLL_INTERVAL_SEC = 0.5


def run_loop(stop_event: threading.Event, poll_interval: float = DEFAULT_POLL_INTERVAL_SEC) -> None:
    """Claim and run queued jobs one at a time until `stop_event` is set."""
    jobs.init_db()
    while not stop_event.is_set():
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
