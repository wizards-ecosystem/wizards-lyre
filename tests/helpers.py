"""Helpers shared by Lyre's tests. Not collected by pytest (no `test_` prefix)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    """Poll `GET /api/jobs/{id}` until the job leaves the queued/running
    states, and return its final row. Only usable with a client that has a
    worker draining the queue (the `worker_client` fixture).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.01)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
