"""SPEC.md sec 12 (phase 2): enqueue cover against a source take -> mocked
worker -> resulting take records task_type='cover' and parent_take_id ==
the source take's id.

Thin, focused coverage of the cover job flow; broader job/take behavior
(error paths, plan-fill, batch_size clamping, phase gating, etc.) is
covered by tests/test_phase1_api.py. Mirrors
tests/test_generate_take_flow.py's fixture/harness setup.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app
from worker.run_worker import run_loop


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker_thread.start()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.01)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_cover_take_flow(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Cover Flow"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert gen_resp.status_code == 200
    gen_job = _wait_for_job(client, gen_resp.json()["id"])
    assert gen_job["status"] == "done", gen_job.get("error")
    source_take_id = gen_job["take_id"]
    assert source_take_id

    cover_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "cover",
            "dit_profile": "iterate",
            "source_take_id": source_take_id,
            "audio_cover_strength": 0.5,
            "seed": -1,
        },
    )
    assert cover_resp.status_code == 200
    cover_job = _wait_for_job(client, cover_resp.json()["id"])
    assert cover_job["status"] == "done", cover_job.get("error")
    cover_take_id = cover_job["take_id"]
    assert cover_take_id

    detail = client.get(f"/api/projects/{project_id}").json()
    cover_take = next(t for t in detail["takes"] if t["id"] == cover_take_id)
    assert cover_take["task_type"] == "cover"
    assert cover_take["parent_take_id"] == source_take_id
