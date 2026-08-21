"""SPEC.md sec 12 (phase 2): enqueue repaint against a source take with a
region (repainting_start/repainting_end) -> mocked worker -> resulting take
records task_type='repaint', parent_take_id == the source take's id, and
the repaint interval actually submitted.

Thin, focused coverage of the repaint job flow (the web UI's drag-to-select
region feeds these two fields -- see web/src/App.tsx's repaint()); broader
job/take behavior (error paths, plan-fill, batch_size clamping, phase
gating, etc.) is covered by tests/test_phase1_api.py. Mirrors
tests/test_cover_flow.py's fixture/harness setup.
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


def test_repaint_take_flow(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Repaint Flow"}).json()
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

    repaint_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "repaint",
            "dit_profile": "iterate",
            "source_take_id": source_take_id,
            "repainting_start": 1.5,
            "repainting_end": 3.0,
            "seed": -1,
        },
    )
    assert repaint_resp.status_code == 200
    repaint_job = _wait_for_job(client, repaint_resp.json()["id"])
    assert repaint_job["status"] == "done", repaint_job.get("error")
    repaint_take_id = repaint_job["take_id"]
    assert repaint_take_id
    # A repaint (like every job) creates a brand-new, immutable take rather
    # than mutating the source's mix.wav in place (SPEC.md sec 7.3: "Every
    # take is immutable").
    assert repaint_take_id != source_take_id

    detail = client.get(f"/api/projects/{project_id}").json()
    repaint_take = next(t for t in detail["takes"] if t["id"] == repaint_take_id)
    assert repaint_take["task_type"] == "repaint"
    assert repaint_take["parent_take_id"] == source_take_id

    meta = (
        storage.config.projects_dir()
        / project_id
        / "takes"
        / repaint_take_id
        / "meta.json"
    )
    import json

    meta_json = json.loads(meta.read_text())
    assert meta_json["repaint"] == {"start": 1.5, "end": 3.0}

    source_meta = (
        storage.config.projects_dir()
        / project_id
        / "takes"
        / source_take_id
        / "mix.wav"
    )
    assert source_meta.exists()  # the source take's audio is untouched
