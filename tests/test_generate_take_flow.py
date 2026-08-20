"""SPEC.md sec 11: enqueue generate -> mocked worker -> take appears with
mix file + meta seed recorded.

Thin, focused coverage of the specific SPEC sec 11 bullet; broader job/take
behavior (error paths, plan-fill, batch_size clamping, etc.) is covered by
tests/test_phase1_api.py.
"""

from __future__ import annotations

import json
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

    # enqueue_job only inserts a `queued` row (SPEC.md sec 5); this thread
    # stands in for the dedicated worker/run_worker.py process that drains
    # the same SQLite queue in production.
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


def test_generate_take_flow(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Generate Flow"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    take_id = job["take_id"]
    assert take_id

    # mix.wav exists on disk under the project's take dir
    audio_path = storage.take_audio_path(project_id, take_id)
    assert audio_path.name == "mix.wav"
    assert audio_path.exists()
    assert audio_path.is_relative_to(storage.config.projects_dir())

    # meta.json records a real, non-sentinel seed (SPEC.md sec 7.3: -1 means
    # "worker picks and records it")
    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == take_id
    assert isinstance(meta["seed"], int)
    assert meta["seed"] != -1
    assert meta["error"] is None

    # and it's reachable/playable through the API
    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/audio")
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
