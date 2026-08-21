"""Phase 4 (mocked): LoRA storage + job scaffolding. SPEC.md sec 4.4 ('Style
pack | LoRA train / load | 8+ songs') and sec 7 (`projects/<id>/loras/`).

This only exercises the mocked worker's `train_lora` -- see
worker/mock_worker.py and server/jobs.py's train_lora dispatch. The real
ACE-Step training call is a follow-up job's responsibility; no upstream API
research happens here.
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

MIN_LORA_SOURCE_TAKES = 8


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


def _make_takes(client: TestClient, project_id: str, count: int) -> list[str]:
    take_ids: list[str] = []
    for _ in range(count):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "dit_profile": "iterate", "seed": -1},
        )
        assert resp.status_code == 200
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")
        take_ids.append(job["take_id"])
    return take_ids


def test_train_lora_job_completes_and_lists(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Style Pack"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "My Style"},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    resp = client.get(f"/api/projects/{project_id}/loras")
    assert resp.status_code == 200
    loras = resp.json()
    assert len(loras) == 1
    lora = loras[0]
    assert lora["name"] == "My Style"
    assert lora["source_take_count"] == MIN_LORA_SOURCE_TAKES

    # adapter file lives on disk under the project's loras/ dir, jailed
    ldir = storage.lora_dir(project_id, lora["id"])
    assert (ldir / "adapter.bin").exists()
    assert (ldir / "meta.json").exists()
    assert ldir.is_relative_to(storage.config.projects_dir())


def test_train_lora_rejects_fewer_than_eight_sources(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Too Few"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES - 1)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Too Few"},
    )
    assert resp.status_code == 400
    # no job row is left behind for a rejected action
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())
    assert client.get(f"/api/projects/{project_id}/loras").json() == []
