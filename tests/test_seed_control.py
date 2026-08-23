"""SPEC.md sec 7.3 seed contract: "-1 from the user means worker picks and
records" the seed -- which implies a real, user-supplied seed must be
honored verbatim. Covers the generate/cover/repaint actions the web UI's
seed input applies to:

- an explicit seed posted to POST /api/projects/{id}/jobs round-trips
  unchanged into the take's meta.json via the mock worker;
- seed -1 still yields a recorded, non--1 seed (the worker picks one).

The backend worker contract is already in place (worker/mock_worker.py
honors an explicit seed, worker/acestep_worker.py flips use_random_seed off
for fixed seeds); these tests pin the HTTP -> job queue -> worker -> meta
round-trip so a regression in any hop is caught.
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


def _create_project(client: TestClient, title: str) -> str:
    project = client.post("/api/projects", json={"title": title}).json()
    project_id = project["id"]
    # A custom-mode plan (caption present) so generate jobs don't rely on
    # the mock worker's simple-mode query fill.
    resp = client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "seed control test tone"},
    )
    assert resp.status_code == 200
    return project_id


def _run_job(client: TestClient, project_id: str, body: dict) -> dict:
    resp = client.post(f"/api/projects/{project_id}/jobs", json=body)
    assert resp.status_code == 200, resp.text
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["take_id"]
    return job


def _take_meta(project_id: str, take_id: str) -> dict:
    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def test_explicit_seed_round_trips_into_generate_take_meta(client: TestClient) -> None:
    project_id = _create_project(client, "Seed Generate")

    job = _run_job(
        client,
        project_id,
        {"action": "generate", "dit_profile": "iterate", "seed": 12345},
    )
    take_id = job["take_id"]

    # The posted fixed seed lands in meta.json unchanged (SPEC.md sec 7.3:
    # store the actual seed used).
    meta = _take_meta(project_id, take_id)
    assert meta["id"] == take_id
    assert meta["seed"] == 12345
    assert meta["error"] is None

    # ...and the same value is what the API reports for the take.
    detail = client.get(f"/api/projects/{project_id}").json()
    (take,) = [t for t in detail["takes"] if t["id"] == take_id]
    assert take["seed"] == 12345


def test_explicit_seed_round_trips_into_cover_take_meta(client: TestClient) -> None:
    project_id = _create_project(client, "Seed Cover")
    source = _run_job(
        client, project_id, {"action": "generate", "dit_profile": "iterate", "seed": -1}
    )

    job = _run_job(
        client,
        project_id,
        {
            "action": "cover",
            "dit_profile": "iterate",
            "source_take_id": source["take_id"],
            "audio_cover_strength": 0.7,
            "seed": 777,
        },
    )

    meta = _take_meta(project_id, job["take_id"])
    assert meta["task_type"] == "cover"
    assert meta["parent_take_id"] == source["take_id"]
    assert meta["seed"] == 777


def test_explicit_seed_round_trips_into_repaint_take_meta(client: TestClient) -> None:
    project_id = _create_project(client, "Seed Repaint")
    source = _run_job(
        client, project_id, {"action": "generate", "dit_profile": "iterate", "seed": -1}
    )

    job = _run_job(
        client,
        project_id,
        {
            "action": "repaint",
            "dit_profile": "iterate",
            "source_take_id": source["take_id"],
            "repainting_start": 0.0,
            "repainting_end": 0.25,
            "seed": 20260822,
        },
    )

    meta = _take_meta(project_id, job["take_id"])
    assert meta["task_type"] == "repaint"
    assert meta["seed"] == 20260822


@pytest.mark.parametrize("action", ["generate", "cover", "repaint"])
def test_seed_minus_one_records_a_real_seed(client: TestClient, action: str) -> None:
    project_id = _create_project(client, f"Seed Random {action}")

    body: dict = {"action": action, "dit_profile": "iterate", "seed": -1}
    if action in ("cover", "repaint"):
        source = _run_job(
            client, project_id, {"action": "generate", "dit_profile": "iterate", "seed": -1}
        )
        body["source_take_id"] = source["take_id"]
    if action == "repaint":
        body["repainting_start"] = 0.0
        body["repainting_end"] = 0.25

    job = _run_job(client, project_id, body)

    # -1 means "worker picks and records" (SPEC.md sec 7.3): the recorded
    # seed must be a real seed, never the -1 sentinel itself.
    meta = _take_meta(project_id, job["take_id"])
    assert isinstance(meta["seed"], int)
    assert meta["seed"] != -1
