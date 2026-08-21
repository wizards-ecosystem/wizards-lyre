"""POST /api/projects/{id}/active_take: lets the UI walk parent_take_id and
restore an earlier take as active without deleting history (SPEC.md sec 12
Phase 5, sec 7.3). No GPU, no CUDA, no ACE-Step -- uses the mocked worker.

Mirrors tests/test_cover_flow.py's fixture/harness setup.
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


def _generate_take(client: TestClient, project_id: str) -> str:
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done"
    assert job["take_id"]
    return job["take_id"]


def test_set_active_take_restores_an_earlier_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Restore Test"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "synthwave"},
    )

    first_take_id = _generate_take(client, project_id)
    second_take_id = _generate_take(client, project_id)
    assert first_take_id != second_take_id

    # Existing auto-set behavior: the newest take is active by default.
    resp = client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["project"]["active_take_id"] == second_take_id

    # Walk parent_take_id back: restore the first take as active without
    # deleting either take's history.
    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": first_take_id},
    )
    assert resp.status_code == 200
    assert resp.json()["active_take_id"] == first_take_id

    resp = client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project"]["active_take_id"] == first_take_id
    # Neither take was deleted.
    assert {t["id"] for t in body["takes"]} == {first_take_id, second_take_id}


def test_set_active_take_404s_on_nonexistent_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Restore 404 Test"}).json()
    project_id = project["id"]

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": "does-not-exist"},
    )
    assert resp.status_code == 404


def test_set_active_take_rejects_a_failed_take(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active take must be playable (SPEC.md sec 7.3 `error`) -- a take
    whose generation failed should be rejected with 400, not silently made
    active."""
    project = client.post("/api/projects", json={"title": "Restore Error Test"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "synthwave"},
    )

    import worker.mock_worker as mock_worker_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated generation failure")

    monkeypatch.setattr(mock_worker_module, "run_job", _boom)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "error"
    failed_take_id = job["take_id"]
    assert failed_take_id

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": failed_take_id},
    )
    assert resp.status_code == 400
