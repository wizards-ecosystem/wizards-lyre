"""Phase 6 (SPEC.md sec 12): take-level favorites + free-text notes,
mirroring tests/test_project_favorites.py but scoped to a take's meta.json
via PATCH /api/projects/{id}/takes/{take_id}. Take generation itself is
mocked (worker.mock_worker) -- nothing here exercises real ACE-Step.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def _make_take(client: TestClient) -> tuple[str, str]:
    project = client.post("/api/projects", json={"title": "Take Annotations"}).json()
    project_id = project["id"]
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done"
    return project_id, job["take_id"]


def test_new_take_is_not_favorited_and_has_no_notes(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is False
    assert take["notes"] == ""


def test_patch_favorite_and_notes_round_trip(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": True, "notes": "great take, keep this seed"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["favorite"] is True
    assert body["notes"] == "great take, keep this seed"

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is True
    assert take["notes"] == "great take, keep this seed"


def test_patch_one_field_leaves_the_other_untouched(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"notes": "initial note"},
    )
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    assert resp.json()["notes"] == "initial note"

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["favorite"] is True
    assert body["notes"] == "initial note"  # untouched by the favorite-only patch

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is True
    assert take["notes"] == "initial note"

    # flip favorite back off, notes still untouched
    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": False},
    )
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    assert resp.json()["notes"] == "initial note"


def test_patch_unknown_take_404s(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "No Takes"}).json()
    resp = client.patch(
        f"/api/projects/{project['id']}/takes/does-not-exist",
        json={"favorite": True},
    )
    assert resp.status_code == 404
