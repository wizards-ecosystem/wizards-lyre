"""SPEC.md sec 12 Phase 5: UI to walk `parent_take_id` (restore an earlier
take as active without deleting history) -- this is the HTTP surface for it,
`POST /api/projects/{id}/active_take`.

Thin, focused coverage of just this endpoint; the auto-set-active-on-job
behavior it builds on is covered by tests/test_generate_take_flow.py and
tests/test_phase1_api.py.
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


def _generate(client: TestClient, project_id: str) -> str:
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    return job["take_id"]


def test_restore_earlier_take_as_active(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Active Take"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    first_take_id = _generate(client, project_id)
    second_take_id = _generate(client, project_id)
    assert first_take_id != second_take_id

    # every successful job auto-promotes its take to active (server/jobs.py)
    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["active_take_id"] == second_take_id

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": first_take_id},
    )
    assert resp.status_code == 200
    assert resp.json()["active_take_id"] == first_take_id

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["active_take_id"] == first_take_id


def test_set_active_take_404_on_unknown_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Active Take 404"}).json()
    project_id = project["id"]

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": "does-not-exist"},
    )
    assert resp.status_code == 404


def test_set_active_take_rejects_error_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Active Take Error"}).json()
    project_id = project["id"]

    storage.write_take_meta(
        project_id,
        "broken-take",
        {
            "id": "broken-take",
            "parent_take_id": None,
            "task_type": "text2music",
            "dit_profile": "iterate",
            "seed": 1,
            "duration_sec": None,
            "caption": None,
            "lyrics": None,
            "bpm": None,
            "keyscale": None,
            "created_at": "2026-01-01T00:00:00Z",
            "score": None,
            "error": "generation failed",
            "repaint": None,
            "track_name": None,
        },
    )

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": "broken-take"},
    )
    assert resp.status_code == 400


def test_set_active_take_404_when_audio_missing(client: TestClient) -> None:
    # meta.json with no `error` but no mix.wav/mix.mp3 on disk -- e.g. a
    # partially written take -- must not be activatable (reviewer-flagged:
    # a clean-looking meta isn't enough, the audio file must actually exist).
    project = client.post("/api/projects", json={"title": "Active Take No Audio"}).json()
    project_id = project["id"]

    storage.write_take_meta(
        project_id,
        "no-audio-take",
        {
            "id": "no-audio-take",
            "parent_take_id": None,
            "task_type": "text2music",
            "dit_profile": "iterate",
            "seed": 1,
            "duration_sec": 10.0,
            "caption": "test",
            "lyrics": "",
            "bpm": 120,
            "keyscale": "C Major",
            "created_at": "2026-01-01T00:00:00Z",
            "score": None,
            "error": None,
            "repaint": None,
            "track_name": None,
        },
    )

    resp = client.post(
        f"/api/projects/{project_id}/active_take",
        json={"take_id": "no-audio-take"},
    )
    assert resp.status_code == 404

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["active_take_id"] is None
