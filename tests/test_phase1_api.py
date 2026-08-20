"""Phase 1 API tests: health, project+plan disk round trip, mocked generate
flow, path jail, and studio_ops enforcement. No GPU, no CUDA, no ACE-Step.
See SPEC.md sec 11 and 14.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    # Tests always use the mocked worker; the real acestep_worker is
    # production's default (see server/jobs.py) and is exercised only by the
    # manual, non-pytest scripts/smoke-gpu.py.
    monkeypatch.setenv("BARD_WORKER", "mock")
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "dit_loaded" in body


def test_project_create_and_plan_roundtrip(client: TestClient) -> None:
    resp = client.post("/api/projects", json={"title": "My Song", "query": "lofi beat"})
    assert resp.status_code == 200
    project = resp.json()
    project_id = project["id"]
    assert project["title"] == "My Song"
    assert project["dit_profile"] == "iterate"

    # project.json actually landed on disk under projects/
    on_disk = storage.load_project(project_id)
    assert on_disk["id"] == project_id

    new_plan = {
        "query": "lofi beat",
        "caption": "lofi, chill, rhodes, brushed drums",
        "negative": [],
        "lyrics": "[Verse]\nquiet streets tonight",
        "instrumental": False,
        "vocal_language": "en",
        "bpm": 84,
        "keyscale": "A Minor",
        "timesignature": "4/4",
        "duration_sec": 90,
        "sections": [],
    }
    resp = client.put(f"/api/projects/{project_id}/plan", json=new_plan)
    assert resp.status_code == 200

    # plan.json round trips through disk, not just the response body
    plan_path = storage.plan_json_path(project_id)
    on_disk_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert on_disk_plan["caption"] == new_plan["caption"]
    assert on_disk_plan["bpm"] == 84

    resp = client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"]["caption"] == new_plan["caption"]
    assert body["takes"] == []

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    listed = resp.json()
    assert any(p["id"] == project_id for p in listed)


def test_generate_job_creates_playable_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Take Test"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "synthwave, driving bass"},
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert resp.status_code == 200
    job = resp.json()
    assert job["status"] == "done"
    assert job["error"] is None
    take_id = job["take_id"]
    assert take_id

    # job is retrievable by id and shows up in recent jobs
    resp = client.get(f"/api/jobs/{job['id']}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"

    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert any(j["id"] == job["id"] for j in resp.json())

    # take metadata recorded an actual seed (seed=-1 means "worker picks")
    resp = client.get(f"/api/projects/{project_id}/takes")
    assert resp.status_code == 200
    takes = resp.json()
    assert len(takes) == 1
    take = takes[0]
    assert take["id"] == take_id
    assert take["task_type"] == "text2music"
    assert isinstance(take["seed"], int)
    assert take["seed"] != -1
    assert take["caption"] == "synthwave, driving bass"

    # audio is a real, playable (tiny) WAV file written under projects/
    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/audio")
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
    assert resp.content[8:12] == b"WAVE"
    assert len(resp.content) > 44  # header + at least some frame data

    audio_path = storage.take_audio_path(project_id, take_id)
    assert audio_path.exists()
    assert audio_path.is_relative_to(storage.config.projects_dir())

    # take is immutable metadata + audio living in its own directory
    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    assert meta_path.exists()


def test_path_jail_rejects_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    with pytest.raises(storage.PathJailError):
        storage.jailed_path("..", "evil.txt")

    with pytest.raises(storage.PathJailError):
        storage.jailed_path("..", "..", "..", "Users", "evil.txt")

    # a well-behaved relative path stays inside the jail
    ok_path = storage.jailed_path("some-project", "takes", "some-take", "mix.wav")
    assert ok_path.is_relative_to(storage.config.projects_dir())


def test_path_jail_rejects_escape_via_api(client: TestClient) -> None:
    resp = client.get("/api/projects/..")
    assert resp.status_code in (400, 404)


def test_studio_ops_required_for_extract_lego_complete(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Ops Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    # a source take to extract/lego/complete from
    gen = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    ).json()
    source_take_id = gen["take_id"]

    for action in ("extract", "lego", "complete"):
        # explicit wrong profile is rejected outright
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "dit_profile": "iterate",
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action

        # omitting dit_profile is coerced to studio_ops and the job runs
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 200, action
        job = resp.json()
        assert job["dit_profile"] == "studio_ops"
        assert job["status"] == "done"

        # explicit studio_ops is accepted
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "dit_profile": "studio_ops",
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 200, action


def test_cover_repaint_extract_require_a_real_source(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Source Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    for action in ("cover", "repaint", "extract", "lego", "complete"):
        # no source_take_id and no upload_path at all
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": action, "track_name": "vocals"},
        )
        assert resp.status_code == 400, action

        # a source_take_id that doesn't exist
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "source_take_id": "does-not-exist",
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action

        # an upload_path that escapes the project's jail
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "upload_path": "../../../../evil.wav",
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action


def test_invalid_action_rejected(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Bad Action"}).json()
    project_id = project["id"]
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "not-a-real-action"},
    )
    assert resp.status_code == 400
