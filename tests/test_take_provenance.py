"""SPEC.md sec 16.4: every committed take traces to its job and its brief,
and records what was requested next to what actually rendered."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from server import storage
from server.generation import artifacts
from tests.helpers import wait_for_job


def _generate(client: TestClient, project_id: str, **body) -> dict:
    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate", **body})
    assert resp.status_code == 200, resp.text
    done = wait_for_job(client, resp.json()["id"])
    assert done["status"] == "done", done.get("error")
    return done


def test_a_take_records_requested_and_actual_values(client: TestClient) -> None:
    project_id = client.post(
        "/api/projects", json={"title": "Prov", "query": "tin whistle"}
    ).json()["id"]
    plan = client.get(f"/api/projects/{project_id}").json()["plan"]
    client.put(f"/api/projects/{project_id}/plan", json={**plan, "bpm": 140})
    done = _generate(client, project_id, seed=-1, submission_id="s-1")
    take = storage.get_take(project_id, done["take_id"])

    assert take["job_id"] == done["id"]
    assert take["provenance_complete"] is True
    prov = take["provenance"]
    assert prov["schema"] == "lyre.take-provenance/1"
    assert prov["job"] == {
        "id": done["id"],
        "attempt": 1,
        "submission_id": "s-1",
        "retry_of": None,
    }
    assert prov["requested"]["seed_policy"] == "random"
    assert prov["requested"]["seed"] == -1
    assert prov["requested"]["creation_mode"] == "from_idea"
    assert prov["actual"]["seed"] == take["seed"] != -1
    # Explicit BPM survived; the planner filled the caption.
    assert prov["actual"]["bpm"] == 140
    assert prov["actual_origins"]["bpm"] == "user"
    assert prov["actual_origins"]["caption"] == "planner"
    assert prov["brief"]["content_hash"].startswith("sha256:")
    assert prov["model"]["checkpoint"] == "acestep-v15-turbo"
    assert prov["model"]["backend"] == "mock"
    assert prov["app"]["version"] == artifacts.app_version()
    stamps = prov["timestamps"]
    assert stamps["submitted_at"] <= stamps["started_at"] <= stamps["committed_at"]


def test_audio_details_are_read_from_the_file(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Audio"}).json()["id"]
    done = _generate(client, project_id)
    take = storage.get_take(project_id, done["take_id"])
    wav = take["provenance"]["files"]["mix.wav"]
    path = storage.take_dir(project_id, done["take_id"]) / "mix.wav"
    assert wav["sample_rate"] == 8000
    assert wav["channels"] == 1
    assert wav["frames"] == 4000
    assert wav["duration_sec"] == 0.5
    assert wav["bytes"] == path.stat().st_size
    assert wav["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    # The take's duration is the measured length, not the requested 120 s.
    assert take["duration_sec"] == 0.5
    mp3 = storage.take_dir(project_id, done["take_id"]) / "mix.mp3"
    if mp3.exists():
        assert take["provenance"]["files"]["mix.mp3"]["bytes"] == mp3.stat().st_size
        assert take["provenance"]["files"]["mix.mp3"]["sample_rate"] is None  # unknown, not guessed


def test_legacy_takes_present_unknown_provenance(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Legacy"}).json()["id"]
    done = _generate(client, project_id)
    meta_path = storage.take_dir(project_id, done["take_id"]) / "meta.json"
    legacy = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("provenance", "job_id"):
        legacy.pop(key)
    meta_path.write_text(json.dumps(legacy), encoding="utf-8")

    take = client.get(f"/v1/projects/{project_id}/takes").json()["data"][0]
    assert take["provenance"] is None
    assert take["provenance_complete"] is False
    assert take["job_id"] is None
    # The file on disk is not rewritten to look complete.
    assert "provenance" not in json.loads(meta_path.read_text(encoding="utf-8"))


def test_a_failed_take_names_its_job(client: TestClient, monkeypatch) -> None:
    import worker.mock_worker as mock_worker_module

    real = mock_worker_module.run_job

    def write_then_fail(**kwargs):
        real(**kwargs)
        raise RuntimeError("decode failed")

    monkeypatch.setattr(mock_worker_module, "run_job", write_then_fail)
    project_id = client.post("/api/projects", json={"title": "Fail"}).json()["id"]
    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    failed = wait_for_job(client, resp.json()["id"])
    assert failed["status"] == "error"
    take = storage.get_take(project_id, failed["take_id"])
    assert take["error"].startswith("decode failed")
    assert take["job_id"] == failed["id"]
