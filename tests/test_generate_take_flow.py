"""SPEC.md sec 11: enqueue generate -> mocked worker -> take appears with
mix file + meta seed recorded.

Thin, focused coverage of the specific SPEC sec 11 bullet; broader job/take
behavior (error paths, plan-fill, batch_size clamping, etc.) is covered by
tests/test_phase1_api.py.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


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
    job = wait_for_job(client, resp.json()["id"])
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
    # SPEC.md sec 16.4: the take traces to its job, and the job's frozen
    # seed is the seed the take records.
    assert meta["job_id"] == job["id"]
    assert meta["provenance"]["actual"]["seed"] == meta["seed"]
    resolved = client.get(f"/v1/jobs/{job['id']}").json()
    assert resolved["seed"] == meta["seed"]

    # and it's reachable/playable through the API
    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/audio")
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


def test_audio_download_is_named_by_take_and_real_extension(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Name"}).json()["id"]
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    take_id = wait_for_job(client, job["id"])["take_id"]
    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/audio")
    assert f'filename="{take_id}.wav"' in resp.headers["content-disposition"]
    assert resp.headers["content-type"] == "audio/wav"
