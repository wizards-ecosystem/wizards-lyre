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

from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


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
    gen_job = wait_for_job(client, gen_resp.json()["id"])
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
    repaint_job = wait_for_job(client, repaint_resp.json()["id"])
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

    meta = storage.config.projects_dir() / project_id / "takes" / repaint_take_id / "meta.json"
    import json

    meta_json = json.loads(meta.read_text())
    assert meta_json["repaint"] == {"start": 1.5, "end": 3.0}

    source_meta = storage.config.projects_dir() / project_id / "takes" / source_take_id / "mix.wav"
    assert source_meta.exists()  # the source take's audio is untouched
