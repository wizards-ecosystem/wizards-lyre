"""SPEC.md sec 12 (phase 2): enqueue cover against a source take -> mocked
worker -> resulting take records task_type='cover' and parent_take_id ==
the source take's id.

Thin, focused coverage of the cover job flow; broader job/take behavior
(error paths, plan-fill, batch_size clamping, phase gating, etc.) is
covered by tests/test_phase1_api.py. Mirrors
tests/test_generate_take_flow.py's fixture/harness setup.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


def test_cover_take_flow(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Cover Flow"}).json()
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

    cover_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "cover",
            "dit_profile": "iterate",
            "source_take_id": source_take_id,
            "audio_cover_strength": 0.5,
            "seed": -1,
        },
    )
    assert cover_resp.status_code == 200
    cover_job = wait_for_job(client, cover_resp.json()["id"])
    assert cover_job["status"] == "done", cover_job.get("error")
    cover_take_id = cover_job["take_id"]
    assert cover_take_id

    detail = client.get(f"/api/projects/{project_id}").json()
    cover_take = next(t for t in detail["takes"] if t["id"] == cover_take_id)
    assert cover_take["task_type"] == "cover"
    assert cover_take["parent_take_id"] == source_take_id
