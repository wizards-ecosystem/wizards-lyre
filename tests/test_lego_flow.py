"""SPEC.md sec 4.4: lego adds/replaces a track, taking `src_audio`, a target
`track_name`, and an optional repaint interval. The backend
(`server.jobs._resolve_dit_profile`/`_resolve_source_audio`) and worker
adapter (`worker/acestep_worker.py`) already implement lego's full call
contract -- exercised directly by tests/test_acestep_worker_adapter.py --
and `_resolve_track_name`/studio_ops enforcement are already covered for all
three studio_ops actions by tests/test_extract_requires_studio_ops.py and
tests/test_phase1_api.py::test_resolve_dit_profile_studio_ops_enforcement.
This module only covers lego's success path end-to-end, mirroring
tests/test_extract_requires_studio_ops.py::test_extract_endpoint_succeeds_with_studio_ops_and_real_source,
including the optional waveform-region interval the web UI's Lego button
forwards as repainting_start/repainting_end.
"""

from __future__ import annotations


from fastapi.testclient import TestClient

from server import storage

from helpers import wait_for_job


def test_lego_endpoint_succeeds_with_studio_ops_and_region(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Lego Flow"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    # A fake waveform region, standing in for the drag-to-select region the
    # web UI's Lego button optionally forwards (SPEC.md sec 4.4: lego's
    # "optional interval" reuses repaint's repainting_start/repainting_end).
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "lego",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "drums",
            "repainting_start": 1.5,
            "repainting_end": 4.0,
        },
    )
    assert resp.status_code == 200
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    take_id = job["take_id"]
    assert take_id
    take = next(
        t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == take_id
    )
    assert take["task_type"] == "lego"
    assert take["parent_take_id"] == source_take_id
    assert take["track_name"] == "drums"


def test_lego_endpoint_succeeds_without_a_region(client: TestClient) -> None:
    """lego's interval is optional (SPEC.md sec 4.4) -- a request that omits
    repainting_start/repainting_end entirely must still succeed."""
    project = client.post("/api/projects", json={"title": "Lego No Region"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "lego",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "bass",
        },
    )
    assert resp.status_code == 200
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    take_id = job["take_id"]
    take = next(
        t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == take_id
    )
    assert take["task_type"] == "lego"
    assert take["parent_take_id"] == source_take_id
    assert take["track_name"] == "bass"
