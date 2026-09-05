"""SPEC.md sec 7 `lyrics.lrc` / sec 12 Phase 4: LRC download endpoint.

`GET /api/projects/{id}/takes/{take_id}/lrc` 404s when no `lyrics.lrc` was
written for a take, and serves it as plain text when one exists.
`worker.mock_worker` never produces real timestamps (it has no ACE-Step to
call `get_lyric_timestamp` against -- see
tests/test_acestep_worker_adapter.py for that coverage), so this drives a
take through the same mocked-worker generate flow as
tests/test_generate_take_flow.py and then writes a trivial fixture
`lyrics.lrc` directly via `storage.write_take_lrc`, which is enough to
exercise the endpoint itself end to end.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


def _make_take(client: TestClient) -> tuple[str, str]:
    project = client.post("/api/projects", json={"title": "LRC Endpoint Test"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone"},
    )
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    return project_id, job["take_id"]


def test_lrc_endpoint_404s_without_a_lrc_file(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    # The mock worker never produces one (has_lrc is False in its meta.json)
    # -- the take list should agree, so the UI never even shows the link.
    takes = client.get(f"/api/projects/{project_id}/takes").json()
    take = next(t for t in takes if t["id"] == take_id)
    assert take["has_lrc"] is False

    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/lrc")
    assert resp.status_code == 404


def test_lrc_endpoint_serves_file_when_present(client: TestClient) -> None:
    project_id, take_id = _make_take(client)
    fixture_lrc = "[00:00.00]We were born to run\n[00:02.50]Down the neon highway\n"
    storage.write_take_lrc(project_id, take_id, fixture_lrc)

    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/lrc")
    assert resp.status_code == 200
    assert resp.text == fixture_lrc
