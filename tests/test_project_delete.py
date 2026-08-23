"""SPEC.md sec 9.1 Library: "Delete (confirm)". Covers server.storage.delete_project
/ DELETE /api/projects/{id} -- successful deletion, unknown ids, path-jail
resistance, and that deleting a project cancels its own queued jobs without
touching another project's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs, storage
from server.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")
    # No worker thread here (unlike tests/test_generate_take_flow.py) --
    # jobs stay `queued` so pending-job cancellation can be observed
    # directly instead of racing a background drain.
    with TestClient(app) as c:
        yield c


def test_delete_project_removes_it_from_library_and_disk(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Delete Me"}).json()
    project_id = project["id"]
    pdir = storage.project_dir(project_id)
    assert pdir.is_dir()

    resp = client.delete(f"/api/projects/{project_id}")
    assert resp.status_code == 204

    assert not pdir.exists()
    listed = client.get("/api/projects").json()
    assert all(p["id"] != project_id for p in listed)

    detail = client.get(f"/api/projects/{project_id}")
    assert detail.status_code == 404


def test_delete_unknown_project_returns_404(client: TestClient) -> None:
    resp = client.delete("/api/projects/does-not-exist")
    assert resp.status_code == 404

    with pytest.raises(storage.ProjectNotFound):
        storage.delete_project("does-not-exist")


def test_delete_project_path_jail(client: TestClient) -> None:
    # ".." as a project id must never let delete_project rmtree anything
    # outside projects_dir() -- same jail every other write in storage.py
    # goes through (SPEC.md sec 8.1/11). This is the exception this test is
    # really about: real HTTP clients (httpx here, same as a browser) collapse
    # a ".."  path segment via RFC 3986 normalization before the request is
    # ever sent, so the server-side jail can never actually see a raw ".."
    # over HTTP -- only a caller inside the process (server.jobs, a future
    # endpoint) can hand storage.delete_project an unnormalized id.
    with pytest.raises(storage.PathJailError):
        storage.delete_project("..")

    # Whatever route the client-normalized path happens to land on (a 404 if
    # nothing matches, a 405 if web/dist is built and the SPA's catch-all
    # static mount claims the path but only serves GET/HEAD, mirroring
    # tests/test_phase1_api.py's test_path_jail_rejects_escape_via_api for
    # GET) -- it must never be a successful delete.
    resp = client.delete("/api/projects/..")
    assert resp.status_code in (400, 404, 405)
    # projects_dir() itself must survive an attempted ".." delete
    assert storage.config.projects_dir().exists()


def test_delete_leaves_other_projects_untouched(client: TestClient) -> None:
    keep = client.post("/api/projects", json={"title": "Keep Me"}).json()
    doomed = client.post("/api/projects", json={"title": "Doomed"}).json()

    resp = client.delete(f"/api/projects/{doomed['id']}")
    assert resp.status_code == 204

    assert storage.project_dir(keep["id"]).is_dir()
    listed = client.get("/api/projects").json()
    ids = {p["id"] for p in listed}
    assert keep["id"] in ids
    assert doomed["id"] not in ids


def test_delete_cancels_the_projects_own_queued_jobs_only(client: TestClient) -> None:
    doomed = client.post("/api/projects", json={"title": "Doomed"}).json()
    survivor = client.post("/api/projects", json={"title": "Survivor"}).json()

    doomed_job = client.post(
        f"/api/projects/{doomed['id']}/jobs",
        json={"action": "generate", "seed": -1},
    ).json()
    survivor_job = client.post(
        f"/api/projects/{survivor['id']}/jobs",
        json={"action": "generate", "seed": -1},
    ).json()
    assert doomed_job["status"] == "queued"
    assert survivor_job["status"] == "queued"

    resp = client.delete(f"/api/projects/{doomed['id']}")
    assert resp.status_code == 204

    cancelled = client.get(f"/api/jobs/{doomed_job['id']}").json()
    assert cancelled["status"] == "error"
    assert cancelled["error"]

    # Deleting one project's queue backlog must never touch another
    # project's still-actionable job (reviewer-flagged).
    untouched = client.get(f"/api/jobs/{survivor_job['id']}").json()
    assert untouched["status"] == "queued"

    # The cancelled job can no longer be claimed by a worker draining the
    # queue -- it was `queued`, now it's a terminal `error`.
    assert jobs.claim_next_queued_job()["id"] == survivor_job["id"]


def test_cancel_queued_jobs_for_project_returns_cancelled_ids(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Cancel Return"}).json()
    job_a = client.post(
        f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
    ).json()
    job_b = client.post(
        f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
    ).json()

    cancelled_ids = jobs.cancel_queued_jobs_for_project(project["id"])
    assert set(cancelled_ids) == {job_a["id"], job_b["id"]}

    # calling it again once nothing is queued is a no-op, not an error
    assert jobs.cancel_queued_jobs_for_project(project["id"]) == []
