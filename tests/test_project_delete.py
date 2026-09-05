"""SPEC.md sec 9.1 Library: "Delete (confirm)". Covers server.storage.delete_project
/ DELETE /api/projects/{id} -- successful deletion, unknown ids, path-jail
resistance, that deleting a project cancels its own queued jobs without
touching another project's, that a concurrent enqueue can never land once
deletion has begun (server.jobs.begin_project_deletion's tombstone), and
that a filesystem failure during deletion rolls the cancellation back
instead of leaving a still-existing project with irreversibly cancelled
jobs (server.jobs.abort_project_deletion).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs, storage
from server.app import app


def test_delete_project_removes_it_from_library_and_disk(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Delete Me"}).json()
    project_id = project["id"]
    pdir = storage.project_dir(project_id)
    assert pdir.is_dir()

    resp = api_client.delete(f"/api/projects/{project_id}")
    assert resp.status_code == 204

    assert not pdir.exists()
    listed = api_client.get("/api/projects").json()
    assert all(p["id"] != project_id for p in listed)

    detail = api_client.get(f"/api/projects/{project_id}")
    assert detail.status_code == 404


def test_delete_unknown_project_returns_404(api_client: TestClient) -> None:
    resp = api_client.delete("/api/projects/does-not-exist")
    assert resp.status_code == 404

    with pytest.raises(storage.ProjectNotFound):
        storage.delete_project("does-not-exist")


def test_delete_project_path_jail(api_client: TestClient) -> None:
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

    # Whatever route the api_client-normalized path happens to land on (a 404 if
    # nothing matches, a 405 if web/dist is built and the SPA's catch-all
    # static mount claims the path but only serves GET/HEAD, mirroring
    # tests/test_phase1_api.py's test_path_jail_rejects_escape_via_api for
    # GET) -- it must never be a successful delete.
    resp = api_client.delete("/api/projects/..")
    assert resp.status_code in (400, 404, 405)
    # projects_dir() itself must survive an attempted ".." delete
    assert storage.config.projects_dir().exists()


def test_delete_leaves_other_projects_untouched(api_client: TestClient) -> None:
    keep = api_client.post("/api/projects", json={"title": "Keep Me"}).json()
    doomed = api_client.post("/api/projects", json={"title": "Doomed"}).json()

    resp = api_client.delete(f"/api/projects/{doomed['id']}")
    assert resp.status_code == 204

    assert storage.project_dir(keep["id"]).is_dir()
    listed = api_client.get("/api/projects").json()
    ids = {p["id"] for p in listed}
    assert keep["id"] in ids
    assert doomed["id"] not in ids


def test_delete_cancels_the_projects_own_queued_jobs_only(api_client: TestClient) -> None:
    doomed = api_client.post("/api/projects", json={"title": "Doomed"}).json()
    survivor = api_client.post("/api/projects", json={"title": "Survivor"}).json()

    doomed_job = api_client.post(
        f"/api/projects/{doomed['id']}/jobs",
        json={"action": "generate", "seed": -1},
    ).json()
    survivor_job = api_client.post(
        f"/api/projects/{survivor['id']}/jobs",
        json={"action": "generate", "seed": -1},
    ).json()
    assert doomed_job["status"] == "queued"
    assert survivor_job["status"] == "queued"

    resp = api_client.delete(f"/api/projects/{doomed['id']}")
    assert resp.status_code == 204

    cancelled = api_client.get(f"/api/jobs/{doomed_job['id']}").json()
    assert cancelled["status"] == "error"
    assert cancelled["error"]

    # Deleting one project's queue backlog must never touch another
    # project's still-actionable job (reviewer-flagged).
    untouched = api_client.get(f"/api/jobs/{survivor_job['id']}").json()
    assert untouched["status"] == "queued"

    # The cancelled job can no longer be claimed by a worker draining the
    # queue -- it was `queued`, now it's a terminal `error`.
    assert jobs.claim_next_queued_job()["id"] == survivor_job["id"]


def test_begin_project_deletion_returns_cancelled_ids(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Cancel Return"}).json()
    job_a = api_client.post(
        f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
    ).json()
    job_b = api_client.post(
        f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
    ).json()

    cancelled_ids = jobs.begin_project_deletion(project["id"])
    assert set(cancelled_ids) == {job_a["id"], job_b["id"]}

    # A tombstone now exists -- calling it again (a second concurrent DELETE,
    # or a retry) must not silently re-cancel/re-tombstone, it must reject.
    with pytest.raises(jobs.ProjectDeletionConflict):
        jobs.begin_project_deletion(project["id"])


def test_enqueue_is_rejected_once_project_deletion_has_begun(api_client: TestClient) -> None:
    # Reviewer-flagged race: a concurrent enqueue_job that lands between
    # begin_project_deletion's cancel step and storage.delete_project's
    # rmtree must never succeed in inserting an orphaned queued job -- the
    # tombstone begin_project_deletion writes is what enqueue_job checks
    # atomically before inserting (server.jobs.enqueue_job).
    project = api_client.post("/api/projects", json={"title": "Deleting"}).json()
    jobs.begin_project_deletion(project["id"])

    resp = api_client.post(
        f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
    )
    assert resp.status_code == 400

    # No job row leaked in for this project despite the rejected request.
    recent = jobs.list_recent_jobs(limit=50)
    assert all(j["project_id"] != project["id"] for j in recent)


def test_concurrent_enqueue_never_orphans_a_queued_job_past_deletion(
    api_client: TestClient,
) -> None:
    # Reviewer-flagged race, exercised under real concurrent execution rather
    # than by hand-sequencing calls: hammer server.jobs.enqueue_job from
    # several threads while a deletion runs concurrently, and assert the
    # invariant the tombstone in begin_project_deletion/enqueue_job exists to
    # guarantee -- once storage.delete_project has removed the project
    # directory, no `queued` (or `running`) row for it can still exist,
    # whichever way the enqueue/delete calls happened to interleave.
    project = api_client.post("/api/projects", json={"title": "Race"}).json()
    project_id = project["id"]

    # Seed one queued job synchronously. The final `remaining` assert needs
    # at least one row for begin_project_deletion to have cancelled, and the
    # hammer threads below are not guaranteed to land an enqueue before the
    # deletion starts racing them -- thread startup can take longer than the
    # sleep below on a loaded runner, in which case every enqueue is
    # (correctly) rejected by the tombstone and `remaining` comes back
    # empty. The seed covers the "enqueue landed entirely before the
    # deletion transaction" interleave deterministically; the hammer still
    # covers the racing interleaves.
    jobs.enqueue_job(project_id, {"action": "generate", "seed": -1})

    unexpected_errors: list[BaseException] = []

    def enqueue_repeatedly() -> None:
        for _ in range(50):
            try:
                jobs.enqueue_job(project_id, {"action": "generate", "seed": -1})
            except jobs.JobError:
                pass  # rejected once the tombstone is in place -- expected
            except storage.ProjectNotFound:
                pass  # project already gone -- expected once delete finishes
            except BaseException as exc:
                unexpected_errors.append(exc)

    threads = [threading.Thread(target=enqueue_repeatedly) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.01)  # let a few enqueues land before the delete starts racing them
    jobs.begin_project_deletion(project_id)
    storage.delete_project(project_id)
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert unexpected_errors == []
    remaining = [j for j in jobs.list_recent_jobs(limit=1000) if j["project_id"] == project_id]
    assert remaining, "expected at least the jobs begin_project_deletion cancelled"
    assert all(j["status"] == "error" for j in remaining)


def test_delete_waits_for_inflight_save_and_directory_stays_absent(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save past its existence check cannot recreate a deleted project."""
    project = api_client.post("/api/projects", json={"title": "Save Race"}).json()
    project_id = project["id"]
    pdir = storage.project_dir(project_id)
    save_paused = threading.Event()
    resume_save = threading.Event()
    real_write_json = storage.jsonio._write_json

    def paused_write(path: Path, data: dict) -> None:
        if path == storage.plan_json_path(project_id):
            save_paused.set()
            assert resume_save.wait(timeout=10)
        real_write_json(path, data)

    monkeypatch.setattr(storage.jsonio, "_write_json", paused_write)
    save_thread = threading.Thread(target=storage.save_plan, args=(project_id, {"query": "new"}))
    save_thread.start()
    assert save_paused.wait(timeout=10)

    delete_result: list[int] = []

    def delete() -> None:
        delete_result.append(api_client.delete(f"/api/projects/{project_id}").status_code)

    delete_thread = threading.Thread(target=delete)
    delete_thread.start()
    time.sleep(0.1)
    assert delete_thread.is_alive(), "DELETE must wait for the in-flight writer"
    resume_save.set()
    save_thread.join(timeout=10)
    delete_thread.join(timeout=10)

    assert not save_thread.is_alive()
    assert not delete_thread.is_alive()
    assert delete_result == [204]
    assert not pdir.exists()


def test_filesystem_failure_rolls_back_job_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reviewer-flagged: if shutil.rmtree fails after queued jobs were
    # already cancelled, the project (which still exists on disk) must not
    # be left with irreversibly cancelled jobs.
    #
    # raise_server_exceptions=False here (unlike the module's `api_client`
    # fixture) so an unhandled OSError from the simulated disk failure comes
    # back as a real 500 response to inspect, the way a production ASGI
    # server's ServerErrorMiddleware would return it, instead of TestClient
    # re-raising it into the test for debugging.
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("LYRE_DB_PATH", str(tmp_path / "lyre.db"))
    monkeypatch.setenv("LYRE_WORKER", "mock")
    with TestClient(app, raise_server_exceptions=False) as api_client:
        project = api_client.post("/api/projects", json={"title": "Rmtree Fails"}).json()
        job = api_client.post(
            f"/api/projects/{project['id']}/jobs", json={"action": "generate", "seed": -1}
        ).json()

        real_rmtree = storage.projects.shutil.rmtree

        def _boom(*_args, **_kwargs):
            raise OSError("simulated disk failure")

        storage.projects.shutil.rmtree = _boom
        try:
            resp = api_client.delete(f"/api/projects/{project['id']}")
        finally:
            storage.projects.shutil.rmtree = real_rmtree
        assert resp.status_code == 500

        # The project still exists (rmtree never actually ran) and is still
        # listed...
        pdir = storage.project_dir(project["id"])
        assert pdir.is_dir()
        listed = api_client.get("/api/projects").json()
        assert any(p["id"] == project["id"] for p in listed)

        # ...and its job was restored to `queued`, not left `error`.
        restored = api_client.get(f"/api/jobs/{job['id']}").json()
        assert restored["status"] == "queued"
        assert restored["error"] is None

        # The tombstone was removed too, so a real (non-failing) retry succeeds.
        resp = api_client.delete(f"/api/projects/{project['id']}")
        assert resp.status_code == 204
        assert not pdir.exists()
