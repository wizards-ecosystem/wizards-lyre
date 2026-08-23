"""SPEC.md sec 4.4/12: LoRA style-pack training end to end against the
mocked worker -- enqueue `train_lora` with 8+ fake source takes, poll to
`done`, and assert a lora shows up under GET /api/projects/{id}/loras.
Also covers the '8+ songs' enqueue-time rejection (SPEC.md sec 4.4).

Real ACE-Step call-contract coverage (DatasetBuilder -> preprocess_to_tensors
-> LoRATrainer.train_from_preprocessed) lives in
tests/test_acestep_worker_adapter.py instead, same split as every other
action.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app
from worker import mock_worker
from worker.run_worker import run_loop

MIN_LORA_SOURCES = 8


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    # enqueue_job only inserts a `queued` row (SPEC.md sec 5); this thread
    # stands in for the dedicated worker/run_worker.py process that drains
    # the same SQLite queue in production.
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


def _make_take_sources(client: TestClient, project_id: str, count: int) -> list[str]:
    """Generate `count` real (mocked) takes to use as train_lora sources."""
    take_ids = []
    for _ in range(count):
        resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")
        take_ids.append(job["take_id"])
    return take_ids


def test_train_lora_end_to_end_with_mocked_worker(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Style Pack Project"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    source_take_ids = _make_take_sources(client, project_id, MIN_LORA_SOURCES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "train_lora",
            "name": "dreamy synthwave pack",
            "source_take_ids": source_take_ids,
        },
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"], timeout=10.0)
    assert job["status"] == "done", job.get("error")

    lora_id = job["lora_id"]
    assert lora_id

    resp = client.get(f"/api/projects/{project_id}/loras")
    assert resp.status_code == 200
    loras = resp.json()
    assert len(loras) == 1
    lora = loras[0]
    assert lora["id"] == lora_id
    assert lora["name"] == "dreamy synthwave pack"
    assert lora["source_take_count"] == MIN_LORA_SOURCES

    # meta.json actually landed on disk under projects/<id>/loras/<lora_id>/
    lora_dir = storage.lora_dir(project_id, lora_id)
    assert (lora_dir / "meta.json").exists()
    assert lora_dir.is_relative_to(storage.config.projects_dir())

    # the mocked worker wrote a fake adapter file somewhere under lora_dir
    assert any(lora_dir.rglob("*")), "expected the mocked trainer to write some file"


def test_train_lora_rejects_fewer_than_minimum_sources_at_enqueue_time(client: TestClient) -> None:
    """SPEC.md sec 4.4: 'Style pack | LoRA train / load | 8+ songs' -- must
    be rejected before a job is ever queued, not discovered later inside the
    worker."""
    project = client.post("/api/projects", json={"title": "Too Few Sources"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    source_take_ids = _make_take_sources(client, project_id, MIN_LORA_SOURCES - 1)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "train_lora",
            "name": "too few",
            "source_take_ids": source_take_ids,
        },
    )
    assert resp.status_code == 400
    assert "8" in resp.json()["detail"]

    # no job row left behind for the rejected request
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())

    # no lora exists either
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_rejects_missing_name_at_enqueue_time(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "No Name"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )
    source_take_ids = _make_take_sources(client, project_id, MIN_LORA_SOURCES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": source_take_ids},
    )
    assert resp.status_code == 400
    assert "name" in resp.json()["detail"]


def test_resolve_lora_sources_requires_minimum_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct unit coverage for _resolve_lora_sources (mirrors how
    test_resolve_source_audio_requires_a_real_source covers
    _resolve_source_audio)."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    from server import jobs as jobs_module

    project = storage.create_project(title="Unit Test")
    project_id = project["id"]

    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora_sources(project_id, {})

    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora_sources(
            project_id, {"source_take_ids": ["does-not-exist"] * MIN_LORA_SOURCES}
        )


def test_jobs_list_filters_by_project_and_action(client: TestClient) -> None:
    """GET /api/jobs contract the web UI's training-recovery relies on: the
    optional project_id/action filters narrow the listing server-side
    (before the limit), and omitting them keeps the old unfiltered shape."""
    project_a = client.post("/api/projects", json={"title": "Trains A Pack"}).json()
    project_b = client.post("/api/projects", json={"title": "Just Generates"}).json()
    for project in (project_a, project_b):
        client.put(
            f"/api/projects/{project['id']}/plan",
            json={**storage.default_plan(), "caption": "reference take"},
        )

    source_take_ids = _make_take_sources(client, project_a["id"], MIN_LORA_SOURCES)
    resp = client.post(
        f"/api/projects/{project_a['id']}/jobs",
        json={
            "action": "train_lora",
            "name": "filter me",
            "source_take_ids": source_take_ids,
        },
    )
    train_job = _wait_for_job(client, resp.json()["id"], timeout=10.0)
    assert train_job["status"] == "done", train_job.get("error")

    # Another project's jobs must not leak into this project's listing.
    client.post(f"/api/projects/{project_b['id']}/jobs", json={"action": "generate"})

    train_only = client.get(
        "/api/jobs",
        params={"project_id": project_a["id"], "action": "train_lora"},
    ).json()
    assert [j["id"] for j in train_only] == [train_job["id"]]
    assert train_only[0]["project_id"] == project_a["id"]
    assert train_only[0]["action"] == "train_lora"

    project_jobs = client.get("/api/jobs", params={"project_id": project_a["id"]}).json()
    assert project_jobs, "expected this project's jobs"
    assert all(j["project_id"] == project_a["id"] for j in project_jobs)
    assert any(j["action"] == "train_lora" for j in project_jobs)
    assert any(j["action"] == "generate" for j in project_jobs)

    generate_only = client.get("/api/jobs", params={"action": "generate"}).json()
    assert generate_only, "expected generate jobs"
    assert all(j["action"] == "generate" for j in generate_only)
    # ...across both projects -- `action` alone does not scope to a project.
    assert {j["project_id"] for j in generate_only} == {project_a["id"], project_b["id"]}

    # No filters: unchanged old contract -- recent jobs across all projects.
    everything = client.get("/api/jobs").json()
    assert {j["id"] for j in train_only} <= {j["id"] for j in everything}
    assert {j["project_id"] for j in everything} == {project_a["id"], project_b["id"]}


def test_queued_train_lora_job_is_recoverable_via_filtered_jobs_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery contract after a mid-training page refresh: the UI finds the
    project's still-active training via GET /api/jobs?project_id&action=
    train_lora. No worker thread drains the queue here, so the job stays
    `queued` -- exactly the state a refresh must be able to rediscover --
    and its `lora_id` is still null (the pack id only exists once training
    allocates one), which the UI renders as an in-flight entry rather than
    a pack."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    with TestClient(app) as client:
        project = storage.create_project(title="Mid-Training Refresh")
        project_id = project["id"]

        # Source takes on disk without running jobs through the (absent)
        # worker -- enqueue-time train_lora validation only needs their
        # audio paths to resolve (server.jobs._resolve_lora_sources).
        take_ids: list[str] = []
        for _ in range(MIN_LORA_SOURCES):
            take_id, tdir = storage.allocate_take_dir(project_id)
            mock_worker._write_silent_wav(tdir / "mix.wav")
            take_ids.append(take_id)

        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": "train_lora",
                "name": "survives a refresh",
                "source_take_ids": take_ids,
            },
        )
        assert resp.status_code == 200
        queued = resp.json()
        assert queued["status"] == "queued"

        jobs = client.get(
            "/api/jobs",
            params={"project_id": project_id, "action": "train_lora"},
        ).json()
        assert [j["id"] for j in jobs] == [queued["id"]]
        assert jobs[0]["status"] == "queued"
        assert jobs[0]["lora_id"] is None
        assert jobs[0]["error"] is None
        # ...and it is still visible in the unfiltered recent listing too.
        assert any(
            j["id"] == queued["id"]
            for j in client.get("/api/jobs").json()
        )
