"""Phase 4 (mocked): LoRA storage + job scaffolding. SPEC.md sec 4.4 ('Style
pack | LoRA train / load | 8+ songs') and sec 7 (`projects/<id>/loras/`).

This only exercises the mocked worker's `train_lora` -- see
worker/mock_worker.py and server/jobs.py's train_lora dispatch. The real
ACE-Step training call is a follow-up job's responsibility; no upstream API
research happens here.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app
from worker.run_worker import run_loop

MIN_LORA_SOURCE_TAKES = 8


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

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


def _make_takes(client: TestClient, project_id: str, count: int) -> list[str]:
    take_ids: list[str] = []
    for _ in range(count):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "dit_profile": "iterate", "seed": -1},
        )
        assert resp.status_code == 200
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")
        take_ids.append(job["take_id"])
    return take_ids


def test_train_lora_job_completes_and_lists(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Style Pack"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "My Style"},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    resp = client.get(f"/api/projects/{project_id}/loras")
    assert resp.status_code == 200
    loras = resp.json()
    assert len(loras) == 1
    lora = loras[0]
    assert lora["name"] == "My Style"
    assert lora["source_take_count"] == MIN_LORA_SOURCE_TAKES

    # adapter file lives on disk under the project's loras/ dir, jailed
    ldir = storage.lora_dir(project_id, lora["id"])
    assert (ldir / "adapter.bin").exists()
    assert (ldir / "meta.json").exists()
    assert ldir.is_relative_to(storage.config.projects_dir())


def test_train_lora_rejects_fewer_than_eight_sources(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Too Few"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES - 1)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Too Few"},
    )
    assert resp.status_code == 400
    # no job row is left behind for a rejected action
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_rejects_duplicate_source_ids(client: TestClient) -> None:
    """The '8+ songs' floor (SPEC.md sec 4.4) must not be satisfiable by
    repeating one real take_id -- distinct sources are required, not just a
    list of the required length (reviewer-flagged)."""
    project = client.post("/api/projects", json={"title": "LoRA Duplicate Sources"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, 1)
    duplicated = take_ids * MIN_LORA_SOURCE_TAKES  # 8 entries, 1 distinct take

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": duplicated, "name": "Dupes"},
    )
    assert resp.status_code == 400
    assert "distinct" in resp.json()["detail"]
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_rejects_duplicates_mixed_with_enough_distinct_sources(
    client: TestClient,
) -> None:
    """8 raw ids but only 7 distinct (one repeated) must still be rejected --
    the floor is on distinct songs, not list length."""
    project = client.post("/api/projects", json={"title": "LoRA Mixed Dupes"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES - 1)
    submitted = take_ids + [take_ids[0]]  # 8 entries, 7 distinct
    assert len(submitted) == MIN_LORA_SOURCE_TAKES

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": submitted, "name": "Mixed Dupes"},
    )
    assert resp.status_code == 400
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_fails_cleanly_when_backend_lacks_train_lora(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production defaults to worker.acestep_worker, which this job
    deliberately does not touch (real ACE-Step LoRA training needs upstream
    research that's out of scope here). Simulate that missing-backend
    situation against the mock -- with the enqueue-time capability gate
    (test_train_lora_rejected_when_backend_capability_unsupported, below)
    forced to 'supported' so this test deterministically exercises the
    *runtime* fallback instead, not a race against that other check -- and
    assert the job fails with a clear, actionable message -- not a bare
    AttributeError -- and, since the runtime check runs before any
    directory is allocated, no orphan loras/<id>/ directory is left on disk
    (reviewer-flagged)."""
    from server import jobs as jobs_module

    project = client.post("/api/projects", json={"title": "LoRA Backend Missing"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    import worker.mock_worker as mock_worker_module

    monkeypatch.delattr(mock_worker_module, "train_lora")
    # Force the enqueue-time capability check to pass (as if the last
    # publish happened before the attribute went missing) so this test
    # isolates the runtime hasattr fallback in _run_train_lora_job, not
    # whichever of the two layers happens to win a timing race.
    jobs_module.publish_worker_capability("train_lora", True, None)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "No Backend"},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "error"
    assert "does not implement train_lora" in job["error"]

    # no orphan directory: the runtime check rejects before allocating one
    assert not storage.loras_dir(project_id).exists() or not any(
        storage.loras_dir(project_id).iterdir()
    )
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_error_meta_tracked_when_worker_fails_mid_training(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that *does* implement train_lora but blows up mid-run (e.g.
    a real future ACE-Step training failure) must still leave a tracked,
    visible entry with `error` set -- not a meta-less orphan directory that
    GET .../loras silently skips (reviewer-flagged)."""
    project = client.post("/api/projects", json={"title": "LoRA Mid Training Failure"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    import worker.mock_worker as mock_worker_module

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(mock_worker_module, "train_lora", _boom)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Boom"},
    )
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "error"
    assert "synthetic training failure" in job["error"]

    loras = client.get(f"/api/projects/{project_id}/loras").json()
    assert len(loras) == 1
    assert loras[0]["error"] and "synthetic training failure" in loras[0]["error"]


def test_train_lora_rejected_when_backend_capability_unsupported(client: TestClient) -> None:
    """Mirrors tests/test_phase1_api.py::test_quality_profile_rejected_
    without_cpu_offload_support: a real backend that hasn't wired up
    train_lora yet (SPEC.md sec 4.4 -- this job deliberately doesn't touch
    worker/acestep_worker.py) is published as a capability by
    worker/run_worker.py's _publish_train_lora_capability and enforced at
    enqueue time, 'phase-gating' the action for that backend specifically
    instead of presenting it as available and letting every request queue
    only to fail (reviewer-flagged)."""
    from server import jobs as jobs_module

    project = client.post("/api/projects", json={"title": "LoRA Capability Gate"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    # wait for the client fixture's background worker thread to publish its
    # own startup capability before overriding it below, or it can race and
    # clobber our override right back to "supported" afterward (same race
    # test_quality_profile_rejected_without_cpu_offload_support guards
    # against).
    deadline = time.time() + 5.0
    while jobs_module.get_worker_capability("train_lora") is None and time.time() < deadline:
        time.sleep(0.01)

    # simulate what worker/run_worker.py publishes for a real backend that
    # hasn't wired up train_lora yet
    jobs_module.publish_worker_capability(
        "train_lora",
        False,
        "worker backend 'worker.acestep_worker' does not implement train_lora yet",
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Gated"},
    )
    assert resp.status_code == 400
    assert "does not implement train_lora" in resp.json()["detail"]
    # no job row is left behind for a rejected action
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())
    assert client.get(f"/api/projects/{project_id}/loras").json() == []

    # once the (real) worker reports it can train after all, it's accepted
    jobs_module.publish_worker_capability("train_lora", True, None)
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Ungated"},
    )
    assert resp.status_code == 200


def test_train_lora_error_meta_source_count_matches_distinct_sources(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted request (8+ distinct ids, plus extra duplicates) that
    fails during training must report the same distinct source count in its
    error metadata that the success path would have (reviewer-flagged: error
    metadata previously counted the raw submitted list length, inflating the
    count whenever duplicates rode along with enough distinct sources)."""
    project = client.post("/api/projects", json={"title": "LoRA Dup Count Consistency"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)
    submitted = take_ids + [take_ids[0], take_ids[1]]  # 10 raw entries, 8 distinct
    assert len(submitted) == MIN_LORA_SOURCE_TAKES + 2

    import worker.mock_worker as mock_worker_module

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(mock_worker_module, "train_lora", _boom)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": submitted, "name": "Dup Count"},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "error"

    loras = client.get(f"/api/projects/{project_id}/loras").json()
    assert len(loras) == 1
    assert loras[0]["source_take_count"] == MIN_LORA_SOURCE_TAKES
