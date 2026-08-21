"""SPEC.md sec 8.1: reject extract unless dit_profile is 'studio_ops'.

Two layers make up this contract in the current codebase:

1. `server.jobs._resolve_dit_profile` is the actual sec 8.1 enforcement: an
   extract/lego/complete job with no dit_profile is coerced to 'studio_ops',
   an explicit non-'studio_ops' profile is rejected, and 'studio_ops' itself
   is accepted. That's exercised directly below.
2. Phase 3 (SPEC.md sec 12) has landed for extract now that the web UI has a
   base-model-swap confirmation/loading workflow (SPEC.md sec 4.3/9.2), so
   POST /api/projects/{id}/jobs actually queues and runs a well-formed
   extract request. lego/complete remain phase-gated behind their own
   follow-up UI (see
   tests/test_phase1_api.py::test_phase_gated_actions_rejected_until_their_phase);
   this module only covers extract's own dit_profile enforcement, which is
   the layer (1) contract above.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs as jobs_module
from server import storage
from server.app import app
from worker.run_worker import run_loop


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    # enqueue_job only inserts a `queued` row (SPEC.md sec 5); this thread
    # stands in for the dedicated worker/run_worker.py process that drains
    # the same SQLite queue in production.
    import threading

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


def test_extract_with_iterate_profile_is_rejected() -> None:
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_dit_profile("extract", "iterate", "iterate")


def test_extract_with_studio_ops_profile_is_accepted() -> None:
    assert jobs_module._resolve_dit_profile("extract", "studio_ops", "iterate") == "studio_ops"
    # an omitted profile is coerced to studio_ops, not silently rejected
    assert jobs_module._resolve_dit_profile("extract", None, "iterate") == "studio_ops"


def test_extract_endpoint_rejects_non_studio_ops_profile(client: TestClient) -> None:
    """An explicit dit_profile mismatch still 400s -- but now because of
    _resolve_dit_profile's studio_ops enforcement, not the phase gate."""
    project = client.post("/api/projects", json={"title": "Extract Gate"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "extract", "dit_profile": "iterate", "track_name": "vocals"},
    )
    assert resp.status_code == 400
    assert "requires dit_profile='studio_ops'" in resp.json()["detail"]


def test_extract_endpoint_succeeds_with_studio_ops_and_real_source(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Extract Flow"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = _wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals",
        },
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    take_id = job["take_id"]
    assert take_id
    take = next(
        t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == take_id
    )
    assert take["task_type"] == "extract"
    assert take["parent_take_id"] == source_take_id
    assert take["track_name"] == "vocals"
