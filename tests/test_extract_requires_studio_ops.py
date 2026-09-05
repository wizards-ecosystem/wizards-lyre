"""SPEC.md sec 8.1: reject extract unless dit_profile is 'studio_ops'.

Several layers make up this contract in the current codebase:

1. `server.jobs._resolve_dit_profile` is the actual sec 8.1 enforcement: an
   extract/lego/complete job with no dit_profile is coerced to 'studio_ops',
   an explicit non-'studio_ops' profile is rejected, and 'studio_ops' itself
   is accepted. That's exercised directly below.
2. `server.jobs._resolve_track_name` enforces that extract/lego/complete
   also carry a real target track (SPEC.md sec 4.4: it maps onto ACE-Step's
   task-specific `instruction` field) -- missing, non-string, or
   whitespace-only input is rejected at enqueue time, not just guarded by
   the web UI's disabled-button state.
3. Phase 3 (SPEC.md sec 12) has landed for extract, lego, and complete alike
   now that the web UI has a base-model-swap confirmation/loading workflow
   (SPEC.md sec 4.3/9.2) shared by all three, so POST /api/projects/{id}/jobs
   actually queues and runs a well-formed request for any of them. This
   module only covers extract's own dit_profile/track_name enforcement, the
   layer (1)/(2) contracts above, plus (below) that the base-model swap
   itself is published as soon as it happens rather than only once the whole
   job finishes -- see tests/test_lego_flow.py and tests/test_complete_flow.py
   for lego/complete's own success-path coverage.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import jobs as jobs_module
from server import storage


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


def test_resolve_track_name_requires_non_empty_string() -> None:
    """Direct unit coverage for `_resolve_track_name` (reviewer-flagged: the
    web UI disables Extract until a track name is typed, but that's only a
    client-side convenience -- a request posted straight to the HTTP API
    must be rejected the same way for missing, null, non-string, or
    whitespace-only input)."""
    for bad_body in (
        {},
        {"track_name": None},
        {"track_name": ""},
        {"track_name": "   "},
        {"track_name": 123},
        {"track_name": ["vocals"]},
    ):
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_track_name("extract", bad_body)

    # leading/trailing whitespace is trimmed, not rejected
    assert jobs_module._resolve_track_name("extract", {"track_name": "  vocals  "}) == "vocals"

    # actions that don't route through studio_ops never require one
    assert jobs_module._resolve_track_name("generate", {}) is None
    assert jobs_module._resolve_track_name("cover", {"source_take_id": "x"}) is None


def test_extract_endpoint_rejects_missing_or_blank_track_name(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Extract Track Name Gate"}).json()
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

    base_body = {
        "action": "extract",
        "dit_profile": "studio_ops",
        "source_take_id": source_take_id,
    }
    # None/blank pass JobBody's `Optional[str]` validation but must still be
    # rejected by enqueue_job's own studio_ops requirement.
    for bad_track_name in (None, "", "   "):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={**base_body, "track_name": bad_track_name},
        )
        assert resp.status_code == 400, bad_track_name
        assert "track_name" in resp.json()["detail"]

    # a non-string track_name never reaches enqueue_job at all -- JobBody's
    # `Optional[str]` rejects it at the request-validation layer first.
    resp = client.post(f"/api/projects/{project_id}/jobs", json={**base_body, "track_name": 123})
    assert resp.status_code == 422

    # omitted entirely, not just falsy
    resp = client.post(f"/api/projects/{project_id}/jobs", json=base_body)
    assert resp.status_code == 400
    assert "track_name" in resp.json()["detail"]

    # no job row is left behind for any of the rejected requests
    assert all(j["action"] != "extract" for j in client.get("/api/jobs").json())


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
    gen = wait_for_job(client, gen_resp.json()["id"])
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
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    take_id = job["take_id"]
    assert take_id
    take = next(
        t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == take_id
    )
    assert take["task_type"] == "extract"
    assert take["parent_take_id"] == source_take_id
    assert take["track_name"] == "vocals"


def test_worker_status_reflects_studio_ops_swap_before_job_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 4.3: swapping the loaded DiT profile happens once, up
    front, inside a job -- not repeated for the rest of that job's
    (potentially long) generation. `/api/health`'s `dit_loaded` must flip to
    'studio_ops' as soon as the swap itself finishes, not only after the
    whole extract job completes (reviewer-flagged: publishing only at job
    end makes a "loading base model..." UI state indistinguishable from
    ordinary in-progress extraction for the job's entire duration)."""
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("LYRE_DB_PATH", str(tmp_path / "lyre.db"))
    monkeypatch.setenv("LYRE_WORKER", "mock")

    jobs_module.init_db()
    project = storage.create_project(title="Swap Publish Test")
    project_id = project["id"]

    gen_job = jobs_module.enqueue_job(project_id, {"action": "generate"})
    jobs_module.run_claimed_job(jobs_module.claim_next_queued_job())
    gen = jobs_module.get_job(gen_job["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    jobs_module.publish_worker_status(
        True, "worker: 'iterate' DiT + LM currently loaded", "iterate"
    )
    assert jobs_module.get_worker_status()["loaded_dit_profile"] == "iterate"

    import worker.mock_worker as mock_worker_module

    real_run_job = mock_worker_module.run_job
    swapped = threading.Event()
    proceed = threading.Event()

    def blocking_run_job(job, plan, take_id, take_dir, on_dit_loaded=None):
        def _on_loaded(profile):
            if on_dit_loaded is not None:
                on_dit_loaded(profile)
            # The (simulated) base-model swap has finished -- block here,
            # standing in for the extraction inference that follows it, so
            # the test can observe published status while the job is still
            # `running`.
            swapped.set()
            proceed.wait(timeout=5)

        return real_run_job(job, plan, take_id, take_dir, on_dit_loaded=_on_loaded)

    monkeypatch.setattr(mock_worker_module, "run_job", blocking_run_job)

    extract_job = jobs_module.enqueue_job(
        project_id,
        {
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals",
        },
    )
    claimed = jobs_module.claim_next_queued_job()

    worker_thread = threading.Thread(target=jobs_module.run_claimed_job, args=(claimed,))
    worker_thread.start()
    try:
        assert swapped.wait(timeout=5)
        assert jobs_module.get_job(extract_job["id"])["status"] == "running"
        status = jobs_module.get_worker_status()
        assert status is not None
        assert status["loaded_dit_profile"] == "studio_ops"
    finally:
        proceed.set()
        worker_thread.join(timeout=5)

    final = jobs_module.get_job(extract_job["id"])
    assert final["status"] == "done", final.get("error")
