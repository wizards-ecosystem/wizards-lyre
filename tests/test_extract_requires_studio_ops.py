"""SPEC.md sec 8.1: reject extract|lego|complete unless dit_profile is
'studio_ops'.

Two layers make up this contract in the current codebase:

1. `server.jobs._resolve_dit_profile` is the actual sec 8.1 enforcement: an
   extract/lego/complete job with no dit_profile is coerced to 'studio_ops',
   an explicit non-'studio_ops' profile is rejected, and 'studio_ops' itself
   is accepted. That's exercised directly below.
2. SPEC.md sec 12 additionally locks phase order ("Do not skip ahead"):
   extract/lego/complete are Phase 3 (base-model swap) work, each needing UI
   this Phase 1 build doesn't have (source selection, base-swap
   confirmation). `server.jobs.enqueue_job` therefore phase-gates all three
   actions outright in Phase 1, *before* dit_profile is even considered --
   so POST /api/projects/{id}/jobs rejects extract with 'studio_ops' too,
   for now. That gate is deliberate (see
   tests/test_phase1_api.py::test_phase_gated_actions_rejected_until_their_phase)
   and is not something this task should loosen -- doing so would implement
   Phase 3 ahead of its UI. Enabling extract/lego/complete later is a
   one-line PHASE_GATED_ACTIONS edit; layer (1) below is what makes that
   edit safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs as jobs_module
from server import storage
from server.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")
    with TestClient(app) as c:
        yield c


def test_extract_with_iterate_profile_is_rejected() -> None:
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_dit_profile("extract", "iterate", "iterate")


def test_extract_with_studio_ops_profile_is_accepted() -> None:
    assert jobs_module._resolve_dit_profile("extract", "studio_ops", "iterate") == "studio_ops"
    # an omitted profile is coerced to studio_ops, not silently rejected
    assert jobs_module._resolve_dit_profile("extract", None, "iterate") == "studio_ops"


def test_extract_endpoint_is_phase_gated_regardless_of_profile(client: TestClient) -> None:
    """Phase 1 (SPEC.md sec 12) hasn't shipped extract's UI yet, so the HTTP
    endpoint rejects it outright -- for both a wrong and a correct
    dit_profile -- rather than letting a well-formed request queue a job the
    UI can't drive. See module docstring."""
    project = client.post("/api/projects", json={"title": "Extract Gate"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "extract", "dit_profile": "iterate", "track_name": "vocals"},
    )
    assert resp.status_code == 400

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "extract", "dit_profile": "studio_ops", "track_name": "vocals"},
    )
    assert resp.status_code == 400
    assert "not available yet" in resp.json()["detail"]
