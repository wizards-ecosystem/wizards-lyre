"""Phase 4 "LoRA train / load" -- the load half (SPEC.md sec 4.4/12).

`tests/test_train_lora_flow.py`/`test_lora_scaffolding.py` cover training a
lora; this module covers applying a *trained* lora to a generate/cover/
repaint job: `server.jobs`'s `lora_id` resolution/validation,
`_resolve_dit_profile`'s studio_ops gating for a lora-attached job, and
`worker.mock_worker`/`worker.acestep_worker` recording/loading it. Real
ACE-Step call-contract coverage for `_ensure_lora_adapter` (add_lora /
set_active_lora_adapter / set_use_lora / unload_lora) lives in
tests/test_acestep_worker_adapter.py, same split as every other action.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs as jobs_module
from server import storage

from helpers import wait_for_job

MIN_LORA_SOURCE_TAKES = 8


def _make_takes(client: TestClient, project_id: str, count: int) -> list[str]:
    take_ids: list[str] = []
    for _ in range(count):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "dit_profile": "iterate"},
        )
        assert resp.status_code == 200
        job = wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")
        take_ids.append(job["take_id"])
    return take_ids


def _train_lora(client: TestClient, project_id: str, name: str = "dreamy synthwave") -> str:
    source_take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "name": name, "source_take_ids": source_take_ids},
    )
    assert resp.status_code == 200
    job = wait_for_job(client, resp.json()["id"], timeout=10.0)
    assert job["status"] == "done", job.get("error")
    lora_id = job["lora_id"]
    assert lora_id
    return lora_id


def _new_project(client: TestClient, title: str) -> str:
    project = client.post("/api/projects", json={"title": title}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )
    return project_id


def test_generate_with_lora_id_is_coerced_to_studio_ops_and_recorded_on_the_take(
    client: TestClient,
) -> None:
    project_id = _new_project(client, "Style Pack Consumer")
    lora_id = _train_lora(client, project_id)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "lora_id": lora_id},
    )
    assert resp.status_code == 200
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["dit_profile"] == "studio_ops"

    takes = client.get(f"/api/projects/{project_id}/takes").json()
    take = next(t for t in takes if t["id"] == job["take_id"])
    assert take["lora_id"] == lora_id


def test_generate_with_lora_id_and_conflicting_explicit_profile_is_rejected(
    client: TestClient,
) -> None:
    project_id = _new_project(client, "Style Pack Conflict")
    lora_id = _train_lora(client, project_id)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "lora_id": lora_id, "dit_profile": "iterate"},
    )
    assert resp.status_code == 400
    assert "studio_ops" in resp.json()["detail"]


def test_generate_with_unknown_lora_id_is_rejected(client: TestClient) -> None:
    project_id = _new_project(client, "Style Pack Unknown")

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "lora_id": "does-not-exist"},
    )
    assert resp.status_code == 400
    assert "does-not-exist" in resp.json()["detail"]


def test_generate_rejects_studio_ops_without_a_lora_attached(client: TestClient) -> None:
    """Unchanged existing behavior (SPEC.md sec 8.1): studio_ops is still
    rejected for an ordinary generate with no lora_id attached."""
    project_id = _new_project(client, "Style Pack No Lora")

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops"},
    )
    assert resp.status_code == 400
    assert "studio_ops" in resp.json()["detail"]


def test_extract_lego_complete_reject_lora_id(client: TestClient) -> None:
    """SPEC.md sec 4.4: a style-pack lora only applies to
    generate/cover/repaint (LORA_ELIGIBLE_ACTIONS) -- extract/lego/complete
    are structural editing ops that already force studio_ops for an
    unrelated reason (STUDIO_OPS_ACTIONS), so without an explicit rejection
    a lora_id attached to one of them would sail past _resolve_dit_profile's
    lora_attached branch (it only special-cases the eligible three) and
    still get its adapter path loaded, silently altering structural-editing
    output (reviewer-flagged). Rejected before source_take_id/track_name are
    even validated, so this request is otherwise minimal/invalid on purpose."""
    project_id = _new_project(client, "Style Pack Structural Ops")
    lora_id = _train_lora(client, project_id)

    for action in ("extract", "lego", "complete"):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": action, "lora_id": lora_id},
        )
        assert resp.status_code == 400, action
        assert "lora_id" in resp.json()["detail"], action

    # no job rows were left behind for any of the rejected requests
    jobs = client.get("/api/jobs").json()
    assert all(j["action"] not in ("extract", "lego", "complete") for j in jobs)


def test_train_lora_rejects_lora_id(client: TestClient) -> None:
    """train_lora ignores any adapter it would resolve -- it trains a new
    lora, it doesn't apply one -- so an attached lora_id must be rejected
    up front rather than silently ignored (reviewer-flagged)."""
    project_id = _new_project(client, "Style Pack Train Rejects Lora")
    lora_id = _train_lora(client, project_id)
    source_take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "train_lora",
            "name": "second pack",
            "source_take_ids": source_take_ids,
            "lora_id": lora_id,
        },
    )
    assert resp.status_code == 400
    assert "lora_id" in resp.json()["detail"]


def test_cover_and_repaint_are_also_lora_eligible() -> None:
    """Direct unit coverage for _resolve_dit_profile's lora_attached gating
    across all three lora-eligible actions (mirrors
    test_resolve_dit_profile_studio_ops_enforcement's per-action loop)."""
    for action in ("generate", "cover", "repaint"):
        assert (
            jobs_module._resolve_dit_profile(action, None, "iterate", lora_attached=True)
            == "studio_ops"
        )
        assert (
            jobs_module._resolve_dit_profile(action, "studio_ops", "iterate", lora_attached=True)
            == "studio_ops"
        )
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_dit_profile(action, "iterate", "iterate", lora_attached=True)

    # extract/lego/complete are unaffected -- they already force studio_ops
    # for an unrelated reason, lora_attached or not.
    assert (
        jobs_module._resolve_dit_profile("extract", None, "iterate", lora_attached=False)
        == "studio_ops"
    )


def test_resolve_lora_rejects_missing_failed_or_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct unit coverage for _resolve_lora."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    project = storage.create_project(title="Unit Test")
    project_id = project["id"]

    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora(project_id, "does-not-exist")

    failed_id, failed_dir = storage.allocate_lora_dir(project_id)
    storage.write_lora_meta(
        project_id, failed_id, {"id": failed_id, "status": None, "error": "training blew up"}
    )
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora(project_id, failed_id)

    unfinished_id, unfinished_dir = storage.allocate_lora_dir(project_id)
    storage.write_lora_meta(
        project_id, unfinished_id, {"id": unfinished_id, "status": None, "error": None}
    )
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora(project_id, unfinished_id)

    ok_id, ok_dir = storage.allocate_lora_dir(project_id)
    storage.write_lora_meta(
        project_id, ok_id, {"id": ok_id, "status": "epoch 10/10", "error": None}
    )
    lora = jobs_module._resolve_lora(project_id, ok_id)
    assert lora["id"] == ok_id
