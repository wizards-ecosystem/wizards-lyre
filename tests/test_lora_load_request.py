"""Phase 4 'LoRA load' request scaffolding (SPEC.md sec 4.4/12): attach an
existing, successfully-trained `lora_id` to a generate/cover/repaint job so
it *requires* (not merely permits) the `studio_ops` checkpoint that lora's
weights were trained against (worker/acestep_worker.py's
`LORA_BASE_DIT_PROFILE`) -- an omitted dit_profile is coerced to studio_ops
and any explicit non-studio_ops profile is rejected, the same shape
STUDIO_OPS_ACTIONS already enforces for extract/lego/complete. The real
adapter-loading call itself is still not implemented (worker/acestep_worker.py
fails loudly with `WorkerUnavailable` if `lora_id` reaches `run_job`, see
tests/test_acestep_worker_adapter.py::test_run_job_rejects_lora_id_until_real_loading_exists)
-- researching and wiring ACE-Step's real inference-time LoRA API is a
follow-up job's concern. This module covers the request shape, validation,
and forced-studio_ops gating end to end against the mocked worker.

See tests/test_train_lora_flow.py for the training side this builds on, and
tests/test_extract_requires_studio_ops.py for the pre-existing studio_ops
gating this extends without disturbing.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs as jobs_module
from server import storage
from server.app import app
from worker.run_worker import run_loop

MIN_LORA_SOURCE_TAKES = 8


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


def _train_lora(client: TestClient, project_id: str, name: str = "style pack") -> str:
    """Train a real lora via the existing mocked train_lora flow (same
    pattern as tests/test_train_lora_flow.py) so tests below exercise a
    genuinely valid, successfully-trained lora_id."""
    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "name": name, "source_take_ids": take_ids},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"], timeout=10.0)
    assert job["status"] == "done", job.get("error")
    lora_id = job["lora_id"]
    assert lora_id
    return lora_id


def test_generate_with_studio_ops_and_no_lora_still_rejected(client: TestClient) -> None:
    """Regression guard: an ordinary generate with no lora_id attached must
    keep rejecting studio_ops exactly as it did before this job's gating
    extension."""
    project = client.post("/api/projects", json={"title": "No LoRA Gate"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops"},
    )
    assert resp.status_code == 400
    assert "studio_ops" in resp.json()["detail"]
    assert all(j["action"] != "generate" for j in client.get("/api/jobs").json())


def test_generate_with_studio_ops_and_valid_lora_is_accepted(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Load"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    lora_id = _train_lora(client, project_id)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops", "lora_id": lora_id},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["dit_profile"] == "studio_ops"

    take_id = job["take_id"]
    assert take_id
    take = next(
        t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == take_id
    )
    assert take["lora_id"] == lora_id


def test_generate_with_lora_and_omitted_dit_profile_forces_studio_ops(client: TestClient) -> None:
    """A lora_id must *require* studio_ops, not just permit it alongside
    whatever profile happens to be in play -- an omitted dit_profile must
    not silently fall back to the project default (iterate)."""
    project = client.post("/api/projects", json={"title": "LoRA Forces Studio Ops"}).json()
    project_id = project["id"]
    assert client.get(f"/api/projects/{project_id}").json()["project"]["dit_profile"] == "iterate"
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    lora_id = _train_lora(client, project_id)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "lora_id": lora_id},
    )
    assert resp.status_code == 200
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")
    assert job["dit_profile"] == "studio_ops"


def test_generate_with_lora_and_incompatible_dit_profile_is_rejected(client: TestClient) -> None:
    """A lora is only valid applied back onto studio_ops -- an explicit
    incompatible profile alongside lora_id must 400, not silently ignore the
    lora or apply it to the wrong checkpoint."""
    project = client.post("/api/projects", json={"title": "LoRA Incompatible Profile"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    lora_id = _train_lora(client, project_id)
    jobs_before = len(client.get("/api/jobs").json())

    for dit_profile in ("iterate", "polish", "quality"):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "dit_profile": dit_profile, "lora_id": lora_id},
        )
        assert resp.status_code == 400, dit_profile
        assert "studio_ops" in resp.json()["detail"]

    # none of the rejected requests left a job row behind (the training's own
    # source-take generate jobs above are the only ones that exist).
    assert len(client.get("/api/jobs").json()) == jobs_before


def test_cover_and_repaint_also_accept_studio_ops_with_a_lora(client: TestClient) -> None:
    """LORA_STYLE_ACTIONS covers generate, cover, and repaint alike -- the
    gating extension isn't generate-only."""
    project = client.post("/api/projects", json={"title": "LoRA Cover Repaint"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "reference take"},
    )

    source_take_id = _make_takes(client, project_id, 1)[0]
    lora_id = _train_lora(client, project_id)

    for action, extra in (
        ("cover", {"source_take_id": source_take_id}),
        (
            "repaint",
            {"source_take_id": source_take_id, "repainting_start": 0, "repainting_end": 1},
        ),
    ):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": action, "dit_profile": "studio_ops", "lora_id": lora_id, **extra},
        )
        assert resp.status_code == 200, (action, resp.json())
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")


def test_lora_id_not_found_404s_before_job_created(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "LoRA Missing"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops", "lora_id": "does-not-exist"},
    )
    assert resp.status_code == 404
    assert all(j["action"] != "generate" for j in client.get("/api/jobs").json())


def test_lora_id_pointing_at_failed_training_is_rejected(client: TestClient) -> None:
    """A lora whose own training run errored out (SPEC.md sec 10 point 5's
    error-meta contract, mirrored by `_error_lora_meta`) must not be
    silently treated as usable just because the id resolves."""
    project = client.post("/api/projects", json={"title": "LoRA Failed"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    lora_id, _lora_dir = storage.allocate_lora_dir(project_id)
    storage.write_lora_meta(
        project_id,
        lora_id,
        {
            "id": lora_id,
            "name": "Broken Pack",
            "created_at": "2026-01-01T00:00:00+00:00",
            "source_take_count": MIN_LORA_SOURCE_TAKES,
            "base_checkpoint": None,
            "error": "training crashed",
        },
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops", "lora_id": lora_id},
    )
    assert resp.status_code == 400
    assert lora_id in resp.json()["detail"]
    assert all(j["action"] != "generate" for j in client.get("/api/jobs").json())


def test_resolve_dit_profile_gates_studio_ops_on_lora_requested() -> None:
    """Direct unit coverage for _resolve_dit_profile's lora_requested param:
    it must *force* studio_ops for generate/cover/repaint, not just permit
    it alongside an explicit or default profile."""
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_dit_profile("generate", "studio_ops", "iterate")
    assert (
        jobs_module._resolve_dit_profile("generate", "studio_ops", "iterate", lora_requested=True)
        == "studio_ops"
    )

    # an omitted dit_profile is coerced to studio_ops when lora_requested,
    # not left to fall back to the project default.
    assert (
        jobs_module._resolve_dit_profile("generate", None, "iterate", lora_requested=True)
        == "studio_ops"
    )
    for action in ("generate", "cover", "repaint"):
        assert (
            jobs_module._resolve_dit_profile(action, None, "polish", lora_requested=True)
            == "studio_ops"
        )
        # an explicit incompatible profile is rejected, not silently ignored.
        for incompatible in ("iterate", "polish", "quality"):
            with pytest.raises(jobs_module.JobError):
                jobs_module._resolve_dit_profile(
                    action, incompatible, "iterate", lora_requested=True
                )

    # extract/lego/complete are unaffected by lora_requested either way --
    # they were already required (and coerced) onto studio_ops before this job.
    assert jobs_module._resolve_dit_profile("extract", None, "iterate") == "studio_ops"
    assert (
        jobs_module._resolve_dit_profile("extract", None, "iterate", lora_requested=True)
        == "studio_ops"
    )

    # train_lora isn't in LORA_STYLE_ACTIONS -- lora_requested doesn't open
    # (or force) studio_ops for it.
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_dit_profile(
            "train_lora", "studio_ops", "iterate", lora_requested=True
        )
    assert (
        jobs_module._resolve_dit_profile("train_lora", None, "iterate", lora_requested=True)
        == "iterate"
    )


def test_resolve_requested_lora_returns_none_when_absent() -> None:
    """Direct unit coverage for _resolve_requested_lora: no lora_id in the
    body is not an error, just "no lora requested"."""
    assert jobs_module._resolve_requested_lora("some-project", {}) is None
    assert jobs_module._resolve_requested_lora("some-project", {"lora_id": None}) is None
