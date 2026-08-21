"""Phase 4 LoRA storage + job scaffolding, phase-gated (SPEC.md sec 12 phase
order / sec 4.4 "Style pack | LoRA train / load"). `train_lora` sits in
`server.jobs.PHASE_GATED_ACTIONS`, not `VALID_ACTIONS`, until a production
backend (worker/acestep_worker.py) actually implements ACE-Step LoRA
training/loading and a web UI workflow exists to drive it -- only
worker/mock_worker.py has train_lora today, and it's tests-only, no CUDA
(reviewer-flagged: exposing the action to POST /api/projects/{id}/jobs while
only the mock backend supports it would let a client queue a job production
can only ever reject).

This module covers the phase gate itself, plus direct unit coverage of the
(currently unreachable via the API) resolution helpers and the mock worker's
train_lora, kept correct and ready for whichever follow-up lands a real
backend and moves the action back into VALID_ACTIONS -- the same pattern
tests/test_phase1_api.py used to keep extract/lego/complete's dit_profile
logic tested while those actions were themselves phase-gated (see its
test_resolve_dit_profile_studio_ops_enforcement).
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
from worker import mock_worker
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


def test_train_lora_action_is_phase_gated(client: TestClient) -> None:
    """POST /api/projects/{id}/jobs must reject train_lora outright,
    regardless of how well-formed the rest of the request is (mirrors the
    historical treatment of extract/lego/complete during their own gated
    phase in tests/test_phase1_api.py)."""
    project = client.post("/api/projects", json={"title": "LoRA Phase Gate"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": take_ids, "name": "Gated"},
    )
    assert resp.status_code == 400
    assert "not available yet" in resp.json()["detail"]

    # no job row and no lora directory left behind by the rejected request
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())
    assert client.get(f"/api/projects/{project_id}/loras").json() == []


def test_train_lora_gated_even_with_too_few_or_duplicate_sources(client: TestClient) -> None:
    """The phase gate fires before any request-shape validation runs -- an
    otherwise-invalid train_lora request (too few sources) still gets the
    phase-gate message, not the source-count error, since the action never
    reaches that validation while gated."""
    project = client.post("/api/projects", json={"title": "LoRA Gate Before Validation"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": [], "name": "Gated"},
    )
    assert resp.status_code == 400
    assert "not available yet" in resp.json()["detail"]


def test_distinct_lora_source_ids_deduplicates_order_preserving() -> None:
    """Direct unit coverage for `_distinct_lora_source_ids`, currently
    unreachable via the API while train_lora is phase-gated -- kept correct
    for whichever follow-up re-enables the action."""
    body = {"source_take_ids": ["a", "b", "a", "c", "b"]}
    assert jobs_module._distinct_lora_source_ids(body) == ["a", "b", "c"]


def test_resolve_lora_sources_requires_min_distinct_takes(client: TestClient) -> None:
    """Direct unit coverage for `_resolve_lora_sources` against real
    take_ids, currently unreachable via the API while train_lora is
    phase-gated -- kept correct for whichever follow-up re-enables it."""
    project = client.post("/api/projects", json={"title": "LoRA Sources Unit"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    take_ids = _make_takes(client, project_id, MIN_LORA_SOURCE_TAKES)

    paths = jobs_module._resolve_lora_sources(project_id, {"source_take_ids": take_ids})
    assert len(paths) == MIN_LORA_SOURCE_TAKES

    # one short of the floor
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora_sources(project_id, {"source_take_ids": take_ids[:-1]})

    # the floor is on distinct songs, not list length -- one id repeated
    # enough times to hit the raw count must still be rejected
    repeated = [take_ids[0]] * MIN_LORA_SOURCE_TAKES
    with pytest.raises(jobs_module.JobError):
        jobs_module._resolve_lora_sources(project_id, {"source_take_ids": repeated})


def test_mock_worker_train_lora_writes_adapter_and_meta(tmp_path: Path) -> None:
    """Direct unit coverage of worker/mock_worker.py's train_lora -- the
    only backend that implements it today (tests only, no CUDA) -- kept
    correct and ready to be swapped for a real implementation without this
    coverage lapsing while the action itself stays phase-gated."""
    lora_dir = tmp_path / "lora"
    meta = mock_worker.train_lora(
        job={"name": "My Style"},
        project_id="proj",
        lora_id="lora1",
        lora_dir=lora_dir,
        source_paths=[tmp_path / f"take{i}.wav" for i in range(MIN_LORA_SOURCE_TAKES)],
    )
    assert meta["name"] == "My Style"
    assert meta["source_take_count"] == MIN_LORA_SOURCE_TAKES
    assert meta["error"] is None
    assert (lora_dir / "adapter.bin").exists()
    assert lora_dir.is_relative_to(tmp_path)
