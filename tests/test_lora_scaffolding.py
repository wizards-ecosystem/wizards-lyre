"""Phase 4 LoRA storage + job scaffolding (SPEC.md sec 12 phase order /
sec 4.4 "Style pack | LoRA train / load"). `train_lora` is a live action in
`server.jobs.VALID_ACTIONS`: worker/acestep_worker.py wraps ACE-Step's
training pipeline, and worker/mock_worker.py implements the same call shape
for tests (no CUDA). End-to-end enqueue/run coverage lives in
tests/test_train_lora_flow.py; this module covers the 8+ source floor,
dedup helpers, and the mock worker's adapter file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import jobs as jobs_module
from server import storage
from worker import mock_worker

MIN_LORA_SOURCE_TAKES = 8


def _make_takes(client: TestClient, project_id: str, count: int) -> list[str]:
    take_ids: list[str] = []
    for _ in range(count):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "dit_profile": "iterate", "seed": -1},
        )
        assert resp.status_code == 200
        job = wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")
        take_ids.append(job["take_id"])
    return take_ids


def test_train_lora_rejects_too_few_sources_before_queuing(client: TestClient) -> None:
    """The 8+ songs floor fires at enqueue time -- no job row is created."""
    project = client.post("/api/projects", json={"title": "LoRA Too Few"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": [], "name": "Too few"},
    )
    assert resp.status_code == 400
    assert "8" in resp.json()["detail"]
    assert all(j["action"] != "train_lora" for j in client.get("/api/jobs").json())


def test_distinct_lora_source_ids_deduplicates_order_preserving() -> None:
    """Direct unit coverage for `_distinct_lora_source_ids`."""
    body = {"source_take_ids": ["a", "b", "a", "c", "b"]}
    assert jobs_module._distinct_lora_source_ids(body) == ["a", "b", "c"]


def test_resolve_lora_sources_requires_min_distinct_takes(client: TestClient) -> None:
    """Direct unit coverage for `_resolve_lora_sources` against real take_ids."""
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
    """Direct unit coverage of worker/mock_worker.py's train_lora."""
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
