"""SPEC.md sec 4.4: complete fills out an arrangement, taking a partial
`src_audio` plus target `track_name`/track classes. The backend
(`server.jobs._resolve_dit_profile`/`_resolve_source_audio`) and worker
adapter (`worker/acestep_worker.py`) already implement complete's full call
contract -- exercised directly by tests/test_acestep_worker_adapter.py --
and `_resolve_track_name`/studio_ops enforcement are already covered for all
three studio_ops actions by tests/test_extract_requires_studio_ops.py and
tests/test_phase1_api.py::test_resolve_dit_profile_studio_ops_enforcement.
This module only covers complete's success path end-to-end, mirroring
tests/test_extract_requires_studio_ops.py::test_extract_endpoint_succeeds_with_studio_ops_and_real_source.
Unlike lego, complete has no optional interval -- just source_take_id and
track_name (SPEC.md sec 4.4: "track classes" maps onto the same track_name
field as extract/lego).
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


def test_complete_endpoint_succeeds_with_studio_ops_and_real_source(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Complete Flow"}).json()
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
            "action": "complete",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals, drums, bass",
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
    assert take["task_type"] == "complete"
    assert take["parent_take_id"] == source_take_id
    assert take["track_name"] == "vocals, drums, bass"
