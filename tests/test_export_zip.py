"""SPEC.md sec 12 Phase 5 / sec 9.2: GET /api/projects/{id}/export builds an
in-memory zip of project.json, plan.json, the active take's mix, and
(optionally) every extract/lego/complete take's audio -- see
storage.build_export_zip's docstring for why "stems" means "these
task_types" rather than a stems/ directory that nothing in the current
storage layout actually writes.
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
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


def test_export_with_no_takes_still_200s_with_minimal_zip(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Empty Export"}).json()
    project_id = project["id"]

    resp = client.get(f"/api/projects/{project_id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    assert "Empty Export-export.zip" in resp.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert names == {"project.json", "plan.json"}

    project_json = json.loads(zf.read("project.json"))
    assert project_json["id"] == project_id
    plan_json = json.loads(zf.read("plan.json"))
    assert plan_json == storage.default_plan()


def test_export_after_generate_includes_active_mix(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Generate Export"}).json()
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
    take_id = gen["take_id"]

    resp = client.get(f"/api/projects/{project_id}/export")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "project.json" in names
    assert "plan.json" in names
    assert "mix.wav" in names or "mix.mp3" in names

    project_json = json.loads(zf.read("project.json"))
    assert project_json["active_take_id"] == take_id


def test_export_includes_or_excludes_extract_stem_by_flag(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Stem Export"}).json()
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

    extract_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals",
        },
    )
    extract_job = _wait_for_job(client, extract_resp.json()["id"])
    assert extract_job["status"] == "done", extract_job.get("error")
    stem_take_id = extract_job["take_id"]

    expected_stem_name = f"{stem_take_id}-extract-vocals.wav"

    resp_with_stems = client.get(f"/api/projects/{project_id}/export?include_stems=true")
    assert resp_with_stems.status_code == 200
    names_with_stems = set(zipfile.ZipFile(io.BytesIO(resp_with_stems.content)).namelist())
    assert expected_stem_name in names_with_stems

    resp_without_stems = client.get(f"/api/projects/{project_id}/export?include_stems=false")
    assert resp_without_stems.status_code == 200
    names_without_stems = set(zipfile.ZipFile(io.BytesIO(resp_without_stems.content)).namelist())
    assert expected_stem_name not in names_without_stems
    assert not any(name.startswith(f"{stem_take_id}-extract-") for name in names_without_stems)

    # default (no query param) includes stems
    resp_default = client.get(f"/api/projects/{project_id}/export")
    names_default = set(zipfile.ZipFile(io.BytesIO(resp_default.content)).namelist())
    assert expected_stem_name in names_default
