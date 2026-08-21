"""SPEC.md sec 12 (phase 6): POST /api/projects/{id}/uploads lets a client
put a file at the path `server.jobs._resolve_source_audio` /
`storage.resolve_upload_path` already know how to consume via a job's
`upload_path` (SPEC.md sec 8.1) -- those two already existed end to end,
but nothing ever wrote to disk. Covers: a valid .wav upload resolves under
the project's jailed dir, a disallowed extension is rejected 400, and a
cover job using `upload_path` (instead of `source_take_id`) completes via
the mocked worker. Mirrors tests/test_cover_flow.py's fixture/harness setup
and tests/test_acestep_worker_adapter.py's `_write_tiny_wav` helper.
"""

from __future__ import annotations

import io
import threading
import time
import wave
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


def _tiny_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 400)
    return buf.getvalue()


def test_upload_wav_resolves_under_project_jail(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Upload Test"}).json()
    project_id = project["id"]

    resp = client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": ("my song.wav", _tiny_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    upload_path = resp.json()["upload_path"]
    assert upload_path.startswith("uploads/")
    assert upload_path.endswith(".wav")

    resolved = storage.resolve_upload_path(project_id, upload_path)
    assert resolved.exists()
    assert resolved.is_relative_to(storage.project_dir(project_id))


def test_upload_rejects_disallowed_extension(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Upload Reject Test"}).json()
    project_id = project["id"]

    resp = client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_extension_outside_wav_mp3(client: TestClient) -> None:
    # SPEC.md sec 12 Phase 6 scopes drag-drop ingest to "a local WAV/MP3"
    # specifically -- .flac is a real audio format but outside that scope
    # (reviewer-flagged: the .gitignore-derived extension list silently
    # widened the feature's contract).
    project = client.post("/api/projects", json={"title": "Upload Scope Test"}).json()
    project_id = project["id"]

    resp = client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": ("source.flac", _tiny_wav_bytes(), "audio/flac")},
    )
    assert resp.status_code == 400


def test_oversized_upload_rejected_before_reaching_disk(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An oversized body must be rejected at the ASGI receive layer -- before
    # Starlette's multipart parser spools it to a temp file -- not only
    # after the fact by the endpoint's own byte counter (reviewer-flagged:
    # the endpoint-only check let an arbitrarily large body consume
    # temp-disk space before MAX_UPLOAD_BYTES was ever checked).
    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 10)
    project = client.post("/api/projects", json={"title": "Upload Oversize Test"}).json()
    project_id = project["id"]

    oversized = b"0" * 200_000  # well past MAX_UPLOAD_BYTES(10) + middleware slack
    resp = client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": ("big.wav", oversized, "audio/wav")},
    )
    assert resp.status_code == 413, resp.text

    uploads_dir = storage.uploads_dir(project_id)
    assert not uploads_dir.exists() or list(uploads_dir.iterdir()) == []


def test_cover_from_uploaded_source(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Upload Cover Flow"}).json()
    project_id = project["id"]

    upload_resp = client.post(
        f"/api/projects/{project_id}/uploads",
        files={"file": ("source.wav", _tiny_wav_bytes(), "audio/wav")},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    upload_path = upload_resp.json()["upload_path"]

    cover_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "cover",
            "dit_profile": "iterate",
            "upload_path": upload_path,
            "audio_cover_strength": 0.5,
            "seed": -1,
        },
    )
    assert cover_resp.status_code == 200, cover_resp.text
    cover_job = _wait_for_job(client, cover_resp.json()["id"])
    assert cover_job["status"] == "done", cover_job.get("error")
    cover_take_id = cover_job["take_id"]
    assert cover_take_id

    detail = client.get(f"/api/projects/{project_id}").json()
    cover_take = next(t for t in detail["takes"] if t["id"] == cover_take_id)
    assert cover_take["task_type"] == "cover"
    assert cover_take["parent_take_id"] is None
