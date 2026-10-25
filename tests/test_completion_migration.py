"""SPEC.md sec 16.7: additive migration, legacy data, and a consistent backup.

Old databases, projects, plans, and takes keep working and keep their
authored values; unknown provenance stays unknown; a backup taken with the
writers stopped restores the matching set; and nothing on the ordinary
paths can rent Remote GPU hardware.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import backup, config, jobs, storage
from server.app import app
from tests.helpers import wait_for_job
from worker.run_worker import run_loop

_V0_JOBS = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    action TEXT NOT NULL,
    dit_profile TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    take_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _columns(db: Path) -> set[str]:
    with closing(sqlite3.connect(db)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}


def test_an_old_jobs_database_migrates_in_place(lyre_env: Path) -> None:
    db = config.db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(_V0_JOBS)
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-done",
                "p",
                "generate",
                "iterate",
                "done",
                json.dumps({"action": "generate", "seed": 5}),
                "t",
                None,
                "2026-01-01",
                "2026-01-01",
            ),
        )
        conn.commit()
    jobs.init_db()
    assert {
        "attempt_token",
        "resolved_json",
        "remote_handle",
        "submission_id",
        "retry_of",
        "stage",
        "cancel_requested",
    } <= _columns(db)
    row = jobs.read_job("old-done")
    assert row["status"] == "done"
    assert row["resolved"] is None
    assert row["submission_id"] is None


def test_an_interrupted_migration_completes_on_the_next_start(lyre_env: Path) -> None:
    """A process that died after adding some columns leaves the rest for the
    next init_db; every step is idempotent."""
    db = config.db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(_V0_JOBS)
        conn.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT")
        conn.execute("ALTER TABLE jobs ADD COLUMN attempt_token TEXT")
        conn.commit()
    jobs.init_db()
    jobs.init_db()
    assert {"attempt_token", "resolved_json", "retry_of", "attempts"} <= _columns(db)


def _write_legacy_project(root: Path) -> tuple[str, str]:
    """A project exactly as a pre-section-16 Lyre wrote it."""
    project_id, take_id = "legacyproject", "legacytake"
    pdir = root / project_id
    (pdir / "takes" / take_id).mkdir(parents=True)
    (pdir / "project.json").write_text(
        json.dumps(
            {
                "id": project_id,
                "title": "Old song",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "dit_profile": "iterate",
                "lm_model": "acestep-5Hz-lm-1.7B",
                "active_take_id": take_id,
            }
        ),
        encoding="utf-8",
    )
    (pdir / "plan.json").write_text(
        json.dumps(
            {
                "query": "",
                "caption": "authored caption",
                "negative": [],
                "lyrics": "[Verse]\nauthored words",
                "instrumental": False,
                "vocal_language": "en",
                "bpm": 120,
                "keyscale": "C Major",
                "timesignature": "4/4",
                "duration_sec": 120,
                "sections": [],
            }
        ),
        encoding="utf-8",
    )
    take = pdir / "takes" / take_id
    (take / "meta.json").write_text(
        json.dumps(
            {
                "id": take_id,
                "parent_take_id": None,
                "task_type": "text2music",
                "dit_profile": "iterate",
                "seed": 77,
                "duration_sec": 118.4,
                "caption": "authored caption",
                "lyrics": "[Verse]\nauthored words",
                "bpm": 120,
                "keyscale": "C Major",
                "created_at": "2026-01-01T00:00:00+00:00",
                "score": None,
                "error": None,
                "repaint": None,
                "track_name": None,
            }
        ),
        encoding="utf-8",
    )
    import wave

    with wave.open(str(take / "mix.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return project_id, take_id


def test_old_projects_keep_their_authored_values_and_unknown_origins(
    api_client: TestClient,
) -> None:
    project_id, take_id = _write_legacy_project(config.projects_dir())
    detail = api_client.get(f"/v1/projects/{project_id}").json()
    plan = detail["plan"]
    assert plan["caption"] == "authored caption"
    assert plan["bpm"] == 120
    assert plan["caption_rewrite"] is False  # the legacy behavior
    assert plan["creation_mode"] == "from_brief"
    assert set(plan["field_origins"].values()) == {"unknown"}
    take = detail["takes"][0]
    assert take["seed"] == 77
    assert take["provenance"] is None
    assert take["provenance_complete"] is False
    assert take["notes"] == "" and take["favorite"] is False
    # Nothing was rewritten on disk by reading it.
    on_disk = json.loads((config.projects_dir() / project_id / "plan.json").read_text())
    assert "field_origins" not in on_disk

    # A legacy plan renders as it always did: unknown origins in from_brief
    # mode keep their values; nothing invents a lock.
    job = api_client.post(f"/v1/projects/{project_id}/jobs", json={"action": "generate"})
    creation = jobs.read_job(job.json()["id"])["payload"]["creation"]
    assert creation["caption"] == "authored caption"
    assert creation["keyscale"] == "C major"
    assert creation["origins"]["bpm"] == "unknown"

    # Annotations and export still work on the old take.
    api_client.patch(f"/v1/projects/{project_id}/takes/{take_id}", json={"notes": "still here"})
    assert storage.get_take(project_id, take_id)["notes"] == "still here"
    assert api_client.get(f"/v1/projects/{project_id}/export").status_code == 200


def test_backup_and_restore_carry_a_matching_set(
    api_client: TestClient, lyre_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This test owns its worker so it can stop it: a worker left running
    # would heartbeat into whichever database the environment names next,
    # including the fresh install restored into below.
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker_thread.start()
    try:
        project_id = api_client.post("/api/projects", json={"title": "Keep me"}).json()["id"]
        job = api_client.post(
            f"/api/projects/{project_id}/jobs", json={"action": "generate"}
        ).json()
        take_id = wait_for_job(api_client, job["id"])["take_id"]
        audio = storage.take_audio_path(project_id, take_id).read_bytes()
        # The worker is still heartbeating: a backup now refuses.
        with pytest.raises(backup.BackupError, match="worker is running"):
            backup.create_backup(tmp_path / "too-soon")
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)
    assert not worker_thread.is_alive()

    # Quiesce: the worker lease goes stale once its owner stops.
    with closing(jobs._connect()) as conn:
        conn.execute("UPDATE worker_lease SET heartbeat_at = '2000-01-01T00:00:00+00:00'")
        conn.commit()
    monkeypatch.setattr(backup, "_writers_running", lambda: None)
    record = backup.create_backup(tmp_path / "set")
    assert record["projects"] == 1
    assert record["database"]["sha256"].startswith("sha256:")
    assert not any(".staging" in entry["path"] for entry in record["files"])
    assert all(entry["path"] != "lyre.db" for entry in record["files"])

    # Restore into a fresh install.
    fresh = tmp_path / "fresh"
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(fresh / "projects"))
    monkeypatch.setenv("LYRE_DB_PATH", str(fresh / "projects" / "lyre.db"))
    backup.restore_backup(tmp_path / "set")
    with TestClient(app) as restored:
        detail = restored.get(f"/v1/projects/{project_id}").json()
        assert detail["project"]["title"] == "Keep me"
        assert detail["takes"][0]["id"] == take_id
        assert restored.get(f"/v1/jobs/{job['id']}").json()["status"] == "done"
    assert storage.take_audio_path(project_id, take_id).read_bytes() == audio

    # A second restore over a non-empty install refuses instead of mixing sets.
    with pytest.raises(backup.BackupError, match="not empty"):
        backup.restore_backup(tmp_path / "set")


def test_a_tampered_backup_is_refused(lyre_env: Path, tmp_path: Path, monkeypatch) -> None:
    storage.create_project(title="Tamper")
    jobs.init_db()
    monkeypatch.setattr(backup, "_writers_running", lambda: None)
    backup.create_backup(tmp_path / "set")
    plan = next((tmp_path / "set" / "projects").rglob("plan.json"))
    plan.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(tmp_path / "fresh"))
    monkeypatch.setenv("LYRE_DB_PATH", str(tmp_path / "fresh" / "lyre.db"))
    with pytest.raises(backup.BackupError, match="missing or changed"):
        backup.restore_backup(tmp_path / "set")


def test_ordinary_paths_never_rent_remote_hardware(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generate through /api, /v1, and MCP, read capabilities, and start the
    worker loop's readiness check with the Remote GPU lane selected: no
    request reaches a provisioner (SPEC.md sec 3.1/16.5)."""
    from server.remote_gpu import client as remote_client

    calls: list[str] = []

    def transport(method, url, headers, body, timeout, max_bytes):
        calls.append(f"{method} {url}")
        raise AssertionError(f"unexpected network call: {method} {url}")

    monkeypatch.setattr(remote_client, "TRANSPORT", transport)
    monkeypatch.setenv("LYRE_WORKER", "remote")
    monkeypatch.setenv("LYRE_REMOTE_GPU_PROVISIONER", "runpod")
    monkeypatch.setenv("LYRE_RUNPOD_API_KEY", "test-key-not-real")

    project_id = api_client.post("/api/projects", json={"title": "No rent"}).json()["id"]
    api_job = api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    assert api_job.status_code == 200
    v1_job = api_client.post(f"/v1/projects/{project_id}/jobs", json={"action": "generate"})
    assert v1_job.status_code == 202
    assert api_client.post("/v1/songs", json={"prompt": "rain"}).status_code == 202
    assert api_client.get("/v1/capabilities").json()["worker"]["backend"] == "remote"
    from worker import remote_backend

    ready, _message = remote_backend.initialize_worker()
    assert ready is False
    assert calls == []
