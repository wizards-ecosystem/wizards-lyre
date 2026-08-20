"""SQLite job queue. Phase 1 runs jobs synchronously against the mocked
worker so `pytest` never needs a background process or a GPU. See
SPEC.md sec 5 (IPC) and sec 8.1 (job body / studio_ops enforcement).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any

from server import config, storage
from worker import mock_worker

VALID_ACTIONS = {"generate", "cover", "repaint", "extract", "lego", "complete"}
STUDIO_OPS_ACTIONS = {"extract", "lego", "complete"}


class JobError(ValueError):
    pass


class JobNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
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
        )
        conn.commit()


def _resolve_dit_profile(action: str, dit_profile: str | None) -> str:
    if dit_profile is not None and dit_profile not in storage.VALID_DIT_PROFILES:
        raise JobError(f"invalid dit_profile: {dit_profile}")
    if action in STUDIO_OPS_ACTIONS:
        # SPEC.md sec 8.1: reject extract/lego/complete unless dit_profile is
        # studio_ops. An unset profile is coerced; an explicit mismatch is rejected.
        if dit_profile is None:
            return "studio_ops"
        if dit_profile != "studio_ops":
            raise JobError(
                f"action '{action}' requires dit_profile='studio_ops' (got '{dit_profile}')"
            )
        return "studio_ops"
    return dit_profile or "iterate"


def enqueue_job(project_id: str, body: dict[str, Any]) -> dict:
    storage.load_project(project_id)

    action = body.get("action")
    if action not in VALID_ACTIONS:
        raise JobError(f"invalid action: {action}")

    dit_profile = _resolve_dit_profile(action, body.get("dit_profile"))

    job_id = uuid.uuid4().hex
    now = _now()
    payload = dict(body)
    payload["dit_profile"] = dit_profile

    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO jobs (id, project_id, action, dit_profile, status, payload_json, "
            "take_id, error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                project_id,
                action,
                dit_profile,
                "queued",
                json.dumps(payload),
                None,
                None,
                now,
                now,
            ),
        )
        conn.commit()

    _run_job(job_id, project_id, action, dit_profile, payload)
    return get_job(job_id)


def _set_status(
    job_id: str, status: str, take_id: str | None = None, error: str | None = None
) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, take_id = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, take_id, error, _now(), job_id),
        )
        conn.commit()


def _run_job(job_id: str, project_id: str, action: str, dit_profile: str, payload: dict) -> None:
    _set_status(job_id, "running")
    try:
        plan = storage.load_plan(project_id)
        take_id, tdir = storage.allocate_take_dir(project_id)
        meta = mock_worker.run_job(
            job={**payload, "action": action, "dit_profile": dit_profile},
            plan=plan,
            take_id=take_id,
            take_dir=tdir,
        )
        storage.write_take_meta(project_id, take_id, meta)
        storage.set_active_take(project_id, take_id)
        _set_status(job_id, "done", take_id=take_id)
    except Exception as exc:  # noqa: BLE001 - persist worker failure onto the job row
        _set_status(job_id, "error", error=str(exc))


def get_job(job_id: str) -> dict:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFound(job_id)
    return _row_to_dict(row)


def list_recent_jobs(limit: int = 20) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "action": row["action"],
        "dit_profile": row["dit_profile"],
        "status": row["status"],
        "take_id": row["take_id"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
