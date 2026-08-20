"""SQLite job queue (`queued` -> `running` -> `done` | `error`).

`enqueue_job` only validates and inserts a `queued` row; it never runs a
job itself. Jobs are claimed and executed by whoever calls
`process_one_queued_job` -- in production that's the dedicated
`worker/run_worker.py` process (`python -m worker.run_worker`), a separate
OS process from the FastAPI server so a long generation never blocks HTTP
and a native GPU crash can't take the server down with it. See SPEC.md
sec 5 (IPC / three processes), sec 8.1 (job body / studio_ops enforcement),
and sec 10 (worker contract).
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Callable

from server import config, storage
from worker import mock_worker

VALID_ACTIONS = {"generate", "cover", "repaint", "extract", "lego", "complete"}
STUDIO_OPS_ACTIONS = {"extract", "lego", "complete"}
SOURCE_REQUIRED_ACTIONS = {"cover", "repaint", "extract", "lego", "complete"}

# A worker returns the take's meta.json plus an optional plan.json patch
# (simple-mode generation fills caption/lyrics/metas from the LM and the
# filled plan must be persisted -- SPEC.md sec 7.2).
WorkerFn = Callable[..., tuple[dict, dict | None]]


class JobError(ValueError):
    pass


class JobNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    # A non-zero timeout lets concurrent readers/writers (the HTTP process
    # inserting jobs, the worker loop claiming/updating them) retry instead
    # of raising "database is locked".
    conn = sqlite3.connect(config.db_path(), timeout=5.0)
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


def _resolve_worker() -> WorkerFn:
    """Pick the worker backend. `mock` is for tests/local dev only; production
    defaults to the real ACE-Step worker (SPEC.md sec 10 / sec 14)."""
    backend = os.environ.get("BARD_WORKER", "acestep")
    if backend == "mock":
        return mock_worker.run_job
    if backend == "acestep":
        from worker import acestep_worker

        return acestep_worker.run_job
    raise JobError(f"unknown BARD_WORKER backend: {backend}")


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


def _resolve_source_audio(project_id: str, action: str, body: dict[str, Any]) -> str | None:
    """Resolve cover/repaint/extract/lego/complete's source to a real, jailed
    filesystem path (SPEC.md sec 8.1 / sec 11). Called both at enqueue time
    (fail fast) and again when the job runs (payload_json only stores the
    client's identifiers, not the resolved path)."""
    if action not in SOURCE_REQUIRED_ACTIONS:
        return None

    source_take_id = body.get("source_take_id")
    upload_path = body.get("upload_path")
    if not source_take_id and not upload_path:
        raise JobError(f"action '{action}' requires source_take_id or upload_path")

    if source_take_id:
        try:
            path = storage.take_audio_path(project_id, source_take_id)
        except storage.TakeNotFound as exc:
            raise JobError(f"source_take_id not found: {source_take_id}") from exc
        return str(path)

    path = storage.resolve_upload_path(project_id, upload_path)
    if not path.exists():
        raise JobError(f"upload_path not found: {upload_path}")
    return str(path)


def enqueue_job(project_id: str, body: dict[str, Any]) -> dict:
    storage.load_project(project_id)

    action = body.get("action")
    if action not in VALID_ACTIONS:
        raise JobError(f"invalid action: {action}")

    dit_profile = _resolve_dit_profile(action, body.get("dit_profile"))
    _resolve_source_audio(project_id, action, body)

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


def _error_take_meta(
    take_id: str, action: str, dit_profile: str, payload: dict[str, Any], error: str
) -> dict:
    """Spec-shaped meta.json for a take whose generation failed (SPEC.md sec
    10 point 5: 'write meta.json with error, mark job error')."""
    return {
        "id": take_id,
        "parent_take_id": payload.get("source_take_id"),
        "task_type": mock_worker.TASK_TYPE_BY_ACTION.get(action, action),
        "dit_profile": dit_profile,
        "seed": payload.get("seed", -1),
        "duration_sec": None,
        "caption": None,
        "lyrics": None,
        "bpm": None,
        "keyscale": None,
        "created_at": _now(),
        "score": None,
        "error": error,
        "repaint": None,
        "track_name": payload.get("track_name"),
    }


def claim_next_queued_job() -> dict[str, Any] | None:
    """Atomically claim the oldest `queued` job, marking it `running`. Used
    by `process_one_queued_job` / the dedicated worker process loop."""
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        conn.commit()

    job = _row_to_dict(row)
    job["payload"] = json.loads(row["payload_json"])
    return job


def run_claimed_job(job: dict[str, Any]) -> None:
    """Run one already-`running` job to completion. GPU work happens here --
    run this from `worker/run_worker.py` as its own process in production so
    a worker/GPU failure or crash never reaches the HTTP process."""
    job_id = job["id"]
    project_id = job["project_id"]
    action = job["action"]
    dit_profile = job["dit_profile"]
    payload = job["payload"]

    take_id: str | None = None
    try:
        worker_fn = _resolve_worker()
        plan = storage.load_plan(project_id)
        src_audio = _resolve_source_audio(project_id, action, payload)
        take_id, tdir = storage.allocate_take_dir(project_id)
        meta, plan_patch = worker_fn(
            job={**payload, "action": action, "dit_profile": dit_profile, "src_audio": src_audio},
            plan=plan,
            take_id=take_id,
            take_dir=tdir,
        )
        storage.write_take_meta(project_id, take_id, meta)
        storage.set_active_take(project_id, take_id)
        if plan_patch is not None:
            storage.save_plan(project_id, plan_patch)
        _set_status(job_id, "done", take_id=take_id)
    except Exception as exc:  # noqa: BLE001 - persist worker failure onto the job row
        if take_id is not None:
            storage.write_take_meta(
                project_id,
                take_id,
                _error_take_meta(take_id, action, dit_profile, payload, str(exc)),
            )
        _set_status(job_id, "error", take_id=take_id, error=str(exc))


def process_one_queued_job() -> bool:
    """Claim and run one queued job. Returns False if the queue was empty."""
    job = claim_next_queued_job()
    if job is None:
        return False
    run_claimed_job(job)
    return True


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
