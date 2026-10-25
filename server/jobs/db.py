"""SQLite connection, schema, and row helpers for the job queue.

The server process and the worker process both open this same database and
both call `init_db()` at startup, so every migration here has to tolerate
losing that race to the other process.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime

from server import config


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_stale(reference: str | None, stale_after: float) -> bool:
    """True if `reference` (an ISO-8601 timestamp, or None/unparseable) is
    older than `stale_after` seconds ago. Shared by job heartbeat reclaim,
    worker-lease staleness, and worker-status freshness checks."""
    if not reference:
        return True
    try:
        ref_ts = datetime.fromisoformat(reference).timestamp()
    except ValueError:
        return True
    return ref_ts < datetime.now(UTC).timestamp() - stale_after


def _connect() -> sqlite3.Connection:
    # A non-zero timeout lets concurrent readers/writers (the HTTP process
    # inserting jobs, the worker loop claiming/updating them) retry instead
    # of raising "database is locked".
    conn = sqlite3.connect(config.db_path(), timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    except sqlite3.OperationalError as exc:
        # The server and the worker process both call init_db() at startup
        # and can race here; if the column showed up between our PRAGMA
        # check and this ALTER, that's another caller finishing the same
        # migration, not a real failure.
        if "duplicate column name" not in str(exc):
            raise


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
        # Added for the heartbeat/lease/stale-reclaim mechanism; migrated in
        # for any jobs.db created before it existed.
        _ensure_column(conn, "jobs", "heartbeat_at", "TEXT")
        _ensure_column(conn, "jobs", "attempts", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "jobs", "lora_id", "TEXT")
        # SPEC.md sec 15.7: progress and cooperative cancel. Existing rows
        # migrate with a null stage and cancel_requested 0.
        _ensure_column(conn, "jobs", "stage", "TEXT")
        _ensure_column(conn, "jobs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0")
        # SPEC.md sec 16.4/16.5, all additive. `attempt_token` fences every
        # write a claimed attempt makes; `resolved_json` is the brief, seed,
        # and model configuration frozen before the first DiT call;
        # `remote_handle` is the Remote GPU job token, so an interrupted
        # remote render is reconciled instead of silently resubmitted;
        # `submission_id` and `retry_of` are the request identities.
        _ensure_column(conn, "jobs", "attempt_token", "TEXT")
        _ensure_column(conn, "jobs", "resolved_json", "TEXT")
        _ensure_column(conn, "jobs", "remote_handle", "TEXT")
        _ensure_column(conn, "jobs", "submission_id", "TEXT")
        _ensure_column(conn, "jobs", "retry_of", "TEXT")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS jobs_submission "
            "ON jobs (project_id, submission_id) WHERE submission_id IS NOT NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_capabilities (
                dit_profile TEXT PRIMARY KEY,
                supported INTEGER NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                ready INTEGER NOT NULL,
                message TEXT,
                loaded_dit_profile TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        # SPEC.md sec 4.3: one GPU occupant. A single row that a worker
        # process must hold (see acquire_worker_lease) before it may
        # initialize ACE-Step or start claiming jobs.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_lease (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                owner_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL
            )
            """
        )
        # SPEC.md sec 9.1 "Delete (confirm)". A tombstone row, present for
        # the duration of a DELETE /api/projects/{id} request, that closes
        # the race between deleting a project and a concurrent enqueue_job
        # for the same project_id: both go through a single BEGIN IMMEDIATE
        # transaction that checks this table, so whichever of "insert the
        # tombstone + cancel queued jobs" (begin_project_deletion) or "check
        # for a tombstone + insert a new queued job" (enqueue_job) commits
        # first is the one that wins -- see both functions' docstrings.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_deletions (
                project_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL
            )
            """
        )
        # SPEC.md sec 15.3. One row per (caller, Idempotency-Key). `expires_at`
        # is a unix timestamp; reserve() drops rows once that second has
        # passed. `state` is `pending` while the first request is in flight
        # and `done` once its response is stored for replay.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                owner_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                status_code INTEGER NOT NULL DEFAULT 0,
                body BLOB NOT NULL,
                location TEXT,
                content_type TEXT,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                PRIMARY KEY (owner_id, idempotency_key)
            )
            """
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "action": row["action"],
        "dit_profile": row["dit_profile"],
        "status": row["status"],
        "take_id": row["take_id"],
        # sqlite3.Row's __contains__ tests values, not column names, so the
        # explicit .keys() is required here (SIM118 does not apply).
        "lora_id": row["lora_id"] if "lora_id" in row.keys() else None,  # noqa: SIM118
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        # sqlite3.Row's __contains__ tests values, not column names.
        "stage": row["stage"] if "stage" in row.keys() else None,  # noqa: SIM118
        "cancel_requested": (
            int(row["cancel_requested"]) if "cancel_requested" in row.keys() else 0  # noqa: SIM118
        ),
        "submission_id": row["submission_id"] if "submission_id" in row.keys() else None,  # noqa: SIM118
        "retry_of": row["retry_of"] if "retry_of" in row.keys() else None,  # noqa: SIM118
    }
