"""SQLite job queue (`queued` -> `running` -> `done` | `error`).

`enqueue_job` only validates and inserts a `queued` row; it never runs a
job itself. Jobs are claimed and executed by whoever calls
`process_one_queued_job` -- in production that's the dedicated
`worker/run_worker.py` process (`python -m worker.run_worker`), a separate
OS process from the FastAPI server so a long generation never blocks HTTP
and a native GPU crash can't take the server down with it. See SPEC.md
sec 5 (IPC / three processes), sec 8.1 (job body / studio_ops enforcement),
and sec 10 (worker contract).

A claimed job carries a heartbeat lease: `run_claimed_job` touches
`heartbeat_at` every `HEARTBEAT_INTERVAL_SEC` while it runs. If the worker
process dies or the GPU crashes mid-job, the heartbeat stops; the next
`reclaim_stale_jobs` call (run at worker startup and each poll) finds the
stuck `running` row and either requeues it for another attempt or, past
`MAX_ATTEMPTS`, marks it `error` instead of leaving it `running` forever.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Callable

from server import config, storage
from worker import mock_worker

# SPEC.md sec 12 (phase order): phase 1 is generate; phase 2 adds cover.
# repaint (rest of phase 2) and extract/lego/complete (phase 3) each need
# frontend workflow this build doesn't have yet -- repaint regions and a
# base-model-swap confirmation/loading UX -- so the API must not accept them
# yet even though worker/acestep_worker.py's adapter already implements
# their call contract (exercised directly by
# tests/test_acestep_worker_adapter.py, independent of this gate). Moving an
# action from PHASE_GATED_ACTIONS to VALID_ACTIONS is the one-line change
# that turns it on once its phase's UI/workflow lands.
VALID_ACTIONS = {"generate", "cover"}
PHASE_GATED_ACTIONS = {"repaint", "extract", "lego", "complete"}
STUDIO_OPS_ACTIONS = {"extract", "lego", "complete"}
SOURCE_REQUIRED_ACTIONS = {"cover", "repaint", "extract", "lego", "complete"}

HEARTBEAT_INTERVAL_SEC = 5.0
STALE_AFTER_SEC = 60.0
MAX_ATTEMPTS = 3

# SPEC.md sec 4.3: one GPU occupant. worker/run_worker.py must hold this
# lease before it initializes ACE-Step or starts polling the job queue, so
# two worker processes can never load models / run jobs concurrently.
WORKER_LEASE_HEARTBEAT_INTERVAL_SEC = 5.0
WORKER_LEASE_STALE_AFTER_SEC = 30.0

# worker/run_worker.py republishes worker_status/worker_capabilities on this
# cadence even while idle (not just after a job runs), so a worker that has
# died or hung stops reading as ready once its last publish is older than
# WORKER_STATUS_STALE_AFTER_SEC -- see get_worker_status / _check_worker_capability.
WORKER_STATUS_HEARTBEAT_INTERVAL_SEC = 5.0
WORKER_STATUS_STALE_AFTER_SEC = 30.0

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
    return ref_ts < datetime.now(timezone.utc).timestamp() - stale_after


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
        conn.commit()


def resolve_worker_module():
    """Pick the worker backend module. `mock` is for tests/local dev only;
    production defaults to the real ACE-Step worker (SPEC.md sec 10 / 14).

    This is the only place that reaches for `worker.acestep_worker` (which
    lazily imports acestep/CUDA) -- callers must be the dedicated worker
    process (`worker/run_worker.py`) or code that only runs there, never the
    FastAPI server (SPEC.md sec 10 point 4 / worker-server isolation)."""
    backend = os.environ.get("BARD_WORKER", "acestep")
    if backend == "mock":
        return mock_worker
    if backend == "acestep":
        from worker import acestep_worker

        return acestep_worker
    raise JobError(f"unknown BARD_WORKER backend: {backend}")


def acquire_worker_lease(
    owner_id: str, stale_after: float = WORKER_LEASE_STALE_AFTER_SEC
) -> bool:
    """Atomically claim the single cross-process worker lease (SPEC.md sec
    4.3: one GPU occupant / serialized jobs). Succeeds if no lease is held,
    we already hold it, or the current holder's heartbeat has gone stale
    (crashed/killed). Returns False if a live worker -- fresh heartbeat,
    different owner -- currently holds it; the caller must not initialize
    ACE-Step or start claiming jobs in that case."""
    now = _now()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT owner_id, heartbeat_at FROM worker_lease WHERE id = 1"
        ).fetchone()
        if (
            row is not None
            and row["owner_id"] != owner_id
            and not _is_stale(row["heartbeat_at"], stale_after)
        ):
            conn.commit()
            return False
        conn.execute(
            "INSERT INTO worker_lease (id, owner_id, acquired_at, heartbeat_at) "
            "VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET owner_id = excluded.owner_id, "
            "acquired_at = excluded.acquired_at, heartbeat_at = excluded.heartbeat_at",
            (owner_id, now, now),
        )
        conn.commit()
        return True


def renew_worker_lease(owner_id: str) -> bool:
    """Refresh the lease heartbeat. Returns False if this process no longer
    owns the lease (it went stale and another worker already claimed it) --
    the caller must stop claiming/running jobs immediately in that case."""
    with closing(_connect()) as conn:
        cur = conn.execute(
            "UPDATE worker_lease SET heartbeat_at = ? WHERE id = 1 AND owner_id = ?",
            (_now(), owner_id),
        )
        conn.commit()
        return cur.rowcount > 0


def release_worker_lease(owner_id: str) -> None:
    """Give up the lease on clean shutdown so a restart doesn't have to wait
    out WORKER_LEASE_STALE_AFTER_SEC. A no-op if we don't currently hold it
    (e.g. it already went stale and was taken over)."""
    with closing(_connect()) as conn:
        conn.execute(
            "DELETE FROM worker_lease WHERE id = 1 AND owner_id = ?", (owner_id,)
        )
        conn.commit()


def _resolve_worker() -> WorkerFn:
    return resolve_worker_module().run_job


def publish_worker_capability(dit_profile: str, supported: bool, reason: str | None) -> None:
    """Record what the active worker backend can currently load. Called by
    `worker/run_worker.py` -- the one process allowed to import acestep --
    so `_check_worker_capability` below can enforce SPEC.md sec 4.1/8.1
    (reject `quality` without CPU-offload support) from the FastAPI process
    without ever importing acestep itself (SPEC.md sec 10 point 4)."""
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO worker_capabilities (dit_profile, supported, reason, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(dit_profile) DO UPDATE SET "
            "supported = excluded.supported, reason = excluded.reason, "
            "updated_at = excluded.updated_at",
            (dit_profile, 1 if supported else 0, reason, _now()),
        )
        conn.commit()


def get_worker_capability(dit_profile: str) -> tuple[bool, str | None] | None:
    """Last-published `(supported, reason)` for `dit_profile`, or None if no
    worker has ever reported on it (e.g. `worker/run_worker.py` hasn't
    started yet)."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT supported, reason FROM worker_capabilities WHERE dit_profile = ?",
            (dit_profile,),
        ).fetchone()
    if row is None:
        return None
    return bool(row["supported"]), row["reason"]


def publish_worker_status(ready: bool, message: str, loaded_dit_profile: str | None) -> None:
    """Record overall worker readiness and which DiT profile (if any) is
    currently loaded in the worker process's memory. Called by
    `worker/run_worker.py` right after its startup readiness check
    (`<backend>.initialize_worker()`) so `/api/health` can report real
    worker state (SPEC.md sec 8: `dit_loaded`) instead of a static guess --
    the FastAPI process can't read the worker's in-memory state directly
    (SPEC.md sec 10 point 4 / worker-server isolation)."""
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO worker_status (id, ready, message, loaded_dit_profile, updated_at) "
            "VALUES (1, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET ready = excluded.ready, message = excluded.message, "
            "loaded_dit_profile = excluded.loaded_dit_profile, updated_at = excluded.updated_at",
            (1 if ready else 0, message, loaded_dit_profile, _now()),
        )
        conn.commit()


def get_worker_status(stale_after: float = WORKER_STATUS_STALE_AFTER_SEC) -> dict[str, Any] | None:
    """Last-published worker readiness, or None if no worker process has
    ever reported in (e.g. `worker/run_worker.py` hasn't started yet).

    Folds in a freshness check: `worker/run_worker.py` republishes this on
    WORKER_STATUS_HEARTBEAT_INTERVAL_SEC even while idle, specifically so a
    worker that exited or hung stops reading as ready once its last publish
    is older than `stale_after`, instead of `/api/health` repeating a dead
    process's last-known-good status forever."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT ready, message, loaded_dit_profile, updated_at FROM worker_status WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    if _is_stale(row["updated_at"], stale_after):
        return {
            "ready": False,
            "message": f"worker heartbeat stale since {row['updated_at']} -- is worker.run_worker still running?",
            "loaded_dit_profile": None,
            "updated_at": row["updated_at"],
        }
    return {
        "ready": bool(row["ready"]),
        "message": row["message"],
        "loaded_dit_profile": row["loaded_dit_profile"],
        "updated_at": row["updated_at"],
    }


def _check_worker_capability(
    dit_profile: str, stale_after: float = WORKER_STATUS_STALE_AFTER_SEC
) -> None:
    """SPEC.md sec 4.1/8.1: `quality` (XL) needs CPU offload on a 16 GB
    card; reject it up front instead of risking an uncontrolled OOM deep
    inside a job run. Reads capability the worker process already published
    to SQLite -- if it hasn't published anything yet, the job is allowed to
    queue and the worker's own `_ensure_loaded` guard still enforces this
    when the job actually runs.

    Also treats a stale publish (worker exited or hung -- see
    WORKER_STATUS_STALE_AFTER_SEC) as unknown-capability rather than
    trusting it: an unknown capability still allows the job to queue (same
    as never-published), instead of enqueue validation acting on a
    potentially wrong, long-dead process's last report."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT supported, reason, updated_at FROM worker_capabilities WHERE dit_profile = ?",
            (dit_profile,),
        ).fetchone()
    if row is None or _is_stale(row["updated_at"], stale_after):
        return
    if not row["supported"]:
        raise JobError(row["reason"] or f"worker cannot currently load dit_profile '{dit_profile}'")


def _resolve_dit_profile(action: str, dit_profile: str | None, project_dit_profile: str) -> str:
    """`dit_profile` is the job body's explicit override, if any;
    `project_dit_profile` is project.json's persisted default (PATCH
    /api/projects/{id}) -- an omitted job-level profile must fall back to
    that, not silently to 'iterate', or a project switched to e.g. 'polish'
    keeps generating with 'iterate' the moment a client omits the field
    (reviewer-flagged: the included frontend always omits it)."""
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
    return dit_profile or project_dit_profile


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
    project = storage.load_project(project_id)

    action = body.get("action")
    if action in PHASE_GATED_ACTIONS:
        raise JobError(
            f"action '{action}' is not available yet -- this build implements "
            "'generate' and 'cover' (SPEC.md sec 12: repaint is the rest of "
            "phase 2, extract/lego/complete land in phase 3, each with "
            "required UI this build doesn't have yet)"
        )
    if action not in VALID_ACTIONS:
        raise JobError(f"invalid action: {action}")

    dit_profile = _resolve_dit_profile(action, body.get("dit_profile"), project["dit_profile"])
    # SPEC.md sec 4.1/8.1 only calls for early rejection of 'quality' (XL
    # needs CPU offload on a 16 GB card) -- every other profile always
    # reports supported=True from the real supports_dit_profile() and only
    # ever shows up as unsupported here because the *whole worker* failed to
    # start (worker/run_worker.py's _publish_capabilities marks every
    # profile unsupported in that case, not just quality). Gating ordinary
    # profiles on that blocks them from ever reaching the queue -- so they
    # can never become a recoverable `error` job with failure metadata
    # (SPEC.md sec 10 point 5, README's documented contract) -- and blocks
    # recovery from a transient startup failure entirely, since no job can
    # reach _ensure_loaded to retry it (reviewer-flagged).
    if dit_profile == "quality":
        _check_worker_capability(dit_profile)
    _resolve_source_audio(project_id, action, body)

    job_id = uuid.uuid4().hex
    now = _now()
    payload = dict(body)
    payload["dit_profile"] = dit_profile
    # SPEC.md sec 8.1: "batch_size > 1 is optional later; v1 may force 1 on
    # 16 GB." Force it rather than trust an unvalidated client value straight
    # into the GPU backend (0, negative, or huge batches -> invalid calls or
    # avoidable OOMs), and phase 1 only ever consumes the first audio anyway.
    payload["batch_size"] = 1

    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO jobs (id, project_id, action, dit_profile, status, payload_json, "
            "take_id, error, created_at, updated_at, heartbeat_at, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                None,
                0,
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


def _touch_heartbeat(job_id: str) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
            (_now(), job_id),
        )
        conn.commit()


def _heartbeat_loop(job_id: str, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL_SEC):
        _touch_heartbeat(job_id)


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
    """Atomically claim the oldest `queued` job, marking it `running` and
    starting its heartbeat lease. Used by `process_one_queued_job` / the
    dedicated worker process loop."""
    now = _now()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', heartbeat_at = ?, "
            "attempts = attempts + 1, updated_at = ? WHERE id = ?",
            (now, now, row["id"]),
        )
        conn.commit()

    job = _row_to_dict(row)
    job["status"] = "running"  # `row` was fetched before the UPDATE above
    job["payload"] = json.loads(row["payload_json"])
    return job


def run_claimed_job(job: dict[str, Any]) -> None:
    """Run one already-`running` job to completion. GPU work happens here --
    run this from `worker/run_worker.py` as its own process in production so
    a worker/GPU failure or crash never reaches the HTTP process. A
    background thread renews the job's heartbeat lease while this runs so a
    hard crash (which skips the except/finally below entirely) is still
    detectable by `reclaim_stale_jobs`."""
    job_id = job["id"]
    project_id = job["project_id"]
    action = job["action"]
    dit_profile = job["dit_profile"]
    payload = job["payload"]

    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop, args=(job_id, stop_heartbeat), daemon=True
    )
    heartbeat_thread.start()

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
        if plan_patch is not None:
            # `plan` above was loaded before this (possibly long-running)
            # generation started. merge_plan_patch re-reads the plan and
            # merges only the LM-filled fields onto whatever is current on
            # disk -- atomically, under storage's per-project lock, so a
            # PUT /plan landing between the read and the write can't be
            # silently overwritten (SPEC.md sec 5/7.2). `plan_patch` is a
            # delta, not a full plan replacement -- see worker.mock_worker
            # / worker.acestep_worker.
            storage.merge_plan_patch(project_id, plan_patch)
        # Promote the take to active only after every fallible persistence
        # step above has actually succeeded (reviewer-flagged): if
        # merge_plan_patch raised, the except block below marks this take
        # `error`, and active_take_id must not be left pointing at it.
        storage.set_active_take(project_id, take_id)
        _set_status(job_id, "done", take_id=take_id)
    except Exception as exc:  # noqa: BLE001 - persist worker failure onto the job row
        if take_id is not None:
            storage.write_take_meta(
                project_id,
                take_id,
                _error_take_meta(take_id, action, dit_profile, payload, str(exc)),
            )
        _set_status(job_id, "error", take_id=take_id, error=str(exc))
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL_SEC)


def reclaim_stale_jobs(stale_after: float = STALE_AFTER_SEC) -> list[str]:
    """Recover jobs a dead/crashed worker left stuck `running` (SPEC.md sec
    10 point 5). A job whose heartbeat is older than `stale_after` seconds is
    requeued for another attempt, or -- past `MAX_ATTEMPTS` -- marked `error`
    instead of retried forever. Called at worker startup and on every poll
    (see `worker/run_worker.py`).

    Note: a take directory the crashed attempt had already allocated (before
    its heartbeat went stale) is not tracked here and is left on disk; only
    the job's queue state is recovered.
    """
    reclaimed: list[str] = []
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id, attempts, heartbeat_at, updated_at FROM jobs WHERE status = 'running'"
        ).fetchall()
        for row in rows:
            reference = row["heartbeat_at"] or row["updated_at"]
            if not _is_stale(reference, stale_after):
                continue

            if row["attempts"] >= MAX_ATTEMPTS:
                conn.execute(
                    "UPDATE jobs SET status = 'error', error = ?, updated_at = ? WHERE id = ?",
                    (
                        f"worker heartbeat lost after {row['attempts']} attempt(s); "
                        "the worker process likely crashed",
                        _now(),
                        row["id"],
                    ),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = 'queued', heartbeat_at = NULL, updated_at = ? "
                    "WHERE id = ?",
                    (_now(), row["id"]),
                )
            reclaimed.append(row["id"])
        conn.commit()
    return reclaimed


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
