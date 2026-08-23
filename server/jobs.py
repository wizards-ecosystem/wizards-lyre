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
from pathlib import Path
from typing import Any, Callable

from server import config, storage
from worker import mock_worker

# SPEC.md sec 12 (phase order): phase 1 is generate; phase 2 adds cover and
# repaint (now that the web UI has a waveform with drag-to-select region
# feeding repainting_start/repainting_end). Phase 3 adds extract, lego, and
# complete, now that the web UI has a base-model-swap confirmation/loading
# workflow (SPEC.md sec 4.3/9.2) reused by all three. PHASE_GATED_ACTIONS is
# the one-line lever (move an action here, out of VALID_ACTIONS) for any
# action that needs to land ahead of its UI.
#
# `train_lora` (SPEC.md sec 4.4 style pack) is live: worker/acestep_worker.py
# wraps ACE-Step's DatasetBuilder -> preprocess_to_tensors -> LoRATrainer
# pipeline, and worker/mock_worker.py implements the same call shape for
# tests. It doesn't fit STUDIO_OPS_ACTIONS/SOURCE_REQUIRED_ACTIONS'
# single-`source_take_id` shape -- it takes a `source_take_ids` list instead
# (see _resolve_lora_sources) and routes to a dedicated worker entry point
# rather than the generate-shaped run_job path (see run_claimed_job).
VALID_ACTIONS = {"generate", "cover", "repaint", "extract", "lego", "complete", "train_lora"}
PHASE_GATED_ACTIONS: set[str] = set()
STUDIO_OPS_ACTIONS = {"extract", "lego", "complete"}
SOURCE_REQUIRED_ACTIONS = {"cover", "repaint", "extract", "lego", "complete"}

# SPEC.md sec 4.4 "LoRA train / load" -- the load half. A LoRA's weight
# deltas are only valid against the exact base checkpoint it was trained
# on, which worker/acestep_worker.py's LORA_BASE_DIT_PROFILE pins to
# "studio_ops" (turbo/xl-turbo are distilled few-step checkpoints the real
# trainer can't run a full diffusion training loop against -- see that
# module's LoRA docstring section). extract/lego/complete already force
# studio_ops for an unrelated reason (they're structural editing ops, not
# style-pack generation) and always did -- generate/cover/repaint are the
# actions a style-pack lora is actually for (SPEC.md's "Style pack" row),
# so _resolve_dit_profile allows studio_ops on exactly these three,
# specifically when a valid lora_id for the project is attached; an
# ordinary generate/cover/repaint with no lora attached still rejects
# studio_ops exactly as before.
LORA_ELIGIBLE_ACTIONS = {"generate", "cover", "repaint"}

# SPEC.md sec 4.4: "Style pack | LoRA train / load | 8+ songs".
MIN_LORA_SOURCE_TAKES = 8

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
        _ensure_column(conn, "jobs", "lora_id", "TEXT")
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


def _resolve_dit_profile(
    action: str,
    dit_profile: str | None,
    project_dit_profile: str,
    lora_attached: bool = False,
) -> str:
    """`dit_profile` is the job body's explicit override, if any;
    `project_dit_profile` is project.json's persisted default (PATCH
    /api/projects/{id}) -- an omitted job-level profile must fall back to
    that, not silently to 'iterate', or a project switched to e.g. 'polish'
    keeps generating with 'iterate' the moment a client omits the field
    (reviewer-flagged: the included frontend always omits it).

    `lora_attached` is True when the job body carries a validated `lora_id`
    for this project (see `_resolve_lora`) -- SPEC.md sec 4.4 "LoRA train /
    load". Loading a trained LoRA is only architecturally valid against the
    exact studio_ops base checkpoint it was trained on (see
    LORA_ELIGIBLE_ACTIONS above and worker/acestep_worker.py's
    LORA_BASE_DIT_PROFILE), so a lora-attached generate/cover/repaint is
    coerced to studio_ops the same way extract/lego/complete always are --
    and, symmetrically, an explicit non-studio_ops profile on a lora-attached
    job is rejected as a conflict instead of silently ignoring the lora."""
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
    if lora_attached and action in LORA_ELIGIBLE_ACTIONS:
        if dit_profile is None:
            return "studio_ops"
        if dit_profile != "studio_ops":
            raise JobError(
                f"action '{action}' with a lora_id attached requires dit_profile='studio_ops' "
                f"(got '{dit_profile}') -- a LoRA's weights are only valid against the "
                "studio_ops base checkpoint it was trained on (SPEC.md sec 4.4)"
            )
        return "studio_ops"
    # studio_ops is reserved for extract/lego/complete, or generate/cover/
    # repaint with a valid lora_id attached (SPEC.md sec 8.1/4.4) -- reject
    # it here for every other case instead of loading the base model for
    # ordinary generation, whether it came from an explicit override or
    # (reviewer-flagged) a project's persisted default.
    resolved = dit_profile or project_dit_profile
    if resolved == "studio_ops":
        raise JobError(
            f"action '{action}' cannot use dit_profile='studio_ops' -- that profile is "
            "reserved for extract/lego/complete, or generate/cover/repaint with a valid "
            "lora_id attached"
        )
    return resolved


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


def _distinct_lora_source_ids(body: dict[str, Any]) -> list[str]:
    """Order-preserving de-duplication of `train_lora`'s `source_take_ids`,
    shared by `_resolve_lora_sources` (validation/path resolution) and
    `_error_lora_meta` below (reviewer-flagged: error metadata was computing
    `source_take_count` from the raw submitted list while success metadata
    used the deduplicated paths -- an accepted request with 8+ distinct ids
    plus extra duplicates reported inconsistent counts depending on whether
    training succeeded or failed). Single source of truth for what "the
    source songs" means for this job."""
    seen: set[str] = set()
    distinct_ids: list[str] = []
    for take_id in body.get("source_take_ids") or []:
        if take_id not in seen:
            seen.add(take_id)
            distinct_ids.append(take_id)
    return distinct_ids


def _resolve_lora_sources(project_id: str, body: dict[str, Any]) -> list[str]:
    """Resolve `train_lora`'s `source_take_ids` to real, jailed filesystem
    paths (SPEC.md sec 4.4 / sec 11), sibling to `_resolve_source_audio`
    above -- that helper resolves a single `source_take_id`, while
    `train_lora` instead takes a list and requires SPEC's '8+ songs' floor.
    Deduplicates first (reviewer-flagged: repeating one take_id N times must
    not satisfy the floor) so both the count check and the paths actually
    handed to the worker reflect distinct songs, not raw list length. Called
    both at enqueue time (fail fast, before any job row exists) and again
    when the job runs (payload_json only stores the client's identifiers,
    not the resolved paths)."""
    distinct_ids = _distinct_lora_source_ids(body)
    if len(distinct_ids) < MIN_LORA_SOURCE_TAKES:
        source_take_ids = body.get("source_take_ids") or []
        raise JobError(
            f"action 'train_lora' requires at least {MIN_LORA_SOURCE_TAKES} distinct "
            f"source_take_ids (got {len(distinct_ids)} distinct of {len(source_take_ids)} submitted)"
        )
    paths: list[str] = []
    for take_id in distinct_ids:
        try:
            paths.append(str(storage.take_audio_path(project_id, take_id)))
        except storage.TakeNotFound as exc:
            raise JobError(f"source_take_ids entry not found: {take_id}") from exc
    return paths


def _resolve_track_name(action: str, body: dict[str, Any]) -> str | None:
    """extract/lego/complete route their target track through `track_name`,
    which the worker forwards onto ACE-Step's task-specific `instruction`
    field (SPEC.md sec 4.4) -- missing, non-string, or blank input would
    otherwise reach ACE-Step as a meaningless instruction. The web UI
    disables its Extract button until a track name is typed, but that's a
    client-side convenience only; a request posted straight to the HTTP API
    must be rejected the same way. Returns the trimmed name for actions that
    require one, else None."""
    if action not in STUDIO_OPS_ACTIONS:
        return None
    track_name = body.get("track_name")
    if not isinstance(track_name, str) or not track_name.strip():
        raise JobError(f"action '{action}' requires a non-empty track_name")
    return track_name.strip()


def _resolve_lora(project_id: str, lora_id: str) -> dict:
    """Resolve and validate `lora_id` (SPEC.md sec 4.4 "LoRA train / load"),
    sibling to `_resolve_source_audio`/`_resolve_track_name` above. Called
    both at enqueue time (fail fast) and again when the job runs
    (payload_json only stores the client's lora_id, not the resolved
    adapter path -- see `enqueue_job`'s `lora_adapter_path` payload field).

    A lora's meta.json is only ever written once training actually finished
    (`_run_train_lora_job` writes either the real success meta
    `worker.acestep_worker.train_lora`/`worker.mock_worker.train_lora`
    return, or `_error_lora_meta` on failure -- see `storage.get_lora`) --
    so there is no "still training" state to special-case here: a lora_id
    either doesn't exist yet (LoraNotFound), finished with an error (a
    non-null `error`), or finished successfully (`error` is None and
    `status` is a truthy value the training pipeline actually reported).
    ACE-Step's own training generator reports free-form progress strings as
    `status` (e.g. "epoch 2/10", see worker/acestep_worker.py's LoRA
    docstring section) rather than a fixed "completed" sentinel, so
    "successful" is checked structurally -- present and truthy, not equal to
    a specific literal -- instead of guessing at upstream's exact wording.
    """
    try:
        lora = storage.get_lora(project_id, lora_id)
    except storage.LoraNotFound as exc:
        raise JobError(f"lora_id not found: {lora_id}") from exc
    if lora.get("error"):
        raise JobError(f"lora_id '{lora_id}' failed training: {lora['error']}")
    if not lora.get("status"):
        raise JobError(f"lora_id '{lora_id}' has not finished training successfully")
    return lora


def enqueue_job(project_id: str, body: dict[str, Any]) -> dict:
    project = storage.load_project(project_id)

    action = body.get("action")
    if action in PHASE_GATED_ACTIONS:
        raise JobError(
            f"action '{action}' is not available yet (SPEC.md sec 12: gated "
            "pending its own required UI)"
        )
    if action not in VALID_ACTIONS:
        raise JobError(f"invalid action: {action}")

    lora_id = body.get("lora_id")
    if lora_id and action not in LORA_ELIGIBLE_ACTIONS:
        # SPEC.md sec 4.4: a style-pack lora only applies to
        # generate/cover/repaint (LORA_ELIGIBLE_ACTIONS above). Without this,
        # extract/lego/complete -- which already resolve to studio_ops for
        # an unrelated reason (STUDIO_OPS_ACTIONS) -- would sail past
        # _resolve_dit_profile's lora_attached branch (it only special-cases
        # LORA_ELIGIBLE_ACTIONS) and still get the adapter path attached
        # below, silently altering structural-editing output with a style
        # pack it was never meant to use. train_lora is likewise unrelated
        # to loading a lora -- reject up front instead of resolving a lora
        # whose adapter path would otherwise never even be looked at
        # (reviewer-flagged).
        raise JobError(
            f"action '{action}' cannot use lora_id -- a style-pack lora only applies to "
            f"{sorted(LORA_ELIGIBLE_ACTIONS)} (SPEC.md sec 4.4)"
        )
    resolved_lora = _resolve_lora(project_id, lora_id) if lora_id else None

    dit_profile = _resolve_dit_profile(
        action,
        body.get("dit_profile"),
        project["dit_profile"],
        lora_attached=resolved_lora is not None,
    )
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
    track_name = _resolve_track_name(action, body)
    if action == "train_lora":
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise JobError("action 'train_lora' requires a non-empty name")
        body["name"] = name.strip()
        # worker/run_worker.py publishes whether the active backend actually
        # has a `train_lora` implementation. Same fail-open semantics as
        # _check_worker_capability elsewhere: unknown/stale capability still
        # allows the job to queue -- the runtime check in _run_train_lora_job
        # is the fallback for that case.
        _check_worker_capability("train_lora")
        _resolve_lora_sources(project_id, body)

    job_id = uuid.uuid4().hex
    now = _now()
    payload = dict(body)
    payload["dit_profile"] = dit_profile
    if track_name is not None:
        payload["track_name"] = track_name
    # SPEC.md sec 8.1: "batch_size > 1 is optional later; v1 may force 1 on
    # 16 GB." Force it rather than trust an unvalidated client value straight
    # into the GPU backend (0, negative, or huge batches -> invalid calls or
    # avoidable OOMs), and phase 1 only ever consumes the first audio anyway.
    payload["batch_size"] = 1
    if resolved_lora is not None:
        # worker/acestep_worker.py's train_lora already writes final adapter
        # weights to <lora_dir>/adapter/final/ (see its module docstring) --
        # this is the exact directory the worker loads at inference time
        # (see worker/acestep_worker.py's _ensure_lora_adapter). Resolved
        # here, not in the worker, so the worker never has to reach back
        # into project storage itself (same division of labor as
        # `_resolve_source_audio`'s src_audio).
        payload["lora_adapter_path"] = str(
            storage.lora_dir(project_id, lora_id) / "adapter" / "final"
        )

    with closing(_connect()) as conn:
        # BEGIN IMMEDIATE, same as begin_project_deletion -- checking for a
        # deletion tombstone and inserting the new `queued` row must be one
        # atomic transaction, or a DELETE for this project_id could commit
        # in the gap between the check and the insert, leaving an orphaned
        # queued job for a project whose directory is about to disappear
        # (reviewer-flagged; see begin_project_deletion's docstring for the
        # other half of this race).
        conn.execute("BEGIN IMMEDIATE")
        deleting = conn.execute(
            "SELECT 1 FROM project_deletions WHERE project_id = ?", (project_id,)
        ).fetchone()
        if deleting is not None:
            conn.rollback()
            raise JobError(f"project is being deleted: {project_id}")
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
    job_id: str,
    status: str,
    take_id: str | None = None,
    lora_id: str | None = None,
    error: str | None = None,
) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, take_id = ?, lora_id = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, take_id, lora_id, error, _now(), job_id),
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
        "has_lrc": False,
        "error": error,
        "repaint": None,
        "track_name": payload.get("track_name"),
        "lora_id": payload.get("lora_id"),
        "favorite": False,
        "notes": "",
    }


def claim_next_queued_job() -> dict[str, Any] | None:
    """Atomically claim the oldest `queued` job, marking it `running` and
    starting its heartbeat lease. Used by `process_one_queued_job` / the
    dedicated worker process loop.

    The `project_deletions` exclusion is defense in depth, not the primary
    guard: `enqueue_job` and `begin_project_deletion` already serialize
    against each other so a `queued` row for a project with a tombstone
    should never exist by the time this runs (reviewer-flagged: claims must
    never pick up a job for a project mid-deletion, even if that invariant
    were ever violated by a future bug)."""
    now = _now()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT jobs.* FROM jobs "
            "LEFT JOIN project_deletions ON project_deletions.project_id = jobs.project_id "
            "WHERE jobs.status = 'queued' AND project_deletions.project_id IS NULL "
            "ORDER BY jobs.created_at ASC LIMIT 1"
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
    try:
        with storage.project_lifecycle_lock(project_id):
            if action == "train_lora":
                # train_lora has no take/DiT-swap shape at all (no source audio
                # generation, no plan) -- it gets its own worker entry point.
                _run_train_lora_job(job_id, project_id, payload)
            else:
                _run_generate_shaped_job(job_id, project_id, action, dit_profile, payload)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=HEARTBEAT_INTERVAL_SEC)


def _error_lora_meta(lora_id: str, payload: dict[str, Any], error: str) -> dict:
    """Spec-shaped meta.json for a LoRA whose training failed, mirroring
    `_error_take_meta` above -- written into the directory `allocate_lora_dir`
    already created so a failed attempt is a tracked, visible entry in
    `GET .../loras` (reviewer-flagged: without this, a failure between
    `allocate_lora_dir` and `write_lora_meta` -- e.g. the worker backend not
    implementing `train_lora` yet -- left a meta-less directory that
    `list_loras` silently skips)."""
    return {
        "id": lora_id,
        "name": payload.get("name") or "Untitled LoRA",
        "created_at": _now(),
        "source_take_count": len(_distinct_lora_source_ids(payload)),
        "base_checkpoint": None,
        "error": error,
    }


def _run_train_lora_job(job_id: str, project_id: str, payload: dict[str, Any]) -> None:
    lora_id: str | None = None
    try:
        source_paths = [Path(p) for p in _resolve_lora_sources(project_id, payload)]
        worker_module = resolve_worker_module()
        if not hasattr(worker_module, "train_lora"):
            raise JobError(
                f"worker backend '{worker_module.__name__}' does not implement "
                "train_lora (SPEC.md sec 4.4)"
            )
        lora_id, lora_dir = storage.allocate_lora_dir(project_id)
        meta = worker_module.train_lora(
            job={**payload, "action": "train_lora"},
            project_id=project_id,
            lora_id=lora_id,
            lora_dir=lora_dir,
            source_paths=source_paths,
        )
        storage.write_lora_meta(project_id, lora_id, meta)
        _set_status(job_id, "done", lora_id=lora_id)
    except Exception as exc:  # noqa: BLE001 - persist worker failure onto the job row
        if lora_id is not None:
            try:
                storage.write_lora_meta(project_id, lora_id, _error_lora_meta(lora_id, payload, str(exc)))
            except Exception:  # noqa: BLE001 - best-effort cleanup only
                pass
        _set_status(job_id, "error", lora_id=lora_id, error=str(exc))


def _run_generate_shaped_job(
    job_id: str, project_id: str, action: str, dit_profile: str, payload: dict[str, Any]
) -> None:
    def _publish_dit_loaded(loaded_profile: str) -> None:
        # Fires as soon as the worker backend confirms `loaded_profile` is
        # actually loaded (right after any base-model swap, before
        # generation runs) -- not just after the whole job finishes, like
        # worker/run_worker.py's own post-job republish. Without this, a
        # studio_ops job (SPEC.md sec 4.3) would leave /api/health reporting
        # the *previous* profile for the job's entire duration, making a
        # "loading base model..." UI state indistinguishable from ordinary
        # in-progress generation (reviewer-flagged).
        publish_worker_status(
            True, f"worker: '{loaded_profile}' DiT + LM currently loaded", loaded_profile
        )

    take_id: str | None = None
    try:
        worker_fn = _resolve_worker()
        plan = storage.load_plan(project_id)
        src_audio = _resolve_source_audio(project_id, action, payload)
        take_id, tdir = storage.allocate_take_dir(project_id)
        meta, plan_patch, lrc_text = worker_fn(
            job={**payload, "action": action, "dit_profile": dit_profile, "src_audio": src_audio},
            plan=plan,
            take_id=take_id,
            take_dir=tdir,
            on_dit_loaded=_publish_dit_loaded,
        )
        # Written before meta.json so a client polling right after the job
        # flips to "done" never sees meta["has_lrc"] True while
        # lyrics.lrc itself doesn't exist yet (SPEC.md sec 7 / sec 12
        # Phase 4). If this raises, the except block below writes an error
        # take_meta instead -- same as a merge_plan_patch failure further
        # down.
        if lrc_text is not None:
            storage.write_take_lrc(project_id, take_id, lrc_text)
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
            try:
                storage.write_take_meta(
                    project_id,
                    take_id,
                    _error_take_meta(take_id, action, dit_profile, payload, str(exc)),
                )
            except Exception:  # noqa: BLE001 - best-effort cleanup only
                # Writing the take's error metadata is a courtesy (surfaces
                # the failure on the take, not just the job row) -- if disk
                # I/O itself is failing (full disk, permissions, lock), that
                # must not stop the job row below from being marked `error`,
                # or the job is left `running` until a later worker process
                # times it out via reclaim_stale_jobs instead of failing fast.
                pass
        _set_status(job_id, "error", take_id=take_id, error=str(exc))


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


class ProjectDeletionConflict(JobError):
    """Raised when a project already has a deletion in progress/completed."""


def begin_project_deletion(project_id: str) -> list[str]:
    """Start deleting `project_id` (SPEC.md sec 9.1 "Delete (confirm)"):
    atomically records a tombstone in `project_deletions` and marks every
    still-`queued` job for this project `error`, all inside one `BEGIN
    IMMEDIATE` transaction. Returns the cancelled job ids, which the caller
    (`server.app.delete_project`) must pass to `abort_project_deletion` if
    the filesystem removal that follows this call ends up failing.

    `BEGIN IMMEDIATE` -- the same pattern `claim_next_queued_job` uses --
    is what actually closes the race a concurrent `enqueue_job` for this
    project_id would otherwise hit: SQLite serializes writers, so
    `enqueue_job`'s own "check the tombstone, then insert" runs either
    entirely before this transaction (the tombstone isn't there yet, the new
    job is inserted, and this function's own UPDATE below cancels it too,
    since it commits after) or entirely after (the tombstone is already
    there and `enqueue_job` rejects the request) -- never interleaved, so a
    job can never be inserted into the gap between this function's cancel
    step and `server.storage.delete_project` actually removing the
    directory.

    Only `queued` jobs are cancelled -- a job already `running` for this
    project fails safely on its own (`_run_generate_shaped_job` /
    `_run_train_lora_job` both wrap their work in a try/except that persists
    an `error` status without needing the project directory to still exist),
    so there's nothing to reconcile for it here.
    """
    now = _now()
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO project_deletions (project_id, started_at) VALUES (?, ?)",
                (project_id, now),
            )
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise ProjectDeletionConflict(
                f"project deletion already in progress or completed: {project_id}"
            ) from exc
        rows = conn.execute(
            "SELECT id FROM jobs WHERE project_id = ? AND status = 'queued'",
            (project_id,),
        ).fetchall()
        conn.execute(
            "UPDATE jobs SET status = 'error', error = ?, updated_at = ? "
            "WHERE project_id = ? AND status = 'queued'",
            ("project deleted", now, project_id),
        )
        conn.commit()
    return [row["id"] for row in rows]


def abort_project_deletion(project_id: str, cancelled_job_ids: list[str]) -> None:
    """Undo `begin_project_deletion`: called when `server.storage.delete_project`
    fails to actually remove the project directory (disk error, permissions,
    a file still open on Windows, ...), so a project that still exists on
    disk is never left with irreversibly cancelled jobs (reviewer-flagged).
    Removes the tombstone and restores exactly the jobs this deletion
    attempt cancelled -- not just any job that happens to be `error` for
    this project, since some of those may have failed for real, unrelated
    reasons and must stay `error`.
    """
    if not cancelled_job_ids:
        with closing(_connect()) as conn:
            conn.execute("DELETE FROM project_deletions WHERE project_id = ?", (project_id,))
            conn.commit()
        return
    now = _now()
    placeholders = ",".join("?" for _ in cancelled_job_ids)
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM project_deletions WHERE project_id = ?", (project_id,))
        conn.execute(
            f"UPDATE jobs SET status = 'queued', error = NULL, updated_at = ? "
            f"WHERE id IN ({placeholders}) AND status = 'error' AND error = 'project deleted'",
            (now, *cancelled_job_ids),
        )
        conn.commit()


def get_job(job_id: str) -> dict:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFound(job_id)
    return _row_to_dict(row)


def list_recent_jobs(
    limit: int = 20,
    project_id: str | None = None,
    action: str | None = None,
    active: bool = False,
) -> list[dict]:
    """Recent jobs, newest first, optionally narrowed to one project and/or
    one action *before* the LIMIT applies. The web UI uses this to recover
    the open project's still-queued/running `train_lora` job after a page
    refresh (SPEC.md sec 4.4 style-pack training runs up to ~1 hour, so a
    refresh mid-training is expected to restore visible progress) -- with
    only the unfiltered top-N, a long training job gets pushed out of the
    window by jobs enqueued after it (which pile up behind it while the GPU
    is locked), exactly when the user most needs to find it. `active` keeps
    only still-active jobs (`queued`/`running`) and returns them ALL, with
    no recency truncation: the caller is recovering the queue's outstanding
    worklist (and deriving "finished" from membership in it), so any cap --
    even one applied after the status filter -- could drop the oldest
    running job again behind newer queued ones. `limit` therefore only
    applies when `active` is false. Filters are optional; with none given
    this is exactly the old unfiltered query."""
    query = "SELECT * FROM jobs"
    clauses: list[str] = []
    params: list[Any] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if action is not None:
        clauses.append("action = ?")
        params.append(action)
    if active:
        clauses.append("status IN ('queued', 'running')")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"
    if not active:
        query += " LIMIT ?"
        params.append(limit)
    with closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "action": row["action"],
        "dit_profile": row["dit_profile"],
        "status": row["status"],
        "take_id": row["take_id"],
        "lora_id": row["lora_id"] if "lora_id" in row.keys() else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
