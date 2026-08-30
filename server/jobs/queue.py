"""The queue itself: enqueue, claim, heartbeat, reclaim, and read back.

`enqueue_job` only ever inserts a `queued` row -- running a job is
`server.jobs.runner`'s business, in the dedicated worker process
(SPEC.md sec 5).
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import closing
from typing import Any

from server import storage
from server.jobs.db import _connect, _is_stale, _now, _row_to_dict
from server.jobs.errors import JobError, JobNotFound
from server.jobs.validation import (
    LORA_ELIGIBLE_ACTIONS,
    PHASE_GATED_ACTIONS,
    VALID_ACTIONS,
    _resolve_dit_profile,
    _resolve_lora,
    _resolve_lora_sources,
    _resolve_source_audio,
    _resolve_track_name,
)
from server.jobs.worker_registry import _check_worker_capability

HEARTBEAT_INTERVAL_SEC = 5.0
STALE_AFTER_SEC = 60.0
MAX_ATTEMPTS = 3


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
            "UPDATE jobs SET status = ?, take_id = ?, lora_id = ?, error = ?, updated_at = ? "
            "WHERE id = ?",
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
