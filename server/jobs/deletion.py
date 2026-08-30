"""Project deletion's queue half (SPEC.md sec 9.1).

A tombstone row plus a single BEGIN IMMEDIATE transaction closes the race
between deleting a project and a concurrent `enqueue_job` for the same
project -- see `enqueue_job` for the other half.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

from server.jobs.db import _connect, _now
from server.jobs.errors import ProjectDeletionConflict


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
