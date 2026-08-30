"""Which worker backend is active, and what it has published about itself.

The worker process announces three things through SQLite: an exclusive lease
(SPEC.md sec 4.3, one GPU occupant), its readiness/loaded profile, and which
DiT profiles it can actually serve. The HTTP server only ever reads them --
it must never import acestep (SPEC.md sec 10 point 4).
"""

from __future__ import annotations

import os
from contextlib import closing
from typing import Any, Callable

from server.jobs.db import _connect, _is_stale, _now
from server.jobs.errors import JobError
from worker import mock_worker


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


def resolve_worker_module():
    """Pick the worker backend module. `mock` is for tests/local dev only;
    production defaults to the real ACE-Step worker (SPEC.md sec 10 / 14).

    This is the only place that reaches for `worker.acestep_worker` (which
    lazily imports acestep/CUDA) -- callers must be the dedicated worker
    process (`worker/run_worker.py`) or code that only runs there, never the
    FastAPI server (SPEC.md sec 10 point 4 / worker-server isolation)."""
    backend = os.environ.get("LYRE_WORKER", "acestep")
    if backend == "mock":
        return mock_worker
    if backend == "acestep":
        from worker import acestep_worker

        return acestep_worker
    raise JobError(f"unknown LYRE_WORKER backend: {backend}")


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
