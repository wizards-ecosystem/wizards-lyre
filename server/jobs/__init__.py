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

This package is the single import surface -- `from server import jobs`, then
`jobs.<name>` -- split by concern:

- `errors`          -- the exception types server/app.py maps onto HTTP codes
- `db`              -- connection, schema, migrations, row shape
- `worker_registry` -- backend selection, GPU lease, published status
- `validation`      -- enqueue-time request validation and resolution
- `queue`           -- enqueue, claim, heartbeat, reclaim, read back
- `runner`          -- executing a claimed job (worker process only)
- `deletion`        -- the queue half of project deletion
"""

from __future__ import annotations

from server.jobs.db import _connect, _ensure_column, _is_stale, _now, _row_to_dict, init_db
from server.jobs.deletion import abort_project_deletion, begin_project_deletion
from server.jobs.errors import JobError, JobNotFound, ProjectDeletionConflict
from server.jobs.queue import (
    HEARTBEAT_INTERVAL_SEC,
    MAX_ATTEMPTS,
    STALE_AFTER_SEC,
    _heartbeat_loop,
    _set_status,
    _touch_heartbeat,
    claim_next_queued_job,
    enqueue_job,
    get_job,
    list_recent_jobs,
    reclaim_stale_jobs,
)
from server.jobs.runner import (
    _error_lora_meta,
    _error_take_meta,
    _run_generate_shaped_job,
    _run_train_lora_job,
    process_one_queued_job,
    run_claimed_job,
)
from server.jobs.validation import (
    LORA_ELIGIBLE_ACTIONS,
    MIN_LORA_SOURCE_TAKES,
    PHASE_GATED_ACTIONS,
    SOURCE_REQUIRED_ACTIONS,
    STUDIO_OPS_ACTIONS,
    VALID_ACTIONS,
    _distinct_lora_source_ids,
    _resolve_dit_profile,
    _resolve_lora,
    _resolve_lora_sources,
    _resolve_source_audio,
    _resolve_track_name,
)
from server.jobs.worker_registry import (
    WORKER_LEASE_HEARTBEAT_INTERVAL_SEC,
    WORKER_LEASE_STALE_AFTER_SEC,
    WORKER_STATUS_HEARTBEAT_INTERVAL_SEC,
    WORKER_STATUS_STALE_AFTER_SEC,
    WorkerFn,
    _check_worker_capability,
    _resolve_worker,
    acquire_worker_lease,
    get_worker_capability,
    get_worker_status,
    publish_worker_capability,
    publish_worker_status,
    release_worker_lease,
    renew_worker_lease,
    resolve_worker_module,
)

__all__ = [
    # Private helpers are part of the historical surface: tests and the
    # worker process reach them as jobs.<name>.
    "_check_worker_capability",
    "_connect",
    "_distinct_lora_source_ids",
    "_ensure_column",
    "_error_lora_meta",
    "_error_take_meta",
    "_heartbeat_loop",
    "_is_stale",
    "_now",
    "_resolve_dit_profile",
    "_resolve_lora",
    "_resolve_lora_sources",
    "_resolve_source_audio",
    "_resolve_track_name",
    "_resolve_worker",
    "_row_to_dict",
    "_run_generate_shaped_job",
    "_run_train_lora_job",
    "_set_status",
    "_touch_heartbeat",
    "HEARTBEAT_INTERVAL_SEC",
    "LORA_ELIGIBLE_ACTIONS",
    "MAX_ATTEMPTS",
    "MIN_LORA_SOURCE_TAKES",
    "PHASE_GATED_ACTIONS",
    "SOURCE_REQUIRED_ACTIONS",
    "STALE_AFTER_SEC",
    "STUDIO_OPS_ACTIONS",
    "VALID_ACTIONS",
    "WORKER_LEASE_HEARTBEAT_INTERVAL_SEC",
    "WORKER_LEASE_STALE_AFTER_SEC",
    "WORKER_STATUS_HEARTBEAT_INTERVAL_SEC",
    "WORKER_STATUS_STALE_AFTER_SEC",
    "JobError",
    "JobNotFound",
    "ProjectDeletionConflict",
    "WorkerFn",
    "abort_project_deletion",
    "acquire_worker_lease",
    "begin_project_deletion",
    "claim_next_queued_job",
    "enqueue_job",
    "get_job",
    "get_worker_capability",
    "get_worker_status",
    "init_db",
    "list_recent_jobs",
    "process_one_queued_job",
    "publish_worker_capability",
    "publish_worker_status",
    "reclaim_stale_jobs",
    "release_worker_lease",
    "renew_worker_lease",
    "resolve_worker_module",
    "run_claimed_job",
]
