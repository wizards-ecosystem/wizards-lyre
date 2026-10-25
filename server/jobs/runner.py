"""Executing a claimed job. Runs only in the worker process.

`run_claimed_job` owns the failure contract from SPEC.md sec 10 point 5: a
job that fails writes a spec-shaped meta.json carrying the error and marks
the row `error`, rather than leaving it `running` or losing the failure.
"""

from __future__ import annotations

import errno
import inspect
import os
import shutil
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from server import storage
from server.generation import artifacts
from server.generation import brief as creation
from server.jobs import queue as job_queue
from server.jobs.db import _connect, _now
from server.jobs.errors import JobCancelled, JobError, JobFenced
from server.jobs.queue import (
    HEARTBEAT_INTERVAL_SEC,
    _heartbeat_loop,
    _set_status,
    cancel_requested,
    claim_next_queued_job,
    finish_done,
    freeze_resolved,
    mark_cancelled,
    record_remote_handle,
    set_stage,
)
from server.jobs.validation import (
    _distinct_lora_source_ids,
    _resolve_lora_sources,
    _resolve_source_audio,
)
from server.jobs.worker_registry import (
    _resolve_worker,
    publish_worker_status,
    resolve_worker_module,
)
from worker import mock_worker
from worker.acestep_worker.settings import DEFAULT_LM

PROVENANCE_SCHEMA = "lyre.take-provenance/1"


def describe_failure(exc: BaseException, *, job_id: str, stage: str | None) -> str:
    """An actionable, bounded error for the job row (SPEC.md sec 16.5): what
    happened, what to do, and which job and stage, so a copied summary
    identifies itself."""
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        advice = "The disk is full. Free space under projects/ and retry; no take was kept."
    elif "out of memory" in lowered or type(exc).__name__ == "OutOfMemoryError":
        advice = (
            "The GPU ran out of memory. Close other GPU programs, or use a shorter "
            "duration or the iterate profile, then retry."
        )
    elif "checkpoint" in lowered and ("not found" in lowered or "missing" in lowered):
        advice = "Model files are missing. Run the model download step, then retry."
    else:
        advice = ""
    if len(text) > 1500:
        text = text[:1500] + "..."
    where = f"[job {job_id}, stage {stage or 'queued'}]"
    return " ".join(part for part in (text, advice, where) if part)


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
    # Set when generate_music has returned and the take is being written.
    # The deadline must not cover the MP3 encode or the meta/LRC write: those
    # can outlast a short remaining budget and a kill mid-write leaves a
    # partial take. LoRA training never sets this, so its budget still covers
    # the whole call.
    generation_done = threading.Event()
    # The heartbeat lease stays fresh while this process is alive, which is
    # what hides a hung generate from reclaim_stale_jobs. deadline_sec is the
    # bound on that lease: past it the loop marks the job error and exits.
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(job_id, stop_heartbeat),
        kwargs={
            "deadline_sec": job_queue.job_timeout_sec(action),
            "generation_done": generation_done,
        },
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        if action == "train_lora":
            # train_lora has no take/DiT-swap shape at all (no source audio
            # generation, no plan) -- it gets its own worker entry point,
            # and keeps the whole-job project lock it always had.
            with storage.project_lifecycle_lock(project_id):
                _run_train_lora_job(job_id, project_id, payload)
        else:
            # SPEC.md sec 16.5: no project lock while rendering. The request
            # was snapshotted at enqueue and the commit takes the lock
            # briefly, so plan edits never wait behind inference.
            _run_generate_shaped_job(
                job_id,
                project_id,
                action,
                dit_profile,
                payload,
                generation_done=generation_done,
                attempt_token=job.get("attempt_token"),
                attempt=job.get("attempt"),
                resolved=job.get("resolved"),
            )
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


def _discard_lora(project_id: str, lora_id: str | None) -> None:
    if lora_id is None:
        return
    lora_path = storage.lora_dir(project_id, lora_id)
    if lora_path.exists():
        shutil.rmtree(lora_path, ignore_errors=True)


def _run_train_lora_job(job_id: str, project_id: str, payload: dict[str, Any]) -> None:
    # Training has no DiT swap or generate_music call, so the only stages
    # that exist for it are skipped (SPEC.md sec 15.7). Cancel is still
    # cooperative: a request that lands during training is honored when the
    # call returns, and the pack is not kept.
    if cancel_requested(job_id):
        mark_cancelled(job_id)
        return
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
        if cancel_requested(job_id):
            _discard_lora(project_id, lora_id)
            mark_cancelled(job_id)
            return
        storage.write_lora_meta(project_id, lora_id, meta)
        if not finish_done(job_id, lora_id=lora_id):
            _discard_lora(project_id, lora_id)
            mark_cancelled(job_id)
    except JobCancelled:
        _discard_lora(project_id, lora_id)
        mark_cancelled(job_id)
    except Exception as exc:
        if cancel_requested(job_id):
            _discard_lora(project_id, lora_id)
            mark_cancelled(job_id)
            return
        if lora_id is not None:
            try:
                storage.write_lora_meta(
                    project_id, lora_id, _error_lora_meta(lora_id, payload, str(exc))
                )
            except Exception:
                pass
        _set_status(job_id, "error", lora_id=lora_id, error=str(exc))


def _snapshot_plan(creation_record: dict[str, Any]) -> dict[str, Any]:
    """The plan fields frozen in the creation input, for a backend that
    reads `plan` rather than `job["creation"]` (a Remote GPU host built
    before SPEC.md sec 16). Never the live plan."""
    plan = {key: creation_record.get(key) for key in creation.TRACKED_FIELDS}
    plan.update(
        query=creation_record.get("query", ""),
        instrumental=creation_record.get("instrumental", False),
        caption_rewrite=creation_record.get("caption_rewrite", False),
        creation_mode=creation_record.get("mode"),
        field_origins=creation_record.get("origins"),
        revision=creation_record.get("plan_revision", 0),
        negative=[],
        sections=[],
    )
    return plan


def committed_take_for_job(project_id: str, job_id: str) -> str | None:
    """The take already committed for this logical job, if any. A retry
    that finds one finishes with it instead of publishing a second
    (SPEC.md sec 16.5)."""
    try:
        takes = storage.list_takes(project_id)
    except storage.ProjectNotFound:
        return None
    for take in takes:
        if take.get("job_id") == job_id and take.get("error") is None:
            return str(take["id"])
    return None


def _resolved_record(
    brief: dict[str, Any], seed: int, dit_profile: str, attempt: int | None
) -> dict[str, Any]:
    return {
        "schema": job_queue.RESOLVED_SCHEMA,
        "brief": brief,
        "seed": seed,
        "dit_profile": dit_profile,
        "checkpoint": job_queue.DIT_CHECKPOINTS.get(dit_profile),
        "inference_steps": job_queue.GENERATION_STEPS.get(dit_profile),
        "guidance_scale": job_queue.GENERATION_GUIDANCE_SCALE.get(dit_profile),
        "lm_model": DEFAULT_LM,
        "frozen_at": _now(),
        "attempt": attempt,
    }


def _describe_take_files(directory: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in ("mix.wav", "mix.mp3", "lyrics.lrc"):
        path = directory / name
        if path.is_file():
            files[name] = artifacts.describe_file(path)
    return files


def _complete_provenance(
    meta: dict[str, Any],
    *,
    job_id: str,
    project_id: str,
    take_id: str,
    payload: dict[str, Any],
    attempt: int | None,
    started_at: str,
    rendered_at: str,
) -> dict[str, Any]:
    """Join the backend's half of the provenance (brief, request, actual)
    with what only the queue knows (SPEC.md sec 16.4)."""
    raw = meta.get("provenance")
    backend_half: dict[str, Any] = raw if isinstance(raw, dict) else {}
    submission = payload.get("submission") or {}
    files = _describe_take_files(storage.staging_dir(project_id, take_id))
    mix = files.get("mix.wav") or files.get("mix.mp3")
    meta = dict(meta)
    meta["job_id"] = job_id
    meta["provenance"] = {
        "schema": PROVENANCE_SCHEMA,
        "complete": bool(backend_half.get("brief")) and mix is not None,
        "job": {
            "id": job_id,
            "attempt": attempt,
            "submission_id": submission.get("submission_id"),
            "retry_of": submission.get("retry_of"),
        },
        "requested": {
            "seed_policy": submission.get("seed_policy"),
            "seed": submission.get("requested_seed"),
            "dit_profile": submission.get("dit_profile"),
            "creation_mode": (payload.get("creation") or {}).get("mode"),
            "plan_revision": (payload.get("creation") or {}).get("plan_revision"),
            "preset": (payload.get("creation") or {}).get("preset"),
        },
        "brief": backend_half.get("brief"),
        "request": backend_half.get("request"),
        "actual": backend_half.get("actual"),
        "actual_origins": backend_half.get("actual_origins"),
        "model": {
            "dit_profile": meta.get("dit_profile"),
            "checkpoint": job_queue.DIT_CHECKPOINTS.get(str(meta.get("dit_profile"))),
            "lm_model": DEFAULT_LM,
            "backend": os.environ.get("LYRE_WORKER", "acestep"),
            "ace_step_revision": submission.get("ace_step_revision"),
            **(backend_half.get("model") or {}),
        },
        "app": {
            "version": submission.get("app_version") or artifacts.app_version(),
            "revision": submission.get("app_revision"),
        },
        "timestamps": {
            "submitted_at": submission.get("submitted_at"),
            "started_at": started_at,
            "rendered_at": rendered_at,
            "committed_at": None,
        },
        "files": files,
    }
    if mix is not None and mix.get("duration_sec") is not None:
        # The measured length of the file, not the requested one.
        meta["duration_sec"] = mix["duration_sec"]
    return meta


def _run_generate_shaped_job(
    job_id: str,
    project_id: str,
    action: str,
    dit_profile: str,
    payload: dict[str, Any],
    generation_done: threading.Event | None = None,
    *,
    attempt_token: str | None = None,
    attempt: int | None = None,
    resolved: dict[str, Any] | None = None,
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

    def _on_stage(stage: str) -> None:
        # `writing_take` is the worker's signal that generate_music returned.
        # Lift the watchdog before the MP3 encode and the file move.
        if stage == "writing_take" and generation_done is not None:
            generation_done.set()
        set_stage(job_id, stage)

    def _cancel_check() -> bool:
        return cancel_requested(job_id)

    def _cancel() -> None:
        mark_cancelled(job_id, attempt_token=attempt_token)

    take_id: str | None = None
    started_at = _now()
    try:
        # Set before the worker is entered: a cancel that landed while the
        # job was queued-but-just-claimed never calls generate_music.
        if _cancel_check():
            _cancel()
            return
        # A retry after a crash that happened between committing the take
        # and marking the job done: the take exists, so finish with it and
        # render nothing (SPEC.md sec 16.5).
        existing = committed_take_for_job(project_id, job_id)
        if existing is not None:
            finish_done(job_id, take_id=existing, attempt_token=attempt_token)
            return
        worker_fn = _resolve_worker()
        creation_record = payload.get("creation")
        # SPEC.md sec 16.4: render the snapshot taken at enqueue. Only a
        # caller that bypassed enqueue_job reaches the live-plan fallback.
        plan = _snapshot_plan(creation_record) if creation_record else storage.load_plan(project_id)
        src_audio = _resolve_source_audio(project_id, action, payload)
        if _cancel_check():
            _cancel()
            return
        # The concrete synthesis seed: frozen by an earlier attempt, fixed
        # by the request, or drawn here so it can be frozen before the DiT
        # runs and reused by any automatic retry.
        if resolved is not None:
            seed = int(resolved["seed"])
        else:
            requested = payload.get("seed", -1)
            seed = creation.draw_seed() if requested in (None, -1) else int(requested)

        def _on_resolved(brief: dict[str, Any]) -> None:
            if resolved is None:
                freeze_resolved(
                    job_id, attempt_token, _resolved_record(brief, seed, dit_profile, attempt)
                )

        _on_stage("loading_model")
        take_id, tdir = storage.allocate_staged_take(
            project_id, {"job_id": job_id, "attempt_token": attempt_token}
        )
        worker_kwargs: dict[str, Any] = {
            "on_dit_loaded": _publish_dit_loaded,
            "on_stage": _on_stage,
            "cancel_check": _cancel_check,
            "on_resolved": _on_resolved,
        }
        if "on_remote_handle" in _parameters(worker_fn):
            worker_kwargs["on_remote_handle"] = lambda handle: record_remote_handle(
                job_id, attempt_token, handle
            )
        meta, plan_patch, lrc_text = worker_fn(
            job={
                **payload,
                "job_id": job_id,
                "action": action,
                "dit_profile": dit_profile,
                "src_audio": src_audio,
                "seed": seed,
                "resolved": resolved,
            },
            plan=plan,
            take_id=take_id,
            take_dir=tdir,
            **worker_kwargs,
        )
        rendered_at = _now()
        # The worker should already have reported `writing_take`. Set it
        # again so the meta/LRC/plan writes below are outside the deadline
        # even if a backend forgot the stage.
        if generation_done is not None:
            generation_done.set()
        if _cancel_check():
            storage.discard_staged_take(project_id, take_id)
            _cancel()
            return
        _on_stage("writing_take")
        # Everything lands in the staging directory first; nothing is
        # visible as a take until the commit renames it into takes/.
        if lrc_text is not None:
            storage.write_staged_lrc(project_id, take_id, lrc_text)
        meta = _complete_provenance(
            meta,
            job_id=job_id,
            project_id=project_id,
            take_id=take_id,
            payload=payload,
            attempt=attempt,
            started_at=started_at,
            rendered_at=rendered_at,
        )
        if not meta["provenance"]["files"]:
            raise RuntimeError("the backend reported success but wrote no audio")
        meta["provenance"]["timestamps"]["committed_at"] = _now()
        storage.write_staged_meta(project_id, take_id, meta)

        outcome = _commit_take(
            job_id,
            attempt_token,
            project_id,
            take_id,
            plan_patch=plan_patch,
            plan_revision=(creation_record or {}).get("plan_revision"),
        )
        if outcome == "cancelled":
            take_id = None
            _cancel()
            return
        if outcome == "project_deleted":
            take_id = None
            _set_status(
                job_id,
                "error",
                error=describe_failure(
                    RuntimeError("project deleted while this job was rendering; no take was kept"),
                    job_id=job_id,
                    stage="writing_take",
                ),
                attempt_token=attempt_token,
            )
            return
        if outcome != take_id:
            return  # an earlier attempt's take was already committed
        take_id = None  # committed: nothing below may discard or overwrite it
    except JobFenced:
        # Another attempt owns this job now. Publish nothing and leave the
        # row to its owner.
        if take_id is not None:
            storage.discard_staged_take(project_id, take_id)
    except JobCancelled:
        if take_id is not None:
            storage.discard_staged_take(project_id, take_id)
        _cancel()
    except Exception as exc:
        if take_id is not None:
            storage.discard_staged_take(project_id, take_id)
        if _cancel_check():
            _cancel()
            return
        message = describe_failure(exc, job_id=job_id, stage=_current_stage(job_id))
        if take_id is not None:
            try:
                error_meta = _error_take_meta(take_id, action, dit_profile, payload, message)
                error_meta["job_id"] = job_id
                storage.write_error_take(project_id, take_id, error_meta)
            except Exception:
                # Writing the take's error metadata is a courtesy (surfaces
                # the failure on the take, not just the job row) -- if disk
                # I/O itself is failing (full disk, permissions, lock), or
                # the project is gone, that must not stop the job row below
                # from being marked `error`.
                take_id = None
        _set_status(job_id, "error", take_id=take_id, error=message, attempt_token=attempt_token)


def _commit_take(
    job_id: str,
    attempt_token: str | None,
    project_id: str,
    take_id: str,
    *,
    plan_patch: dict[str, Any] | None = None,
    plan_revision: int | None = None,
) -> str:
    """Publish a staged take (SPEC.md sec 16.5). Under the project lock and
    one `BEGIN IMMEDIATE`: check the attempt still owns the running job,
    that it was not cancelled, that the project still exists and is not
    being deleted, and that no take was committed for this job before; then
    rename the staged directory into takes/ and mark the job `done` in the
    same transaction. A crash between the rename and the commit leaves a
    take whose `job_id` the next attempt finds and finishes with.

    A planned brief is merged into the draft inside the same critical
    section, before the job reads `done`, so a client that sees the job
    finish also sees the draft it produced. The merge only applies when the
    draft is still at the submitted revision (SPEC.md sec 16.3), and a
    merge that fails never fails the committed take.

    Returns the committed take id (this one, or an earlier attempt's),
    `"cancelled"`, or `"project_deleted"`. Raises `JobFenced` when another
    attempt owns the job. Every outcome but this take's own commit discards
    the staged files."""
    with storage.project_lifecycle_lock(project_id), closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT status, attempt_token, cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or (attempt_token is not None and row["attempt_token"] != attempt_token)
            ):
                raise JobFenced(job_id)
            if row["cancel_requested"]:
                conn.rollback()
                storage.discard_staged_take(project_id, take_id)
                return "cancelled"
            deleting = conn.execute(
                "SELECT 1 FROM project_deletions WHERE project_id = ?", (project_id,)
            ).fetchone()
            try:
                storage.load_project(project_id)
                exists = deleting is None
            except storage.ProjectNotFound:
                exists = False
            if not exists:
                conn.rollback()
                storage.discard_staged_take(project_id, take_id)
                return "project_deleted"
            earlier = committed_take_for_job(project_id, job_id)
            final = earlier or take_id
            if earlier is None:
                storage.promote_staged_take(project_id, take_id)
                if plan_patch is not None:
                    _merge_planned_brief(job_id, project_id, plan_patch, plan_revision)
            conn.execute(
                "UPDATE jobs SET status = 'done', take_id = ?, error = NULL, "
                "stage = 'writing_take', updated_at = ? WHERE id = ?",
                (final, _now(), job_id),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            storage.discard_staged_take(project_id, take_id)
            raise
        if earlier is not None:
            storage.discard_staged_take(project_id, take_id)
            return earlier
        # A finished job never replaces the user's kept take; the first
        # playable take of a project becomes active (SPEC.md sec 16.5).
        storage.set_active_take_if_unset(project_id, take_id)
        return take_id


def _merge_planned_brief(
    job_id: str, project_id: str, plan_patch: dict[str, Any], plan_revision: int | None
) -> None:
    try:
        storage.merge_plan_patch(project_id, plan_patch, expected_revision=plan_revision)
    except Exception as exc:
        # The take keeps its own brief; only the draft did not pick it up.
        print(f"job {job_id}: planned brief not merged into the draft: {exc}")


def cleanup_staging() -> list[str]:
    """Remove staged takes whose owning attempt is no longer running: a
    crashed, reclaimed, cancelled, or finished attempt. Called by the worker
    between jobs, so it never races its own render. Returns removed ids."""
    removed: list[str] = []
    for project_id, take_id, owner in storage.list_staged_takes():
        alive = False
        if owner and owner.get("job_id"):
            try:
                row = job_queue.read_job(str(owner["job_id"]))
            except Exception:
                row = None
            if row is not None and row["status"] == "running":
                with closing(_connect()) as conn:
                    token = conn.execute(
                        "SELECT attempt_token FROM jobs WHERE id = ?", (owner["job_id"],)
                    ).fetchone()
                alive = token is not None and token["attempt_token"] == owner.get("attempt_token")
        if not alive:
            storage.discard_staged_take(project_id, take_id)
            removed.append(take_id)
    return removed


def _parameters(fn: Any) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _current_stage(job_id: str) -> str | None:
    try:
        return job_queue.get_job(job_id).get("stage")
    except Exception:
        return None


def process_one_queued_job() -> bool:
    """Claim and run one queued job. Returns False if the queue was empty."""
    job = claim_next_queued_job()
    if job is None:
        return False
    run_claimed_job(job)
    return True
