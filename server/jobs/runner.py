"""Executing a claimed job. Runs only in the worker process.

`run_claimed_job` owns the failure contract from SPEC.md sec 10 point 5: a
job that fails writes a spec-shaped meta.json carrying the error and marks
the row `error`, rather than leaving it `running` or losing the failure.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from server import storage
from server.jobs import queue as job_queue
from server.jobs.db import _now
from server.jobs.errors import JobCancelled, JobError
from server.jobs.queue import (
    HEARTBEAT_INTERVAL_SEC,
    _heartbeat_loop,
    _set_status,
    cancel_requested,
    claim_next_queued_job,
    finish_done,
    mark_cancelled,
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
        with storage.project_lifecycle_lock(project_id):
            if action == "train_lora":
                # train_lora has no take/DiT-swap shape at all (no source audio
                # generation, no plan) -- it gets its own worker entry point.
                _run_train_lora_job(job_id, project_id, payload)
            else:
                _run_generate_shaped_job(
                    job_id,
                    project_id,
                    action,
                    dit_profile,
                    payload,
                    generation_done=generation_done,
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


def _run_generate_shaped_job(
    job_id: str,
    project_id: str,
    action: str,
    dit_profile: str,
    payload: dict[str, Any],
    generation_done: threading.Event | None = None,
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

    take_id: str | None = None
    try:
        # Set before the worker is entered: a cancel that landed while the
        # job was queued-but-just-claimed never calls generate_music.
        if _cancel_check():
            mark_cancelled(job_id)
            return
        worker_fn = _resolve_worker()
        plan = storage.load_plan(project_id)
        src_audio = _resolve_source_audio(project_id, action, payload)
        if _cancel_check():
            mark_cancelled(job_id)
            return
        _on_stage("loading_model")
        take_id, tdir = storage.allocate_take_dir(project_id)
        meta, plan_patch, lrc_text = worker_fn(
            job={
                **payload,
                "job_id": job_id,
                "action": action,
                "dit_profile": dit_profile,
                "src_audio": src_audio,
            },
            plan=plan,
            take_id=take_id,
            take_dir=tdir,
            on_dit_loaded=_publish_dit_loaded,
            on_stage=_on_stage,
            cancel_check=_cancel_check,
        )
        # The worker should already have reported `writing_take`. Set it
        # again so the meta/LRC/plan writes below are outside the deadline
        # even if a backend forgot the stage.
        if generation_done is not None:
            generation_done.set()
        if _cancel_check():
            storage.discard_unfinished_take(project_id, take_id)
            mark_cancelled(job_id)
            return
        # Written before meta.json so a client polling right after the job
        # flips to "done" never sees meta["has_lrc"] True while
        # lyrics.lrc itself doesn't exist yet (SPEC.md sec 7 / sec 12
        # Phase 4). If this raises, the except block below writes an error
        # take_meta instead -- same as a merge_plan_patch failure further
        # down.
        _on_stage("writing_take")
        if lrc_text is not None:
            storage.write_take_lrc(project_id, take_id, lrc_text)
        storage.write_take_meta(project_id, take_id, meta)
        if plan_patch is not None:
            # A planned brief updates the editable draft only if the draft is
            # still at the revision this job was submitted from: a newer edit
            # wins, and the take keeps its own brief either way (SPEC.md sec
            # 16.3). The merge itself re-reads and writes under storage's
            # per-project lock, so a concurrent PUT cannot interleave.
            creation_record = payload.get("creation") or {}
            storage.merge_plan_patch(
                project_id,
                plan_patch,
                expected_revision=creation_record.get("plan_revision"),
            )
        if _cancel_check():
            storage.discard_unfinished_take(project_id, take_id)
            mark_cancelled(job_id)
            return
        # Promote the take to active only after every fallible persistence
        # step above has actually succeeded (reviewer-flagged): if
        # merge_plan_patch raised, the except block below marks this take
        # `error`, and active_take_id must not be left pointing at it.
        storage.set_active_take(project_id, take_id)
        if not finish_done(job_id, take_id=take_id):
            storage.discard_unfinished_take(project_id, take_id)
            mark_cancelled(job_id)
    except JobCancelled:
        if take_id is not None:
            storage.discard_unfinished_take(project_id, take_id)
        mark_cancelled(job_id)
    except Exception as exc:
        if _cancel_check():
            if take_id is not None:
                storage.discard_unfinished_take(project_id, take_id)
            mark_cancelled(job_id)
            return
        if take_id is not None:
            try:
                storage.write_take_meta(
                    project_id,
                    take_id,
                    _error_take_meta(take_id, action, dit_profile, payload, str(exc)),
                )
            except Exception:
                # Writing the take's error metadata is a courtesy (surfaces
                # the failure on the take, not just the job row) -- if disk
                # I/O itself is failing (full disk, permissions, lock), that
                # must not stop the job row below from being marked `error`,
                # or the job is left `running` until a later worker process
                # times it out via reclaim_stale_jobs instead of failing fast.
                pass
        _set_status(job_id, "error", take_id=take_id, error=str(exc))


def process_one_queued_job() -> bool:
    """Claim and run one queued job. Returns False if the queue was empty."""
    job = claim_next_queued_job()
    if job is None:
        return False
    run_claimed_job(job)
    return True
