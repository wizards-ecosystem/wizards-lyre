"""Mocked ACE-Step worker used by tests and local dev without a GPU.

Writes a tiny silent WAV plus a spec-shaped meta.json (SPEC.md sec 7.3).
Never imports torch, CUDA, or acestep -- see SPEC.md sec 10 and 11.

Also stands in for the 5 Hz LM planner (SPEC.md sec 16.3): a `from_idea`
generate gets a deterministic `create_sample` / `format_sample` result,
merged by the same `server.generation.brief` resolver the real adapter
uses, so precedence and the GenerationParams creative fields are identical
between the two backends.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from server.generation import brief as creation
from worker.acestep_worker.settings import GENERATION_GUIDANCE_SCALE, GENERATION_STEPS
from worker.playback import write_mix_mp3

SAMPLE_RATE = 8000
DURATION_SEC = 0.5

TASK_TYPE_BY_ACTION = {
    "generate": "text2music",
    "cover": "cover",
    "repaint": "repaint",
    "extract": "extract",
    "lego": "lego",
    "complete": "complete",
}


def _write_silent_wav(path: Path) -> float:
    n_frames = int(SAMPLE_RATE * DURATION_SEC)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * n_frames)
    return n_frames / SAMPLE_RATE


def _repaint_meta(job: dict[str, Any]) -> dict | None:
    if job.get("action") != "repaint":
        return None
    return {
        "start": job.get("repainting_start", 0),
        "end": job.get("repainting_end", -1),
    }


def supports_dit_profile(dit_profile: str) -> tuple[bool, str | None]:
    """The mock never loads a real model, so every profile is trivially
    "supported" -- capability rejection (SPEC.md sec 4.1/8.1) is exercised
    against `worker.acestep_worker`, which actually has GPU/offload limits."""
    return True, None


DEFAULT_SIMULATED_DIT_PROFILE = "iterate"

# Tracks a simple simulated "loaded" flag, mirroring worker.acestep_worker's
# real _STATE closely enough that tests can exercise worker/run_worker.py's
# republish-after-recovery behavior (SPEC.md sec 8) against this mock too:
# None until initialize_worker() or run_job() has "loaded" something --
# never a hardcoded constant, or a test simulating a startup failure (by
# monkeypatching initialize_worker) would still see this report ready.
_simulated_loaded_dit_profile: str | None = None


def initialize_worker() -> tuple[bool, str]:
    """Worker-startup readiness (SPEC.md sec 10 point 1). The mock has no
    real CUDA/model to detect or load, so it is always immediately ready --
    real startup detection/preload is exercised against
    `worker.acestep_worker.initialize_worker`."""
    global _simulated_loaded_dit_profile
    _simulated_loaded_dit_profile = DEFAULT_SIMULATED_DIT_PROFILE
    return True, "mock worker: no GPU/model to load, always ready"


def get_loaded_dit_profile() -> str | None:
    """Which profile this mock has simulated "loading", or None if neither
    `initialize_worker` nor `run_job` has run yet (SPEC.md sec 8
    `dit_loaded`)."""
    return _simulated_loaded_dit_profile


# What the stand-in planner suggests for every metadata field, whatever the
# request. Fixed on purpose: a test that sets an explicit value can see the
# suggestion lose to it (SPEC.md sec 16.3 precedence rule 1).
MOCK_PLANNER_SUGGESTIONS = {
    "bpm": 120,
    "keyscale": "C major",
    "duration": 30.0,
    "timesignature": "4",
}
# What the stand-in CoT fills for metas a brief leaves unset, mirroring
# upstream writing `cot_*` onto the per-audio params.
MOCK_COT_FILL = {
    "cot_bpm": 96,
    "cot_keyscale": "A minor",
    "cot_timesignature": "4",
    "cot_duration": 42.0,
    "cot_vocal_language": "en",
}


def _mock_planner(request: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for upstream `create_sample` / `format_sample`,
    returning upstream's field names. Precedence is applied afterwards by the
    same `brief.apply_planner` the real adapter uses."""
    if request["planner"] == "format_sample":
        user_metadata = request.get("user_metadata") or {}
        return {
            "caption": f"auto: {request['caption']}",
            "lyrics": request["lyrics"],
            "language": user_metadata.get("language") or "en",
            **MOCK_PLANNER_SUGGESTIONS,
        }
    query = request["query"]
    return {
        "caption": f"auto: {query}",
        "lyrics": "[Instrumental]" if request["instrumental"] else f"[Verse]\n{query}",
        "language": request.get("vocal_language") or "en",
        **MOCK_PLANNER_SUGGESTIONS,
    }


def _abort_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is None or not cancel_check():
        return
    from server.jobs.errors import JobCancelled

    raise JobCancelled()


def _note_stage(on_stage: Callable[[str], None] | None, stage: str) -> None:
    if on_stage is not None:
        on_stage(stage)


def run_job(
    job: dict[str, Any],
    plan: dict[str, Any],
    take_id: str,
    take_dir: Path,
    on_dit_loaded: Callable[[str], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_resolved: Callable[[dict], None] | None = None,
) -> tuple[dict, dict | None, str | None]:
    """Run one mocked job. Returns `(take_meta, plan_patch, lrc_text)`;
    `plan_patch` is a delta of the fields to persist when this job filled the
    plan in (simple mode), else None. `server.jobs` merges it onto the plan
    that's current on disk when the job finishes, not the stale snapshot
    passed in as `plan`. `lrc_text` is always None here -- this mock never
    calls real ACE-Step, so it has no lyric-timestamp capability to fake
    (SPEC.md sec 12 Phase 4 coverage against the real adapter lives in
    tests/test_acestep_worker_adapter.py).

    `take_dir` must already be inside the projects/ path jail; this function
    only ever writes files inside it.

    `on_dit_loaded`, if given, is called once the (simulated) DiT is
    "loaded" for this job -- before any of the (simulated) generation work
    below -- so a caller can publish worker_status immediately instead of
    only after the whole job finishes (SPEC.md sec 4.3: swapping between
    iterate and studio_ops should show as "loading" only for the swap
    itself, not for the inference that follows it).

    `on_stage` and `cancel_check` implement SPEC.md sec 15.7. Stages that
    do not apply (planning, when this is not a simple-mode generate) are
    skipped. A cancel observed before the silent WAV is written never
    writes it; a cancel observed after the write raises `JobCancelled` so
    the caller can drop the file.
    """
    global _simulated_loaded_dit_profile
    _abort_if_cancelled(cancel_check)
    _note_stage(on_stage, "loading_model")
    # A successful job run implies this profile is now "loaded", same as
    # worker.acestep_worker's _ensure_loaded -- this is what makes a worker
    # that reported unavailable at startup show as recovered afterward.
    _simulated_loaded_dit_profile = job.get("dit_profile", DEFAULT_SIMULATED_DIT_PROFILE)
    if on_dit_loaded is not None:
        on_dit_loaded(_simulated_loaded_dit_profile)
    _abort_if_cancelled(cancel_check)

    take_dir.mkdir(parents=True, exist_ok=True)

    action = job["action"]
    intent = job.get("creation") or creation.validate_creation(
        creation.creation_input(plan, action=action)
    )
    resolved = job.get("resolved")
    request = None if resolved else creation.planner_request(intent)
    if resolved:
        brief = resolved["brief"]
    elif request is not None:
        _note_stage(on_stage, "planning")
        brief = creation.apply_planner(
            intent,
            _mock_planner(request),
            {"name": request["planner"], "lm_model": "mock", "deterministic": True},
        )
        _abort_if_cancelled(cancel_check)
    else:
        brief = creation.brief_without_planning(intent)
    if on_resolved is not None:
        on_resolved(brief)
    fields = creation.generation_fields(brief, action)

    _note_stage(on_stage, "rendering")
    _abort_if_cancelled(cancel_check)
    seed = job.get("seed", -1)
    if seed is None or seed == -1:
        seed = creation.draw_seed()

    audio_path = take_dir / "mix.wav"
    duration = _write_silent_wav(audio_path)
    # Same point as the real worker: generation is done, so the watchdog
    # lifts before the playback encode (server.jobs.runner).
    _note_stage(on_stage, "writing_take")
    # Same playback copy the real worker writes (SPEC.md sec 7). Missing
    # ffmpeg leaves the WAV only; the artifact list then omits mix.mp3.
    write_mix_mp3(audio_path)
    _abort_if_cancelled(cancel_check)

    cot_fill = MOCK_COT_FILL if fields["use_cot_metas"] else {}
    lm_metadata = {"caption": f"cot: {brief['caption']}", "vocal_language": "en"}
    actual, actual_origins = creation.actual_values(brief, fields, cot_fill, lm_metadata)
    plan_patch = creation.plan_patch_from_brief(brief, job.get("job_id") or "")

    meta = {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[action],
        "dit_profile": job["dit_profile"],
        "seed": seed,
        "duration_sec": duration,
        "caption": actual["caption"],
        "lyrics": actual["lyrics"],
        "bpm": actual["bpm"],
        "keyscale": actual["keyscale"],
        "created_at": datetime.now(UTC).isoformat(),
        "score": None,
        "has_lrc": False,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
        # SPEC.md sec 4.4 "LoRA train / load": no real GPU/adapter loading
        # here (see the module docstring) -- just record what was requested
        # so tests can assert lora_id actually flows from the job body
        # through server.jobs to the take, without needing the real
        # acestep.core.generation.handler.lora.* API worker.acestep_worker
        # wraps.
        "lora_id": job.get("lora_id"),
        "favorite": False,
        "notes": "",
        # Same backend half of the take provenance as the real adapter.
        "provenance": {
            "brief": brief,
            "request": {
                "task_type": TASK_TYPE_BY_ACTION[action],
                **fields,
                "seed": job.get("seed", -1),
                "inference_steps": GENERATION_STEPS[job["dit_profile"]],
                "guidance_scale": GENERATION_GUIDANCE_SCALE[job["dit_profile"]],
            },
            "actual": {**actual, "seed": seed},
            "actual_origins": actual_origins,
            "runtime": {"device": "mock"},
        },
    }
    return meta, plan_patch, None


def train_lora(
    job: dict[str, Any],
    project_id: str,
    lora_id: str,
    lora_dir: Path,
    source_paths: list[Path],
) -> dict:
    """Mocked LoRA style-pack training (SPEC.md sec 4.4/7): writes a tiny
    fake adapter file plus a spec-shaped meta.json, no acestep/CUDA/torch
    import. Same call shape as worker.acestep_worker.train_lora so
    server.jobs can dispatch either backend."""
    lora_dir.mkdir(parents=True, exist_ok=True)
    (lora_dir / "adapter.bin").write_bytes(b"\x00" * 16)
    return {
        "id": lora_id,
        "name": job.get("name") or "Untitled LoRA",
        "created_at": datetime.now(UTC).isoformat(),
        "source_take_count": len(source_paths),
        "base_checkpoint": "studio_ops",
        "dit_profile": "studio_ops",
        # A truthy, present `status` plus a null `error` is what
        # server.jobs._resolve_lora treats as "finished training
        # successfully" (worker.acestep_worker.train_lora's real `status`
        # is a free-form progress string from ACE-Step's own trainer, e.g.
        # "epoch 2/10" -- this mock has no real trainer to report from, so
        # it reports a fixed, always-truthy string instead).
        "status": "done",
        "error": None,
    }
