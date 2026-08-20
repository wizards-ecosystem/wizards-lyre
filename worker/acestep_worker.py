"""Real ACE-Step 1.5 worker (SPEC.md sec 4 and sec 10).

This is the production job backend (`server.jobs` selects it whenever
`BARD_WORKER` is unset or `acestep`). `worker.mock_worker` is the test/local
-dev-only stand-in -- see SPEC.md sec 11.

`acestep` and CUDA are imported lazily, inside functions, never at module
scope. Importing this module (e.g. because `server.jobs` looked up the
backend) must stay safe on a machine with no GPU and no ACE-Step install;
only calling `run_job` reaches for either. On a machine without ACE-Step
installed, `run_job` raises `WorkerUnavailable`, which `server.jobs` catches
and records as the job's `error` -- a missing/broken GPU stack fails the job,
not the HTTP process (sec 10 point 5). Run this worker as its own OS process
(`python -m worker.run_worker`, see that module) so a native crash inside
ACE-Step/CUDA can't take the FastAPI server down with it either.
"""

from __future__ import annotations

import inspect
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One GPU occupant: one loaded DiT + one LM, jobs serialize (SPEC.md sec 4.3).
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"dit_profile": None, "handler": None, "lm": None}

DIT_CHECKPOINTS = {
    "iterate": "acestep-v15-turbo",
    "polish": "acestep-v15-sft",
    "quality": "acestep-v15-xl-turbo",
    "studio_ops": "acestep-v15-base",
}
DEFAULT_LM = "acestep-5Hz-lm-1.7B"
LM_BACKEND = "pt"  # SPEC.md sec 4.2: vLLM is not a reliable native-Windows backend.

# SPEC.md sec 4.1: XL turbo (`quality`) needs CPU offload on a 16 GB card;
# every other profile fits without it.
CPU_OFFLOAD_PROFILES = {"quality"}

TASK_TYPE_BY_ACTION = {
    "generate": "text2music",
    "cover": "cover",
    "repaint": "repaint",
    "extract": "extract",
    "lego": "lego",
    "complete": "complete",
}


class WorkerUnavailable(RuntimeError):
    """ACE-Step is not installed / no usable GPU in this environment."""


def _import_acestep():
    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler
    except ImportError as exc:
        raise WorkerUnavailable(
            "acestep is not installed in this environment. Install ACE-Step 1.5 "
            "and download weights (SPEC.md sec 4 and 13), or set BARD_WORKER=mock "
            "for local dev/tests without a GPU."
        ) from exc
    return AceStepHandler, GenerationParams, LLMHandler, generate_music


def _handler_supports_cpu_offload(AceStepHandler: Any) -> bool:
    try:
        params = inspect.signature(AceStepHandler.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "cpu_offload" in params


def supports_dit_profile(dit_profile: str) -> tuple[bool, str | None]:
    """Whether this worker can currently load `dit_profile`. SPEC.md sec 4.1:
    'quality' (XL, 4B) requires CPU offload on a 16 GB GPU; SPEC.md sec 8.1:
    'Reject quality if worker reports it cannot load XL with offload.'
    Checked by `server.jobs` at enqueue time so a job never gets stuck
    mid-run risking an uncontrolled OOM; also enforced defensively in
    `_ensure_loaded` for direct worker callers (e.g. scripts/smoke-gpu.py)."""
    if dit_profile not in CPU_OFFLOAD_PROFILES:
        return True, None
    try:
        AceStepHandler, _, _, _ = _import_acestep()
    except WorkerUnavailable as exc:
        return False, str(exc)
    if not _handler_supports_cpu_offload(AceStepHandler):
        return False, (
            f"dit_profile 'quality' ({DIT_CHECKPOINTS['quality']}) requires a "
            "CPU-offload-capable AceStepHandler on a 16 GB GPU; the installed acestep "
            "does not support cpu_offload. Use 'iterate' or 'polish' instead."
        )
    return True, None


def _ensure_loaded(dit_profile: str) -> tuple[Any, Any, Any]:
    """Swap the loaded DiT if needed. Caller holds _LOCK."""
    AceStepHandler, _, LLMHandler, _ = _import_acestep()

    if _STATE["dit_profile"] != dit_profile or _STATE["handler"] is None:
        # Unload the previous DiT before loading the next; never hold two
        # DiTs at once on a 16 GB card (SPEC.md sec 4.3).
        _STATE["handler"] = None
        checkpoint = DIT_CHECKPOINTS[dit_profile]
        if dit_profile in CPU_OFFLOAD_PROFILES:
            ok, reason = supports_dit_profile(dit_profile)
            if not ok:
                raise WorkerUnavailable(reason)
            _STATE["handler"] = AceStepHandler(
                checkpoint=checkpoint, backend=LM_BACKEND, cpu_offload=True
            )
        else:
            _STATE["handler"] = AceStepHandler(checkpoint=checkpoint, backend=LM_BACKEND)
        _STATE["dit_profile"] = dit_profile

    if _STATE["lm"] is None:
        _STATE["lm"] = LLMHandler(model=DEFAULT_LM, backend=LM_BACKEND)

    return _STATE["handler"], _STATE["lm"], _STATE["dit_profile"]


def _repaint_meta(job: dict[str, Any]) -> dict | None:
    if job.get("action") != "repaint":
        return None
    return {
        "start": job.get("repainting_start", 0),
        "end": job.get("repainting_end", -1),
    }


def _is_simple_mode(job: dict[str, Any], plan: dict[str, Any]) -> bool:
    """SPEC.md sec 7.2: Simple mode is "user types query"; caption/lyrics
    stay empty until the LM (thinking=true) fills them in."""
    return (
        job.get("action") == "generate"
        and bool(plan.get("query"))
        and not plan.get("caption")
        and not plan.get("lyrics")
    )


def run_job(
    job: dict[str, Any], plan: dict[str, Any], take_id: str, take_dir: Path
) -> tuple[dict, dict | None]:
    """Run one real ACE-Step job. Returns `(take_meta, plan_patch)`;
    `plan_patch` is the LM-filled plan to persist when this was a simple-mode
    generation, else None (SPEC.md sec 7.2: "Persist the filled plan.").

    `job["src_audio"]` must already be a resolved, jailed filesystem path (or
    None for `generate`) -- `server.jobs` resolves `source_take_id` /
    `upload_path` before calling this, so the worker never has to reach back
    into project storage itself.

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable; `server.jobs`
    catches that and marks the job `error` without crashing the HTTP process.
    """
    _, GenerationParams, _, generate_music = _import_acestep()
    take_dir.mkdir(parents=True, exist_ok=True)

    simple_mode = _is_simple_mode(job, plan)
    # Null/omitted bpm, key, or duration: let ACE-Step's CoT fill them in
    # (SPEC.md sec 7.2, use_cot_metas), same as simple mode's blank caption.
    use_cot_metas = simple_mode or any(
        plan.get(k) is None for k in ("bpm", "keyscale", "duration_sec")
    )

    with _LOCK:
        handler, lm, dit_profile = _ensure_loaded(job["dit_profile"])
        params = GenerationParams(
            task_type=TASK_TYPE_BY_ACTION[job["action"]],
            query=plan.get("query") if simple_mode else None,
            caption=plan.get("caption", ""),
            lyrics=plan.get("lyrics", ""),
            bpm=plan.get("bpm"),
            keyscale=plan.get("keyscale"),
            duration_sec=plan.get("duration_sec"),
            instrumental=plan.get("instrumental", False),
            thinking=simple_mode,
            use_cot_metas=use_cot_metas,
            seed=job.get("seed", -1),
            src_audio=job.get("src_audio"),
            audio_cover_strength=job.get("audio_cover_strength"),
            repainting_start=job.get("repainting_start"),
            repainting_end=job.get("repainting_end"),
            track_name=job.get("track_name"),
        )
        result = generate_music(
            handler=handler,
            lm_handler=lm,
            params=params,
            save_dir=str(take_dir),
        )

    caption = getattr(result, "caption", None) or plan.get("caption", "")
    lyrics = getattr(result, "lyrics", None) or plan.get("lyrics", "")
    bpm = getattr(result, "bpm", None) or plan.get("bpm")
    keyscale = getattr(result, "keyscale", None) or plan.get("keyscale")
    duration_sec = getattr(result, "duration_sec", None) or plan.get("duration_sec")

    plan_patch = None
    if simple_mode:
        # The LM filled caption/lyrics/metas from `query` -- persist them
        # onto plan.json instead of discarding them (SPEC.md sec 7.2).
        plan_patch = {
            **plan,
            "caption": caption,
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": keyscale,
        }

    meta = {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[job["action"]],
        "dit_profile": dit_profile,
        "seed": getattr(result, "seed", job.get("seed", -1)),
        "duration_sec": duration_sec,
        "caption": caption,
        "lyrics": lyrics,
        "bpm": bpm,
        "keyscale": keyscale,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": None,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
    }
    return meta, plan_patch
