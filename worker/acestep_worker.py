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

Adapter notes (installed ACE-Step 1.5 Python API, SPEC.md sec 13 -- "follow
ACE-Step's current API and keep Bard's HTTP schema stable with an
adapter"): `AceStepHandler`/`LLMHandler` are constructed with no arguments
and then explicitly initialized; `GenerationParams` has no `query` field
(simple-mode planning goes through `LLMHandler.create_sample` instead) and
uses `duration`, not `duration_sec`; `generate_music` takes both a
`GenerationParams` and a `GenerationConfig`; its `GenerationResult` reports
`success` plus generated files in `audios` and LM/CoT-filled metadata in
`extra_outputs`, not directly on the result object. Every acestep call below
is wrapped so a further API drift raises `WorkerUnavailable` (job `error`,
not a crash) instead of an unhandled exception.
"""

from __future__ import annotations

import inspect
import shutil
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

# SPEC.md sec 4.1 DiT table: steps / whether CFG is used, per profile.
GENERATION_STEPS = {
    "iterate": 8,
    "polish": 50,
    "quality": 8,
    "studio_ops": 50,
}
GENERATION_USE_CFG = {
    "iterate": False,
    "polish": True,
    "quality": False,
    "studio_ops": True,
}

TASK_TYPE_BY_ACTION = {
    "generate": "text2music",
    "cover": "cover",
    "repaint": "repaint",
    "extract": "extract",
    "lego": "lego",
    "complete": "complete",
}


class WorkerUnavailable(RuntimeError):
    """ACE-Step is not installed, has no usable GPU, or its Python API no
    longer matches this adapter."""


def _import_acestep():
    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler
    except ImportError as exc:
        raise WorkerUnavailable(
            "acestep is not installed in this environment. Install ACE-Step 1.5 "
            "and download weights (SPEC.md sec 4 and 13), or set BARD_WORKER=mock "
            "for local dev/tests without a GPU."
        ) from exc
    return AceStepHandler, GenerationParams, GenerationConfig, LLMHandler, generate_music


def _api_call(step: str, fn, *args, **kwargs):
    """Call into acestep and turn a signature mismatch into `WorkerUnavailable`
    instead of an unhandled crash -- this adapter's field/argument names are
    our best mapping onto ACE-Step 1.5's documented API and may need
    updating if upstream drifts further (SPEC.md sec 13)."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, AttributeError) as exc:
        raise WorkerUnavailable(
            f"acestep API mismatch in {step}: {exc}. Update worker/acestep_worker.py to "
            "match the installed acestep version, or set BARD_WORKER=mock."
        ) from exc


def _handler_supports_cpu_offload(AceStepHandler: Any) -> bool:
    try:
        params = inspect.signature(AceStepHandler.initialize).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return "cpu_offload" in params


def supports_dit_profile(dit_profile: str) -> tuple[bool, str | None]:
    """Whether this worker can currently load `dit_profile`. SPEC.md sec 4.1:
    'quality' (XL, 4B) requires CPU offload on a 16 GB GPU; SPEC.md sec 8.1:
    'Reject quality if worker reports it cannot load XL with offload.'

    Only ever called from this worker process (directly, or via
    `worker/run_worker.py` publishing the result to SQLite for
    `server.jobs` to read) -- the FastAPI server must never import acestep
    itself just to answer this (SPEC.md sec 10 point 4)."""
    if dit_profile not in CPU_OFFLOAD_PROFILES:
        return True, None
    try:
        AceStepHandler, _, _, _, _ = _import_acestep()
    except WorkerUnavailable as exc:
        return False, str(exc)
    if not _handler_supports_cpu_offload(AceStepHandler):
        return False, (
            f"dit_profile 'quality' ({DIT_CHECKPOINTS['quality']}) requires a "
            "CPU-offload-capable AceStepHandler.initialize() on a 16 GB GPU; the "
            "installed acestep does not support cpu_offload. Use 'iterate' or 'polish' "
            "instead."
        )
    return True, None


def _ensure_loaded(dit_profile: str) -> tuple[Any, Any, Any]:
    """Swap the loaded DiT if needed. Caller holds _LOCK."""
    AceStepHandler, _, _, LLMHandler, _ = _import_acestep()

    if _STATE["dit_profile"] != dit_profile or _STATE["handler"] is None:
        # Unload the previous DiT before loading the next; never hold two
        # DiTs at once on a 16 GB card (SPEC.md sec 4.3).
        _STATE["handler"] = None
        checkpoint = DIT_CHECKPOINTS[dit_profile]
        cpu_offload = dit_profile in CPU_OFFLOAD_PROFILES
        if cpu_offload:
            ok, reason = supports_dit_profile(dit_profile)
            if not ok:
                raise WorkerUnavailable(reason)

        handler = _api_call("AceStepHandler()", AceStepHandler)
        init_kwargs: dict[str, Any] = {"checkpoint": checkpoint, "backend": LM_BACKEND}
        if cpu_offload:
            init_kwargs["cpu_offload"] = True
        _api_call("AceStepHandler.initialize", handler.initialize, **init_kwargs)

        _STATE["handler"] = handler
        _STATE["dit_profile"] = dit_profile

    if _STATE["lm"] is None:
        lm = _api_call("LLMHandler()", LLMHandler)
        _api_call(
            "LLMHandler.initialize", lm.initialize, model=DEFAULT_LM, backend=LM_BACKEND
        )
        _STATE["lm"] = lm

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
    stay empty until the LM fills them in."""
    return (
        job.get("action") == "generate"
        and bool(plan.get("query"))
        and not plan.get("caption")
        and not plan.get("lyrics")
    )


def _plan_from_query(lm: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Simple mode (SPEC.md sec 7.2): turn `query` into a full plan via
    ACE-Step's `create_sample`, then merge the result onto `plan`."""
    sample = _api_call(
        "LLMHandler.create_sample",
        lm.create_sample,
        query=plan.get("query"),
        instrumental=plan.get("instrumental", False),
    )

    def _field(name: str, fallback: Any) -> Any:
        if isinstance(sample, dict):
            return sample.get(name, fallback)
        return getattr(sample, name, fallback)

    return {
        **plan,
        "caption": _field("caption", plan.get("caption", "")),
        "lyrics": _field("lyrics", plan.get("lyrics", "")),
        "bpm": _field("bpm", plan.get("bpm")),
        "keyscale": _field("keyscale", plan.get("keyscale")),
        "duration_sec": _field("duration", plan.get("duration_sec")),
    }


def _result_field(container: Any, name: str, fallback: Any) -> Any:
    if isinstance(container, dict):
        return container.get(name, fallback)
    return getattr(container, name, fallback)


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

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable or its API no
    longer matches this adapter; `server.jobs` catches that and marks the
    job `error` without crashing the HTTP process.
    """
    _, GenerationParams, GenerationConfig, _, generate_music = _import_acestep()
    take_dir.mkdir(parents=True, exist_ok=True)

    simple_mode = _is_simple_mode(job, plan)

    with _LOCK:
        handler, lm, dit_profile = _ensure_loaded(job["dit_profile"])

        effective_plan = _plan_from_query(lm, plan) if simple_mode else plan
        # Null/omitted bpm, key, or duration: let ACE-Step's CoT fill them in
        # (SPEC.md sec 7.2, use_cot_metas).
        use_cot_metas = simple_mode or any(
            effective_plan.get(k) is None for k in ("bpm", "keyscale", "duration_sec")
        )

        params = _api_call(
            "GenerationParams",
            GenerationParams,
            task_type=TASK_TYPE_BY_ACTION[job["action"]],
            caption=effective_plan.get("caption", ""),
            lyrics=effective_plan.get("lyrics", ""),
            bpm=effective_plan.get("bpm"),
            keyscale=effective_plan.get("keyscale"),
            duration=effective_plan.get("duration_sec"),
            instrumental=effective_plan.get("instrumental", False),
            thinking=simple_mode,
            use_cot_metas=use_cot_metas,
            seed=job.get("seed", -1),
            src_audio=job.get("src_audio"),
            audio_cover_strength=job.get("audio_cover_strength"),
            repainting_start=job.get("repainting_start"),
            repainting_end=job.get("repainting_end"),
            track_name=job.get("track_name"),
        )
        config = _api_call(
            "GenerationConfig",
            GenerationConfig,
            num_inference_steps=GENERATION_STEPS[dit_profile],
            use_cfg=GENERATION_USE_CFG[dit_profile],
            batch_size=job.get("batch_size", 1),
        )
        result = _api_call(
            "generate_music",
            generate_music,
            handler=handler,
            lm_handler=lm,
            params=params,
            config=config,
            save_dir=str(take_dir),
        )

    success = getattr(result, "success", True)
    audios = getattr(result, "audios", None) or []
    if not success or not audios:
        detail = getattr(result, "error", None) or getattr(result, "message", None)
        raise RuntimeError(f"ACE-Step generation failed: {detail or 'no audio produced'}")

    audio = audios[0]
    src_path_raw = _result_field(audio, "path", None) or _result_field(audio, "audio_path", None)
    if not src_path_raw:
        raise RuntimeError("ACE-Step reported success but returned no audio path")
    src_path = Path(src_path_raw)
    if not src_path.exists():
        raise RuntimeError(f"ACE-Step reported audio at '{src_path}' but the file does not exist")

    # storage.take_audio_path only recognizes mix.wav / mix.mp3; ACE-Step
    # writes its own output filename, so place it at the canonical path.
    dest_path = take_dir / ("mix.mp3" if src_path.suffix.lower() == ".mp3" else "mix.wav")
    if src_path != dest_path:
        shutil.move(str(src_path), str(dest_path))

    extra = getattr(result, "extra_outputs", None) or {}

    def _extra(name: str, fallback: Any) -> Any:
        if isinstance(extra, dict):
            return extra.get(name, fallback)
        return getattr(extra, name, fallback)

    caption = _extra("caption", effective_plan.get("caption", ""))
    lyrics = _extra("lyrics", effective_plan.get("lyrics", ""))
    bpm = _extra("bpm", effective_plan.get("bpm"))
    keyscale = _extra("keyscale", effective_plan.get("keyscale"))
    duration_sec = _extra("duration", effective_plan.get("duration_sec"))
    seed = _result_field(audio, "seed", None)
    if seed is None:
        seed = _extra("seed", job.get("seed", -1))

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
        "seed": seed,
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
