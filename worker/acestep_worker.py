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
not the HTTP process (sec 10 point 5).
"""

from __future__ import annotations

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


def _ensure_loaded(dit_profile: str) -> tuple[Any, Any, Any]:
    """Swap the loaded DiT if needed. Caller holds _LOCK."""
    AceStepHandler, _, LLMHandler, _ = _import_acestep()

    if _STATE["dit_profile"] != dit_profile or _STATE["handler"] is None:
        # Unload the previous DiT before loading the next; never hold two
        # DiTs at once on a 16 GB card (SPEC.md sec 4.3).
        _STATE["handler"] = None
        checkpoint = DIT_CHECKPOINTS[dit_profile]
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


def run_job(job: dict[str, Any], plan: dict[str, Any], take_id: str, take_dir: Path) -> dict:
    """Run one real ACE-Step job and return the take's meta.json contents.

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable; `server.jobs`
    catches that and marks the job `error` without crashing the HTTP process.
    """
    _, GenerationParams, _, generate_music = _import_acestep()
    take_dir.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        handler, lm, dit_profile = _ensure_loaded(job["dit_profile"])
        params = GenerationParams(
            task_type=TASK_TYPE_BY_ACTION[job["action"]],
            caption=plan.get("caption", ""),
            lyrics=plan.get("lyrics", ""),
            bpm=plan.get("bpm"),
            keyscale=plan.get("keyscale"),
            duration_sec=plan.get("duration_sec"),
            instrumental=plan.get("instrumental", False),
            seed=job.get("seed", -1),
            src_audio=job.get("upload_path") or job.get("source_take_id"),
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

    return {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[job["action"]],
        "dit_profile": dit_profile,
        "seed": getattr(result, "seed", job.get("seed", -1)),
        "duration_sec": getattr(result, "duration_sec", plan.get("duration_sec")),
        "caption": plan.get("caption", ""),
        "lyrics": plan.get("lyrics", ""),
        "bpm": plan.get("bpm"),
        "keyscale": plan.get("keyscale"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": None,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
    }
