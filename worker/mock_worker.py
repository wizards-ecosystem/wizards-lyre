"""Mocked ACE-Step worker used by tests and local dev without a GPU.

Writes a tiny silent WAV plus a spec-shaped meta.json (SPEC.md sec 7.3).
Never imports torch, CUDA, or acestep -- see SPEC.md sec 10 and 11.

Also stands in for the 5Hz LM's simple-mode planning (SPEC.md sec 7.2):
when a `generate` job's plan has a `query` but no caption/lyrics, this
worker deterministically derives them (no real LM) and returns a plan
patch for the caller to persist, exercising the same "query in, filled
plan out" contract that `worker.acestep_worker` implements for real.
"""

from __future__ import annotations

import random
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def initialize_worker() -> tuple[bool, str]:
    """Worker-startup readiness (SPEC.md sec 10 point 1). The mock has no
    CUDA/model to detect or load, so it is always immediately ready --
    real startup detection/preload is exercised against
    `worker.acestep_worker.initialize_worker`."""
    return True, "mock worker: no GPU/model to load, always ready"


DEFAULT_SIMULATED_DIT_PROFILE = "iterate"


def get_loaded_dit_profile() -> str | None:
    """The mock never actually loads a model; report the default profile as
    "loaded" since `initialize_worker` always simulates success for it, so
    `/api/health` has something meaningful to show in mock-backed local
    dev/tests (SPEC.md sec 8 `dit_loaded`)."""
    return DEFAULT_SIMULATED_DIT_PROFILE


def _is_simple_mode(plan: dict[str, Any]) -> bool:
    return bool(plan.get("query")) and not plan.get("caption") and not plan.get("lyrics")


def _plan_from_query(plan: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for the 5Hz LM filling caption/lyrics/metas
    from a simple-mode query (SPEC.md sec 7.2: "Worker sets thinking=true.
    LM fills caption, lyrics, metas. Persist the filled plan.").

    Returns only the fields the LM filled in -- a patch, not a full plan --
    so the caller (server.jobs) can merge it onto whatever plan is current
    on disk when the job finishes, instead of overwriting concurrent edits
    to unrelated fields with this stale snapshot."""
    query = plan["query"]
    instrumental = bool(plan.get("instrumental"))
    return {
        "caption": f"auto: {query}",
        "lyrics": "[Instrumental]" if instrumental else f"[Verse]\n{query}",
        "bpm": plan.get("bpm") or 120,
        "keyscale": plan.get("keyscale") or "C Major",
    }


def run_job(
    job: dict[str, Any], plan: dict[str, Any], take_id: str, take_dir: Path
) -> tuple[dict, dict | None]:
    """Run one mocked job. Returns `(take_meta, plan_patch)`; `plan_patch` is
    a delta of the fields to persist when this job filled the plan in
    (simple mode), else None. `server.jobs` merges it onto the plan that's
    current on disk when the job finishes, not the stale snapshot passed
    in as `plan`.

    `take_dir` must already be inside the projects/ path jail; this function
    only ever writes files inside it.
    """
    take_dir.mkdir(parents=True, exist_ok=True)

    plan_patch: dict[str, Any] | None = None
    effective_plan = plan
    if job.get("action") == "generate" and _is_simple_mode(plan):
        plan_patch = _plan_from_query(plan)
        effective_plan = {**plan, **plan_patch}

    seed = job.get("seed", -1)
    if seed is None or seed == -1:
        seed = random.randint(1, 2**31 - 1)

    audio_path = take_dir / "mix.wav"
    duration = _write_silent_wav(audio_path)

    meta = {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[job["action"]],
        "dit_profile": job["dit_profile"],
        "seed": seed,
        "duration_sec": duration,
        "caption": effective_plan.get("caption", ""),
        "lyrics": effective_plan.get("lyrics", ""),
        "bpm": effective_plan.get("bpm"),
        "keyscale": effective_plan.get("keyscale"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": None,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
    }
    return meta, plan_patch
