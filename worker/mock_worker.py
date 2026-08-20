"""Mocked ACE-Step worker used by tests and local dev without a GPU.

Writes a tiny silent WAV plus a spec-shaped meta.json (SPEC.md sec 7.3).
Never imports torch, CUDA, or acestep -- see SPEC.md sec 10 and 11.
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


def run_job(job: dict[str, Any], plan: dict[str, Any], take_id: str, take_dir: Path) -> dict:
    """Run one mocked job and return the take's meta.json contents.

    `take_dir` must already be inside the projects/ path jail; this function
    only ever writes files inside it.
    """
    take_dir.mkdir(parents=True, exist_ok=True)

    seed = job.get("seed", -1)
    if seed is None or seed == -1:
        seed = random.randint(1, 2**31 - 1)

    audio_path = take_dir / "mix.wav"
    duration = _write_silent_wav(audio_path)

    return {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[job["action"]],
        "dit_profile": job["dit_profile"],
        "seed": seed,
        "duration_sec": duration,
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
