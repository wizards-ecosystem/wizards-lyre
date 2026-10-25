"""What this install can do right now (SPEC.md sec 16.6).

One report for the studio (`GET /api/capabilities`), `/v1`, and MCP
(`get_capabilities`). It keeps apart the things that are easy to blur:
weights present on disk, a profile that has actually rendered, a worker
that is running, and a worker that is busy or loading. The server never
imports CUDA: worker state comes from what the worker published, and file
state from the filesystem.
"""

from __future__ import annotations

import os
from contextlib import closing
from typing import Any

from server import jobs
from server.generation import brief as creation
from server.generation import presets
from worker.acestep_worker import settings as worker_settings

# Below the queue's floor a new render is refused before any GPU work
# (server.jobs.queue.MIN_FREE_DISK_BYTES); below this the studio warns.
LOW_FREE_DISK_BYTES = 4 * 1024 * 1024 * 1024
GENERATION_PROFILES = ("iterate", "polish", "quality")


def free_disk_bytes() -> int | None:
    return jobs.free_disk_bytes()


def _rendered_profiles() -> set[str]:
    """Profiles at least one job has completed on: the only evidence Lyre
    has that a profile's weights load and render."""
    with closing(jobs._connect()) as conn:
        rows = conn.execute(
            "SELECT DISTINCT dit_profile FROM jobs WHERE status = 'done' AND action != 'train_lora'"
        ).fetchall()
    return {row["dit_profile"] for row in rows}


def worker_state() -> dict[str, Any]:
    """`stopped`, `unavailable`, `loading`, `busy`, or `ready`, with a reason
    a person can act on."""
    backend = os.environ.get("LYRE_WORKER", "acestep")
    status = jobs.get_worker_status()
    counts = jobs.queue_counts()
    if status is None:
        return {
            "state": "stopped",
            "reason": "No worker has reported in. Start it with ./scripts/lyre worker; queued "
            "jobs wait until it runs.",
            "backend": backend,
            "dit_loaded": None,
            "queue": counts,
        }
    message = status.get("message") or ""
    if not status["ready"]:
        stale = "heartbeat stale" in message
        return {
            "state": "stopped" if stale else "unavailable",
            "reason": message or "the worker reported that it cannot generate",
            "backend": backend,
            "dit_loaded": None,
            "queue": counts,
        }
    running = jobs.list_recent_jobs(active=True)
    loading = any(
        job["status"] == "running" and job.get("stage") == "loading_model" for job in running
    )
    state = "loading" if loading else "busy" if counts["running"] else "ready"
    return {
        "state": state,
        "reason": message,
        "backend": backend,
        "dit_loaded": status.get("loaded_dit_profile"),
        "queue": counts,
    }


def profile_report(dit_profile: str, worker: dict[str, Any], rendered: set[str]) -> dict[str, Any]:
    info = worker_settings.PROFILE_INFO[dit_profile]
    remote = worker["backend"] == "remote"
    files: dict[str, Any] = (
        {"state": "remote", "parts": []}
        if remote or worker["backend"] == "mock"
        else worker_settings.profile_files(dit_profile)
    )
    capability = jobs.get_worker_capability(dit_profile)
    supported: bool | None = None if capability is None else capability[0]
    reason = None if capability is None else capability[1]
    if files["state"] in ("missing", "incomplete"):
        supported = False
        missing = [part["name"] for part in files["parts"] if part["state"] != "present"]
        reason = (
            f"model files {files['state']}: {', '.join(missing)}. Download them with "
            "./scripts/lyre setup (see docs/INSTALLATION.md)."
        )
    return {
        "id": dit_profile,
        "label": info["label"],
        "summary": info["summary"],
        "checkpoint": worker_settings.DIT_CHECKPOINTS[dit_profile],
        "inference_steps": worker_settings.GENERATION_STEPS[dit_profile],
        "guidance_scale": worker_settings.GENERATION_GUIDANCE_SCALE[dit_profile],
        "cpu_offload": dit_profile in worker_settings.CPU_OFFLOAD_PROFILES,
        "files": files["state"],
        # Only a completed render proves the files load; presence does not.
        "validated": dit_profile in rendered,
        "loaded": worker["dit_loaded"] == dit_profile,
        "supported": supported,
        "reason": reason,
    }


def capabilities() -> dict[str, Any]:
    worker = worker_state()
    rendered = _rendered_profiles()
    free = free_disk_bytes()
    return {
        "schema": "lyre.capabilities/1",
        "worker": worker,
        "profiles": [profile_report(profile, worker, rendered) for profile in GENERATION_PROFILES],
        "lm": {
            "model": worker_settings.DEFAULT_LM,
            # project.json's lm_model is descriptive; the worker always loads
            # this one (SPEC.md sec 16.2), so it is not offered as a choice.
            "selectable": False,
        },
        "disk": {
            "free_bytes": free,
            "low": free is not None and free < LOW_FREE_DISK_BYTES,
            "blocked": free is not None and free < jobs.MIN_FREE_DISK_BYTES,
        },
        "presets": presets.list_presets(),
        "limits": {
            "bpm": [creation.BPM_MIN, creation.BPM_MAX],
            "duration_sec": [creation.DURATION_MIN, creation.DURATION_MAX],
            "timesignatures": list(creation.TIME_SIGNATURES),
            "languages": list(creation.VALID_LANGUAGES),
            "caption_soft_limit": creation.CAPTION_SOFT_LIMIT,
            "lyrics_soft_limit": creation.LYRICS_SOFT_LIMIT,
        },
    }
