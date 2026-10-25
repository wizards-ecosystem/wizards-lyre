"""Build the public `/v1` job object from a queue row.

`/api` and `/v1` read the same SQLite row. This module adds the fields the
agent contract promises (seed, artifact paths) without forking the queue.
"""

from __future__ import annotations

from typing import Any

from server import storage
from server.storage.paths import take_dir


def artifact_urls(project_id: str, take_id: str) -> list[str]:
    """Stable relative paths for the files that exist on this take.
    Clients resolve them against the origin they already use."""
    directory = take_dir(project_id, take_id)
    base = f"/v1/artifacts/{project_id}/{take_id}"
    urls: list[str] = []
    for name in ("mix.wav", "mix.mp3", "lyrics.lrc"):
        if (directory / name).is_file():
            urls.append(f"{base}/{name}")
    return urls


def _seed(job: dict[str, Any], take: dict[str, Any] | None) -> int | None:
    if take is not None:
        recorded = take.get("seed")
        if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded != -1:
            return recorded
    payload = job.get("payload") or {}
    requested = payload.get("seed")
    if isinstance(requested, int) and not isinstance(requested, bool) and requested != -1:
        return requested
    return None


def job_object(job: dict[str, Any]) -> dict[str, Any]:
    """SPEC.md sec 15.3 job object. `job` is a `read_job` row (includes
    `payload`). On `done`, `artifacts` lists the stable paths and `take_id`
    is set. On `error`, `error` is set. On `cancelled`, both `error` and
    `take_id` are null."""
    status = job["status"]
    take_id = job.get("take_id")
    take: dict[str, Any] | None = None
    artifacts: list[str] = []
    if status == "done" and take_id:
        try:
            take = storage.get_take(job["project_id"], take_id)
        except storage.TakeNotFound:
            take = None
        artifacts = artifact_urls(job["project_id"], take_id)
    if status == "cancelled":
        take_id = None
    return {
        "id": job["id"],
        "project_id": job["project_id"],
        "action": job["action"],
        "dit_profile": job["dit_profile"],
        "status": status,
        "stage": job.get("stage"),
        "take_id": take_id,
        "error": None if status == "cancelled" else job.get("error"),
        "seed": _seed(job, take),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "artifacts": artifacts,
        "submission_id": job.get("submission_id"),
        "retry_of": job.get("retry_of"),
    }
