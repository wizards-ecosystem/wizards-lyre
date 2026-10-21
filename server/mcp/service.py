"""Tool operations over `LyreClient`. No protocol types here.

`create_song` and `transform_take` always send `Idempotency-Key`. An optional
wait polls `GET /v1/jobs/{id}` until the job is terminal or the timeout
elapses. The tasks extension decides whether that result is returned as a
task handle; this module only talks to `/v1`.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from mcp.server.mcpserver.exceptions import ToolError

from server.mcp.client import LyreApiError, LyreClient, open_client, public_base_url

# Visible ASCII, no spaces. The same rule as `Idempotency-Key` on `/v1`.
_KEY_RE = re.compile(r"^[!-~]{1,255}$")
_TERMINAL = frozenset({"done", "error", "cancelled"})
_TRANSFORM = frozenset({"cover", "repaint", "extract", "lego", "complete"})
_ARTIFACTS = (
    ("mix.wav", "audio/wav"),
    ("mix.mp3", "audio/mpeg"),
    ("lyrics.lrc", "text/plain"),
)

# Tests shrink this so a wait loop does not sleep.
POLL_INTERVAL_SEC = 0.5
_MAX_WAIT_SEC = 24 * 60 * 60


def resolve_idempotency_key(supplied: str | None) -> str:
    if supplied is None or supplied == "":
        return str(uuid.uuid4())
    if _KEY_RE.fullmatch(supplied) is None:
        raise ToolError("idempotency_key must be 1-255 visible ASCII characters with no spaces")
    return supplied


def song_body(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The `POST /v1/songs` object, without MCP-only fields."""
    body: dict[str, Any] = {}
    for name in (
        "prompt",
        "title",
        "caption",
        "lyrics",
        "instrumental",
        "vocal_language",
        "bpm",
        "keyscale",
        "timesignature",
        "duration_sec",
        "dit_profile",
        "seed",
        "caption_rewrite",
        "webhook_url",
    ):
        if name in arguments and arguments[name] is not None:
            body[name] = arguments[name]
    return body


def transform_body(arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    action = arguments.get("action")
    if action not in _TRANSFORM:
        raise ToolError("action must be cover, repaint, extract, lego, or complete")
    project_id = arguments.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ToolError("project_id is required")
    body: dict[str, Any] = {"action": action}
    for name in (
        "dit_profile",
        "source_take_id",
        "upload_path",
        "repainting_start",
        "repainting_end",
        "track_name",
        "audio_cover_strength",
        "seed",
        "batch_size",
    ):
        if name in arguments and arguments[name] is not None:
            body[name] = arguments[name]
    return project_id, body


def _check_wait(timeout_sec: float | None) -> None:
    if timeout_sec is None:
        return
    if not math.isfinite(timeout_sec) or timeout_sec < 0:
        raise ToolError("wait_timeout_sec must be a non-negative finite number")
    if timeout_sec > _MAX_WAIT_SEC:
        raise ToolError("wait_timeout_sec cannot exceed 86400")


async def create_song(
    arguments: Mapping[str, Any],
    *,
    idempotency_key: str | None,
    wait_timeout_sec: float | None,
) -> tuple[dict[str, Any], bool]:
    _check_wait(wait_timeout_sec)
    key = resolve_idempotency_key(idempotency_key)
    async with _session() as client:
        job = await _call(client.create_song(song_body(arguments), idempotency_key=key))
        if wait_timeout_sec is None:
            return job, False
        return await _wait(client, job, wait_timeout_sec)


async def transform_take(
    arguments: Mapping[str, Any],
    *,
    idempotency_key: str | None,
    wait_timeout_sec: float | None,
) -> tuple[dict[str, Any], bool]:
    project_id, body = transform_body(arguments)
    _check_wait(wait_timeout_sec)
    key = resolve_idempotency_key(idempotency_key)
    async with _session() as client:
        job = await _call(client.create_job(project_id, body, idempotency_key=key))
        if wait_timeout_sec is None:
            return job, False
        return await _wait(client, job, wait_timeout_sec)


async def get_job(job_id: str) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.get_job(job_id))


async def cancel_job(job_id: str) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.cancel_job(job_id))


async def list_projects(*, limit: int | None, cursor: str | None) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.list_projects(limit=limit, cursor=cursor))


async def get_project(project_id: str) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.get_project(project_id))


async def update_plan(project_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.update_plan(project_id, plan))


async def list_takes(project_id: str, *, limit: int | None, cursor: str | None) -> dict[str, Any]:
    async with _session() as client:
        return await _call(client.list_takes(project_id, limit=limit, cursor=cursor))


async def get_artifact(project_id: str, take_id: str) -> dict[str, Any]:
    """URL and metadata for the three stable artifact files. No audio bytes."""
    try:
        base = public_base_url()
    except LyreApiError as exc:
        raise ToolError(_detail(exc)) from exc
    files: list[dict[str, Any]] = []
    async with _session() as client:
        for name, media_type in _ARTIFACTS:
            status, length = await _call(client.artifact_head(project_id, take_id, name))
            project = quote(project_id, safe="")
            take = quote(take_id, safe="")
            filename = quote(name, safe="")
            url = f"/v1/artifacts/{project}/{take}/{filename}"
            files.append(
                {
                    "name": name,
                    "url": url,
                    "absolute_url": f"{base}{url}",
                    "media_type": media_type,
                    "available": status == 200,
                    "content_length": length if status == 200 else None,
                    "http_status": status,
                }
            )
    return {"project_id": project_id, "take_id": take_id, "files": files}


async def read_project_resource(project_id: str) -> dict[str, Any]:
    return await get_project(project_id)


async def read_take_resource(project_id: str, take_id: str) -> dict[str, Any]:
    project = await get_project(project_id)
    takes = project.get("takes")
    if isinstance(takes, list):
        for take in takes:
            if isinstance(take, dict) and take.get("id") == take_id:
                return take
    raise ToolError(f"take not found: {take_id}")


async def _wait(
    client: LyreClient, job: dict[str, Any], timeout_sec: float
) -> tuple[dict[str, Any], bool]:
    _check_wait(timeout_sec)
    deadline = time.monotonic() + timeout_sec
    current = job
    while current.get("status") not in _TERMINAL:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return current, True
        await asyncio.sleep(min(POLL_INTERVAL_SEC, remaining))
        current = await _call(client.get_job(str(current["id"])))
    return current, False


@contextlib.asynccontextmanager
async def _session() -> Any:
    """Open the `/v1` client, reporting configuration failures as tool errors."""
    try:
        client = open_client()
    except LyreApiError as exc:
        raise ToolError(_detail(exc)) from exc
    try:
        yield client
    finally:
        await client.aclose()


def _detail(exc: LyreApiError) -> str:
    if exc.param:
        return f"{exc.message} ({exc.param})"
    return exc.message


async def _call(awaitable: Any) -> Any:
    try:
        return await awaitable
    except LyreApiError as exc:
        raise ToolError(_detail(exc)) from exc
