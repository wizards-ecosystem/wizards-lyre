"""MCP tool and resource registration, and the stdio entrypoint.

`./scripts/lyre mcp` runs `main`. `POST /mcp` uses the same `mcp_server`.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp_types import CallToolResult
from pydantic import Field

from server.mcp import service
from server.mcp.results import (
    ArtifactResult,
    JobResult,
    JsonObject,
    PageResult,
    complete_job,
)
from server.mcp.tasks import TasksExtension

_INSTRUCTIONS = (
    "Lyre generates music on the operator's machine. create_song and "
    "transform_take take minutes. If you do not speak the tasks extension "
    "io.modelcontextprotocol/tasks, pass wait_timeout_sec (for example 600) "
    "so the tool polls until the job finishes. If you do speak it, omit "
    "wait_timeout_sec and poll tasks/get with the returned taskId. Results "
    "carry the job JSON and artifact URLs, never audio bytes. Use "
    "get_artifact for mix.wav, mix.mp3, and lyrics.lrc."
)


def _server() -> MCPServer:
    """Build the server without letting the SDK reconfigure process logging.

    `MCPServer` calls `logging.basicConfig`. A temporary handler makes that a
    no-op when the process had no handlers yet, which is the case for
    `./scripts/lyre mcp` and for pytest importing the app.
    """
    root = logging.getLogger()
    guard = None
    if not root.handlers:
        guard = logging.NullHandler()
        root.addHandler(guard)
    try:
        built = MCPServer(
            name="lyre",
            title="The Wizard's Lyre",
            version="0.2.0",
            instructions=_INSTRUCTIONS,
            log_level="WARNING",
            subscriptions=False,
            extensions=[TasksExtension()],
        )
    finally:
        if guard is not None:
            root.removeHandler(guard)
    _register(built)
    return built


def _register(server: MCPServer) -> None:
    # Spec order, then get_artifact. Registration order is the tools/list order.
    server.add_tool(
        create_song,
        description=(
            "Create a project and queue a song from a prompt. Sends "
            "Idempotency-Key. Returns a task, or the finished job when "
            "wait_timeout_sec is set."
        ),
    )
    server.add_tool(
        transform_take,
        description=(
            "Queue cover, repaint, extract, lego, or complete on an existing "
            "project. Sends Idempotency-Key. Returns a task, or the finished "
            "job when wait_timeout_sec is set."
        ),
    )
    server.add_tool(get_job, description="Fetch one job by id.")
    server.add_tool(
        cancel_job,
        description=(
            "Cancel a job. Queued jobs stop; a running render stops after the current call."
        ),
    )
    server.add_tool(list_projects, description="List projects, newest first.")
    server.add_tool(get_project, description="Fetch one project, its plan, and its takes.")
    server.add_tool(update_plan, description="Replace a project's plan.json.")
    server.add_tool(list_takes, description="List takes for one project.")
    server.add_tool(
        get_artifact,
        description=(
            "URL and metadata for mix.wav, mix.mp3, and lyrics.lrc. Does not download the audio."
        ),
    )
    server.resource(
        "lyre://project/{project_id}",
        name="project",
        mime_type="application/json",
        description="One project, its plan, and its takes, as JSON.",
    )(read_project)
    server.resource(
        "lyre://take/{project_id}/{take_id}",
        name="take",
        mime_type="application/json",
        description="One take's JSON. Audio bytes are at the artifact URLs.",
    )(read_take)


async def create_song(
    prompt: Annotated[str, Field(description="What the song should be.")],
    title: str | None = None,
    caption: str | None = None,
    lyrics: str | None = None,
    instrumental: bool | None = None,
    vocal_language: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    duration_sec: float | None = None,
    dit_profile: str | None = None,
    seed: int | None = None,
    caption_rewrite: bool | None = None,
    webhook_url: str | None = None,
    idempotency_key: Annotated[
        str | None,
        Field(description="Optional Idempotency-Key. A UUID is generated when omitted."),
    ] = None,
    wait_timeout_sec: Annotated[
        float | None,
        Field(
            description=(
                "Seconds to poll GET /v1/jobs/{id}. Set this when the client "
                "cannot handle tasks. Omit it to receive a task handle."
            )
        ),
    ] = None,
) -> Annotated[CallToolResult, JobResult]:
    job, timed_out = await service.create_song(
        {
            "prompt": prompt,
            "title": title,
            "caption": caption,
            "lyrics": lyrics,
            "instrumental": instrumental,
            "vocal_language": vocal_language,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": timesignature,
            "duration_sec": duration_sec,
            "dit_profile": dit_profile,
            "seed": seed,
            "caption_rewrite": caption_rewrite,
            "webhook_url": webhook_url,
        },
        idempotency_key=idempotency_key,
        wait_timeout_sec=wait_timeout_sec,
    )
    return complete_job(job, timed_out=timed_out)


async def transform_take(
    project_id: Annotated[str, Field(description="Project that owns the source take.")],
    action: Annotated[
        str,
        Field(description="One of cover, repaint, extract, lego, complete."),
    ],
    source_take_id: str | None = None,
    upload_path: str | None = None,
    repainting_start: float | None = None,
    repainting_end: float | None = None,
    track_name: str | None = None,
    audio_cover_strength: float | None = None,
    dit_profile: str | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    idempotency_key: str | None = None,
    wait_timeout_sec: float | None = None,
) -> Annotated[CallToolResult, JobResult]:
    job, timed_out = await service.transform_take(
        {
            "project_id": project_id,
            "action": action,
            "source_take_id": source_take_id,
            "upload_path": upload_path,
            "repainting_start": repainting_start,
            "repainting_end": repainting_end,
            "track_name": track_name,
            "audio_cover_strength": audio_cover_strength,
            "dit_profile": dit_profile,
            "seed": seed,
            "batch_size": batch_size,
        },
        idempotency_key=idempotency_key,
        wait_timeout_sec=wait_timeout_sec,
    )
    return complete_job(job, timed_out=timed_out)


async def get_job(job_id: str) -> Annotated[CallToolResult, JobResult]:
    return complete_job(await service.get_job(job_id), timed_out=False)


async def cancel_job(job_id: str) -> Annotated[CallToolResult, JobResult]:
    return complete_job(await service.cancel_job(job_id), timed_out=False)


async def list_projects(limit: int | None = None, cursor: str | None = None) -> PageResult:
    payload = await service.list_projects(limit=limit, cursor=cursor)
    return PageResult.model_validate(payload)


async def get_project(project_id: str) -> JsonObject:
    return JsonObject.model_validate(await service.get_project(project_id))


async def update_plan(
    project_id: str,
    plan: Annotated[
        dict[str, Any], Field(description="Plan fields to store. Unknown keys are dropped by /v1.")
    ],
) -> JsonObject:
    return JsonObject.model_validate(await service.update_plan(project_id, plan))


async def list_takes(
    project_id: str, limit: int | None = None, cursor: str | None = None
) -> PageResult:
    payload = await service.list_takes(project_id, limit=limit, cursor=cursor)
    return PageResult.model_validate(payload)


async def get_artifact(project_id: str, take_id: str) -> ArtifactResult:
    payload = await service.get_artifact(project_id, take_id)
    return ArtifactResult.model_validate(payload)


async def read_project(project_id: str) -> str:
    try:
        payload = await service.read_project_resource(project_id)
    except ToolError as exc:
        raise ResourceNotFoundError(str(exc)) from exc
    return json.dumps(payload, ensure_ascii=False)


async def read_take(project_id: str, take_id: str) -> str:
    try:
        payload = await service.read_take_resource(project_id, take_id)
    except ToolError as exc:
        raise ResourceNotFoundError(str(exc)) from exc
    return json.dumps(payload, ensure_ascii=False)


mcp_server = _server()


def main() -> None:
    """Speak newline-delimited JSON-RPC on stdin and stdout."""
    mcp_server.run(transport="stdio")
