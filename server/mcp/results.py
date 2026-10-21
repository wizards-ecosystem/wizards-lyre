"""Tool and task result shapes. No HTTP here.

A finished generation is the job object as `structuredContent`, the same JSON
as text, and a `resource_link` to the take when one exists. It is never an
audio content block. Task handles use the flat 2026-07-28 `Task` fields.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from mcp_types import CallToolResult, ContentBlock, ResourceLink, TextContent
from pydantic import BaseModel, ConfigDict, Field

POLL_INTERVAL_MS = 2000
_FAILED_CODE = -32000


class JobResult(BaseModel):
    """The `/v1` job object. Extra keys from the server are kept."""

    model_config = ConfigDict(extra="allow")

    id: str
    project_id: str
    action: str
    dit_profile: str
    status: str
    stage: str | None = None
    take_id: str | None = None
    error: str | None = None
    seed: int | None = None
    created_at: str
    updated_at: str
    artifacts: list[str] = Field(default_factory=list)


class PageResult(BaseModel):
    """A `/v1` list page."""

    model_config = ConfigDict(extra="allow")

    data: list[Any] = Field(default_factory=list)
    next_cursor: str | None = None


class JsonObject(BaseModel):
    """A JSON object whose fields are whatever `/v1` returned."""

    model_config = ConfigDict(extra="allow")


class ArtifactFile(BaseModel):
    name: str
    url: str
    absolute_url: str
    media_type: str
    available: bool
    content_length: int | None = None
    http_status: int


class ArtifactResult(BaseModel):
    project_id: str
    take_id: str
    files: list[ArtifactFile]


def take_uri(project_id: str, take_id: str) -> str:
    project = quote(project_id, safe="")
    take = quote(take_id, safe="")
    return f"lyre://take/{project}/{take}"


def complete_job(job: dict[str, Any], *, timed_out: bool) -> CallToolResult:
    """A synchronous tool result for one job. `timed_out` adds a second text block."""
    text = json.dumps(job, ensure_ascii=False)
    content: list[ContentBlock] = [TextContent(type="text", text=text)]
    if timed_out:
        status = job.get("status") or "unfinished"
        content.append(
            TextContent(type="text", text=f"wait_timeout_sec elapsed; the job is still {status}")
        )
    take_id = job.get("take_id")
    project_id = job.get("project_id")
    if job.get("status") == "done" and isinstance(take_id, str) and isinstance(project_id, str):
        content.append(
            ResourceLink(
                type="resource_link",
                uri=take_uri(project_id, take_id),
                name=f"take-{take_id}",
                mime_type="application/json",
            )
        )
    return CallToolResult(
        content=content,
        structured_content=job,
        is_error=job.get("status") == "error",
    )


def tool_result_payload(job: dict[str, Any]) -> dict[str, Any]:
    """The `tools/call` result embedded in a completed task. Not an error result."""
    result = complete_job(job, timed_out=False)
    return result.model_dump(mode="json", by_alias=True, exclude_none=True)


def create_task_result(job: dict[str, Any]) -> dict[str, Any]:
    """`CreateTaskResult` for a job that was just inserted. Status stays `working`."""
    stage = job.get("stage")
    message = stage if isinstance(stage, str) and stage else "queued"
    return {
        "resultType": "task",
        "taskId": job["id"],
        "status": "working",
        "statusMessage": message,
        "createdAt": job.get("created_at"),
        "lastUpdatedAt": job.get("updated_at"),
        "ttlMs": None,
        "pollIntervalMs": POLL_INTERVAL_MS,
    }


def tasks_get_result(job: dict[str, Any]) -> dict[str, Any]:
    """`tasks/get` body. A job `error` is task status `failed` (SPEC.md sec 15.6)."""
    status = job.get("status")
    mapped = {
        "queued": "working",
        "running": "working",
        "done": "completed",
        "error": "failed",
        "cancelled": "cancelled",
    }.get(status if isinstance(status, str) else "", "working")
    stage = job.get("stage")
    if isinstance(stage, str) and stage:
        message: str = stage
    elif status == "queued":
        message = "queued"
    else:
        message = status if isinstance(status, str) and status else "working"
    body: dict[str, Any] = {
        "resultType": "complete",
        "taskId": job.get("id"),
        "status": mapped,
        "statusMessage": message,
        "createdAt": job.get("created_at"),
        "lastUpdatedAt": job.get("updated_at"),
        "ttlMs": None,
        "pollIntervalMs": POLL_INTERVAL_MS,
    }
    if mapped == "completed":
        body["result"] = tool_result_payload(job)
    elif mapped == "failed":
        error = job.get("error")
        text = error if isinstance(error, str) and error else "job failed"
        body["statusMessage"] = text
        body["error"] = {"code": _FAILED_CODE, "message": text}
    return body
