"""`io.modelcontextprotocol/tasks` for generation tools (SPEC.md sec 15.6).

`create_song` and `transform_take` return a task whose `taskId` is the Lyre
job id when the client advertised this extension. Otherwise they return
JSON-RPC `-32021` and do not call `/v1`, unless the caller passed
`wait_timeout_sec`, in which case the tool result is the job itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn

from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import require_client_extension
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp_types import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    RequestParams,
)
from pydantic import Field

from server.mcp import service
from server.mcp.results import create_task_result, tasks_get_result

TASKS = "io.modelcontextprotocol/tasks"
_GENERATION = frozenset({"create_song", "transform_take", "generate_take"})


class TaskParams(RequestParams):
    task_id: str = Field(alias="taskId")


class TasksExtension(Extension):
    identifier = TASKS

    def methods(self) -> Sequence[MethodBinding]:
        versions = frozenset({"2026-07-28"})
        return (
            MethodBinding("tasks/get", TaskParams, _get, versions),
            MethodBinding("tasks/update", TaskParams, _update, versions),
            MethodBinding("tasks/cancel", TaskParams, _cancel, versions),
        )

    async def intercept_tool_call(
        self, params: CallToolRequestParams, ctx: Any, call_next: Any
    ) -> Any:
        if params.name not in _GENERATION:
            return await call_next(ctx)
        arguments = params.arguments or {}
        if arguments.get("wait_timeout_sec") is not None:
            # The caller asked to block. Return the tool result, not a task,
            # whether or not the client advertised this extension.
            return await call_next(ctx)
        require_client_extension(ctx, TASKS)
        result = await call_next(ctx)
        if isinstance(result, CallToolResult) and result.is_error:
            return result
        job = result.structured_content if isinstance(result, CallToolResult) else None
        if not isinstance(job, dict) or "id" not in job:
            return result
        return create_task_result(job)


async def _get(ctx: Any, params: TaskParams) -> dict[str, Any]:
    require_client_extension(ctx, TASKS)
    job = await _load(params.task_id)
    return tasks_get_result(job)


async def _update(ctx: Any, params: TaskParams) -> dict[str, Any]:
    """Nothing to apply: no tool enters `input_required`. A known job acks."""
    require_client_extension(ctx, TASKS)
    await _load(params.task_id)
    return {"resultType": "complete"}


async def _cancel(ctx: Any, params: TaskParams) -> dict[str, Any]:
    require_client_extension(ctx, TASKS)
    try:
        await service.cancel_job(params.task_id)
    except ToolError as exc:
        _reraise(exc)
    # Cooperative: the job may still be `working` until the worker notices.
    return {"resultType": "complete"}


async def _load(task_id: str) -> dict[str, Any]:
    try:
        job = await service.get_job(task_id)
    except ToolError as exc:
        _reraise(exc)
    if not isinstance(job, dict):
        raise MCPError(INVALID_PARAMS, f"Failed to retrieve task: {task_id}")
    return job


def _reraise(exc: ToolError) -> NoReturn:
    text = str(exc)
    if "not found" in text.lower():
        raise MCPError(INVALID_PARAMS, f"Failed to retrieve task: {text}") from exc
    raise MCPError(INTERNAL_ERROR, text) from exc
