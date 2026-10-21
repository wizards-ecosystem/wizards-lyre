"""Stateless Streamable HTTP `POST /mcp` on the Lyre server (SPEC.md sec 15.6).

A fresh session manager is built for each process lifespan. The SDK allows
`run()` once per manager, and tests open many app lifespans. DNS-rebinding
checks stay off so this gateway can apply the spec's Origin rule itself:
a present Origin that is not this server's is 403, and a missing Origin is
allowed. `tasks/get`, `tasks/update`, and `tasks/cancel` must send `Mcp-Name`
equal to `params.taskId` or the response is HTTP 400 and JSON-RPC `-32020`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.inbound import HEADER_MISMATCH, decode_header_value
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server import config
from server.api_keys import ApiKeyStoreError, authenticate
from server.mcp.client import bind_call, unbind_call
from server.mcp.server import mcp_server

logger = logging.getLogger(__name__)

_TASK_METHODS = frozenset({"tasks/get", "tasks/update", "tasks/cancel"})
_inner: ASGIApp | None = None


@asynccontextmanager
async def mcp_lifespan() -> AsyncIterator[None]:
    """Enter a new stateless Streamable HTTP session for this process."""
    global _inner
    starlette_app = mcp_server.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host="127.0.0.1",
    )
    async with mcp_server.session_manager.run():
        _inner = starlette_app
        try:
            yield
        finally:
            _inner = None


class McpGateway:
    """ASGI app mounted at `/mcp`. Rewrites the path to the SDK's `/` route."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": "not found", "param": None}},
            )(scope, receive, send)
            return
        inner = _inner
        if inner is None:
            await JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "unavailable",
                        "message": "MCP transport is not running",
                        "param": None,
                    }
                },
            )(scope, receive, send)
            return
        if _origin_rejected(scope):
            await JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "forbidden",
                        "message": "origin not allowed",
                        "param": "Origin",
                    }
                },
            )(scope, receive, send)
            return
        api_key, denied = _authenticate(scope)
        if denied is not None:
            await denied(scope, receive, send)
            return
        messages = await _collect(receive)
        mismatch = _task_name_rejection(scope, _body(messages))
        if mismatch is not None:
            await mismatch(scope, receive, send)
            return
        child = dict(scope)
        child["path"] = "/"
        child["raw_path"] = b"/"
        token = bind_call(api_key)
        try:
            await inner(child, _replay(messages), send)
        finally:
            unbind_call(token)


gateway = McpGateway()


def server_origins() -> set[str]:
    """HTTP origins that count as this process. Loopback aliases share one."""
    port = config.lyre_port()
    hosts = ("127.0.0.1", "localhost", "::1") if config.is_loopback() else (config.bind_host(),)
    origins: set[str] = set()
    for host in hosts:
        shown = f"[{host}]" if ":" in host else host
        origins.add(f"http://{shown}:{port}")
    return origins


def _origin_rejected(scope: Scope) -> bool:
    origin = Headers(scope=scope).get("origin")
    if origin is None or origin == "":
        return False
    return origin.rstrip("/") not in server_origins()


def _authenticate(scope: Scope) -> tuple[str | None, Response | None]:
    try:
        required = config.auth_required()
    except config.BindConfigError as exc:
        return None, _misconfigured(exc)
    if not required:
        return None, None
    secret = _bearer(Headers(scope=scope).get("authorization"))
    if secret is None:
        return None, _unauthorized()
    try:
        record = authenticate(secret)
    except ApiKeyStoreError as exc:
        return None, _misconfigured(exc)
    if record is None:
        return None, _unauthorized()
    return secret, None


def _bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, rest = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    secret = rest.strip()
    return secret or None


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": "unauthorized",
                "message": "missing or unknown API key",
                "param": None,
            }
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _misconfigured(exc: Exception) -> JSONResponse:
    logger.error("MCP bind configuration error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "misconfigured",
                "message": "server configuration error",
                "param": None,
            }
        },
    )


def _task_name_rejection(scope: Scope, body: bytes) -> Response | None:
    headers = Headers(scope=scope)
    content_type = headers.get("content-type") or ""
    if not content_type.lower().startswith("application/json"):
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("method") not in _TASK_METHODS:
        return None
    params = payload.get("params")
    task_id = params.get("taskId") if isinstance(params, dict) else None
    decoded = decode_header_value(headers.get("mcp-name"))
    matches = decoded == task_id if isinstance(task_id, str) else decoded is None
    if matches:
        return None
    request_id = payload.get("id")
    if not isinstance(request_id, str | int):
        request_id = None
    message = "mcp-name header does not match the request body's 'taskId' parameter"
    encoded = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": HEADER_MISMATCH, "message": message},
        },
        separators=(",", ":"),
    )
    return Response(encoded, status_code=400, media_type="application/json")


async def _collect(receive: Receive) -> list[Message]:
    messages: list[Message] = []
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request" or not message.get("more_body", False):
            break
    return messages


def _body(messages: list[Message]) -> bytes:
    return b"".join(
        message.get("body") or b"" for message in messages if message["type"] == "http.request"
    )


def _replay(messages: list[Message]) -> Receive:
    pending = list(messages)

    async def receive() -> Message:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    return receive
