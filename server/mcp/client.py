"""HTTP client for Lyre's `/v1` API.

stdio reads `LYRE_URL` and an optional `LYRE_API_KEY`. `POST /mcp` uses the
same client in-process, forwarding the caller's bearer token. Tests patch
`TRANSPORT` so nothing here opens a socket.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx2

from server import config

# Patched by tests. When set, both stdio and `POST /mcp` use it instead of
# the network or the in-process app.
TRANSPORT: httpx2.AsyncBaseTransport | None = None

_CALL: ContextVar[Call | None] = ContextVar("lyre_mcp_call", default=None)


@dataclass(frozen=True)
class Call:
    """Who this MCP request should act as when it calls `/v1`."""

    api_key: str | None
    in_process: bool


class LyreApiError(Exception):
    """A `/v1` response the tool should show to the model, or a client failure."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param


def bind_call(api_key: str | None) -> Token[Call | None]:
    """Mark the current task as an in-process `POST /mcp` call."""
    return _CALL.set(Call(api_key=api_key, in_process=True))


def unbind_call(token: Token[Call | None]) -> None:
    _CALL.reset(token)


def public_base_url() -> str:
    """Origin artifact URLs are resolved against. Never the in-process placeholder."""
    raw = os.environ.get("LYRE_URL")
    if raw is not None and raw.strip():
        return _check_base(raw.strip())
    return _default_base()


def open_client() -> LyreClient:
    """The client for this task: the mocked transport, the in-process app, or `LYRE_URL`."""
    call = _CALL.get()
    transport = TRANSPORT
    if call is not None and call.in_process:
        if transport is None:
            # Imported here so constructing the MCP server does not import the
            # FastAPI app while the app is still importing this package.
            from server.app import app

            transport = httpx2.ASGITransport(app=app)
        return LyreClient("http://lyre.internal", api_key=call.api_key, transport=transport)
    api_key = os.environ.get("LYRE_API_KEY")
    if api_key is not None:
        api_key = api_key.strip() or None
    return LyreClient(public_base_url(), api_key=api_key, transport=transport)


def _default_base() -> str:
    host = config.bind_host()
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.lyre_port()}"


def _check_base(url: str) -> str:
    cleaned = url.rstrip("/")
    parts = urlsplit(cleaned)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise LyreApiError(
            0,
            "misconfigured",
            "LYRE_URL must be an http or https origin without userinfo or a path",
            "LYRE_URL",
        )
    return cleaned


def _segment(value: str) -> str:
    return quote(value, safe="")


class LyreClient:
    """Async `/v1` calls. One client per tool invocation; close it when done."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = _check_base(base_url) if base_url != "http://lyre.internal" else base_url
        headers = {"Accept": "application/json"}
        if api_key:
            if any(char in api_key for char in "\r\n\x00"):
                raise LyreApiError(
                    0, "misconfigured", "API key contains a control character", "LYRE_API_KEY"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx2.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            transport=transport,
            timeout=httpx2.Timeout(30.0),
            follow_redirects=False,
        )

    async def __aenter__(self) -> LyreClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def create_song(self, body: Mapping[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return await self._json(
            "POST", "/v1/songs", json_body=dict(body), idempotency_key=idempotency_key
        )

    async def create_job(
        self, project_id: str, body: Mapping[str, Any], *, idempotency_key: str
    ) -> dict[str, Any]:
        path = f"/v1/projects/{_segment(project_id)}/jobs"
        return await self._json("POST", path, json_body=dict(body), idempotency_key=idempotency_key)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/v1/jobs/{_segment(job_id)}")

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._json("POST", f"/v1/jobs/{_segment(job_id)}/cancel", json_body={})

    async def list_projects(self, *, limit: int | None, cursor: str | None) -> dict[str, Any]:
        return await self._json("GET", "/v1/projects", params=_page(limit, cursor))

    async def get_project(self, project_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/v1/projects/{_segment(project_id)}")

    async def update_plan(self, project_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
        path = f"/v1/projects/{_segment(project_id)}/plan"
        return await self._json("PUT", path, json_body=dict(plan))

    async def list_takes(
        self, project_id: str, *, limit: int | None, cursor: str | None
    ) -> dict[str, Any]:
        path = f"/v1/projects/{_segment(project_id)}/takes"
        return await self._json("GET", path, params=_page(limit, cursor))

    async def get_capabilities(self) -> dict[str, Any]:
        return await self._json("GET", "/v1/capabilities")

    async def set_active_take(
        self, project_id: str, take_id: str, *, idempotency_key: str
    ) -> dict[str, Any]:
        path = f"/v1/projects/{_segment(project_id)}/active_take"
        return await self._json(
            "POST", path, json_body={"take_id": take_id}, idempotency_key=idempotency_key
        )

    async def export_manifest(
        self, project_id: str, *, take_id: str | None, include_stems: bool
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"include_stems": "true" if include_stems else "false"}
        if take_id:
            params["take_id"] = take_id
        path = f"/v1/projects/{_segment(project_id)}/export/manifest"
        return await self._json("GET", path, params=params)

    async def artifact_head(
        self, project_id: str, take_id: str, name: str
    ) -> tuple[int, int | None]:
        """HEAD one stable artifact. Returns `(status, content_length)` and does not read a body."""
        path = f"/v1/artifacts/{_segment(project_id)}/{_segment(take_id)}/{_segment(name)}"
        try:
            response = await self._http.request("HEAD", path)
        except httpx2.HTTPError as exc:
            raise _unreachable() from exc
        length = _content_length(response)
        if response.status_code in {401, 403}:
            _raise_api(response)
        return response.status_code, length

    async def _json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        content: bytes | None = None
        if json_body is not None:
            content = json.dumps(json_body, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = await self._http.request(
                method, path, content=content, headers=headers or None, params=params
            )
        except httpx2.HTTPError as exc:
            raise _unreachable() from exc
        if response.status_code >= 400:
            _raise_api(response)
        if response.status_code == 204 or not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise LyreApiError(
                response.status_code, "invalid_response", "Lyre returned a non-JSON body"
            ) from exc
        if not isinstance(payload, dict):
            raise LyreApiError(
                response.status_code, "invalid_response", "Lyre returned a non-object JSON body"
            )
        return payload


def _page(limit: int | None, cursor: str | None) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    if cursor is not None:
        params["cursor"] = cursor
    return params or None


def _content_length(response: httpx2.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def _unreachable() -> LyreApiError:
    return LyreApiError(0, "unavailable", "Lyre is not reachable at LYRE_URL")


def _raise_api(response: httpx2.Response) -> None:
    code = "http_error"
    message = f"Lyre returned HTTP {response.status_code}"
    param: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            raw_code = error.get("code")
            raw_message = error.get("message")
            raw_param = error.get("param")
            if isinstance(raw_code, str) and raw_code:
                code = raw_code
            if isinstance(raw_message, str) and raw_message:
                message = raw_message
            if isinstance(raw_param, str):
                param = raw_param
    raise LyreApiError(response.status_code, code, message, param)
