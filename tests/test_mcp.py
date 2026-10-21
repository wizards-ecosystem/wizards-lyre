"""MCP server (SPEC.md sec 15.6). Speaks JSON-RPC in-process. No GPU, no sockets.

`/v1` is either the real app (one enqueue) or `server.mcp.client.TRANSPORT`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anyio
import httpx2
import pytest
from fastapi.testclient import TestClient

from server.mcp import client as mcp_client
from server.mcp import service as mcp_service
from server.mcp.client import LyreApiError, LyreClient
from server.mcp.http import gateway
from server.mcp.server import main, mcp_server

PROTOCOL = "2026-07-28"
TASKS = "io.modelcontextprotocol/tasks"


def _job(**overrides: Any) -> dict[str, Any]:
    job = {
        "id": "job1",
        "project_id": "proj1",
        "action": "generate",
        "dit_profile": "iterate",
        "status": "queued",
        "stage": None,
        "take_id": None,
        "error": None,
        "seed": None,
        "created_at": "2026-10-21T00:00:00+00:00",
        "updated_at": "2026-10-21T00:00:01+00:00",
        "artifacts": [],
    }
    job.update(overrides)
    return job


class Scripted(httpx2.AsyncBaseTransport):
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)


def _json(status: int, payload: Any) -> httpx2.Response:
    return httpx2.Response(status, json=payload)


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> Scripted:
    scripted = Scripted(handler)
    monkeypatch.setattr(mcp_client, "TRANSPORT", scripted)
    return scripted


def _meta(extensions: dict[str, Any] | None = None) -> dict[str, Any]:
    capabilities: dict[str, Any] = {}
    if extensions is not None:
        capabilities["extensions"] = extensions
    return {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientCapabilities": capabilities,
        "io.modelcontextprotocol/clientInfo": {"name": "pytest", "version": "0"},
    }


def _rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    name: str | None = None,
    origin: str | None = None,
    token: str | None = None,
) -> httpx2.Response:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "mcp-protocol-version": PROTOCOL,
        "mcp-method": method,
    }
    if name is not None:
        headers["mcp-name"] = name
    if origin is not None:
        headers["origin"] = origin
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    return client.post("/mcp", content=json.dumps(body), headers=headers)


def _result(response: httpx2.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "error" not in payload, payload
    return payload["result"]


@pytest.fixture()
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_service, "POLL_INTERVAL_SEC", 0)


def test_lyre_url_must_be_an_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRE_URL", "http://user:pass@lyre.example/extra")
    with pytest.raises(LyreApiError, match="LYRE_URL"):
        mcp_client.public_base_url()


def test_public_base_url_brackets_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LYRE_URL", raising=False)
    monkeypatch.setenv("LYRE_BIND", "::1")
    monkeypatch.setenv("LYRE_PORT", "8421")
    assert mcp_client.public_base_url() == "http://[::1]:8421"


def test_client_maps_api_errors_and_refuses_a_bad_key() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/empty"):
            return httpx2.Response(204)
        if request.url.path.endswith("/text"):
            return httpx2.Response(200, content=b"nope")
        if request.url.path.endswith("/list"):
            return _json(200, ["nope"])
        if request.url.path.endswith("/plain"):
            return httpx2.Response(500, content=b"boom")
        return _json(
            422,
            {"error": {"code": "invalid_request", "message": "bad prompt", "param": "prompt"}},
        )

    scripted = Scripted(handler)

    async def run() -> None:
        async with LyreClient("http://127.0.0.1:8421", transport=scripted) as lyre:
            assert await lyre._json("GET", "/v1/empty") == {}
            with pytest.raises(LyreApiError, match="non-JSON"):
                await lyre._json("GET", "/v1/text")
            with pytest.raises(LyreApiError, match="non-object"):
                await lyre._json("GET", "/v1/list")
            with pytest.raises(LyreApiError, match="HTTP 500"):
                await lyre._json("GET", "/v1/plain")
            with pytest.raises(LyreApiError, match="bad prompt") as caught:
                await lyre._json("GET", "/v1/songs")
            assert caught.value.param == "prompt"
            status, length = await lyre.artifact_head("p", "t", "mix.wav")
        assert status == 422
        assert length is not None

    anyio.run(run)
    with pytest.raises(LyreApiError, match="control character"):
        LyreClient("http://127.0.0.1:8421", api_key="bad\nkey", transport=scripted)


def test_client_reports_an_unreachable_server() -> None:
    class Down(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("refused")

    async def run() -> None:
        async with LyreClient("http://127.0.0.1:9", transport=Down()) as lyre:
            with pytest.raises(LyreApiError, match="not reachable"):
                await lyre.get_job("job1")

    anyio.run(run)


def test_tools_list_is_static_and_discovers_tasks(api_client: TestClient) -> None:
    listed = _result(_rpc(api_client, "tools/list", {"_meta": _meta()}))
    names = [tool["name"] for tool in listed["tools"]]
    assert names == [
        "create_song",
        "transform_take",
        "get_job",
        "cancel_job",
        "list_projects",
        "get_project",
        "update_plan",
        "list_takes",
        "get_artifact",
    ]
    for tool in listed["tools"]:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["outputSchema"]
    discovered = _result(_rpc(api_client, "server/discover", {"_meta": _meta()}))
    assert discovered["supportedVersions"] == [PROTOCOL]
    capabilities = discovered["capabilities"]
    assert capabilities["tools"]["listChanged"] is False
    assert "resources" in capabilities
    assert TASKS in capabilities["extensions"]


def test_create_song_without_tasks_does_not_enqueue(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _patch_transport(monkeypatch, lambda request: _json(500, {}))
    response = _rpc(
        api_client,
        "tools/call",
        {"name": "create_song", "arguments": {"prompt": "rain"}, "_meta": _meta()},
        name="create_song",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32021
    assert scripted.requests == []


def test_create_song_returns_a_task_and_sends_an_idempotency_key(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _json(202, _job(stage=None))

    _patch_transport(monkeypatch, handler)
    monkeypatch.setenv("LYRE_API_KEY", "ignored-in-process")
    result = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "idempotency_key": "song-key-1"},
                "_meta": _meta({TASKS: {}}),
            },
            name="create_song",
        )
    )
    assert result["resultType"] == "task"
    assert result["taskId"] == "job1"
    assert result["status"] == "working"
    assert result["statusMessage"] == "queued"
    assert result["ttlMs"] is None
    assert result["pollIntervalMs"] == 2000
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/songs"
    assert seen["key"] == "song-key-1"
    assert seen["body"] == {"prompt": "rain"}
    assert seen["auth"] is None
    assert "audio" not in json.dumps(result)


def test_create_song_wait_returns_the_take_link(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if request.method == "POST":
            return _json(202, _job())
        return _json(
            200,
            _job(status="done", take_id="take1", artifacts=["/v1/artifacts/proj1/take1/mix.wav"]),
        )

    _patch_transport(monkeypatch, handler)
    result = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "wait_timeout_sec": 5},
                "_meta": _meta(),
            },
            name="create_song",
        )
    )
    assert result["resultType"] == "complete"
    assert result["isError"] is False
    assert result["structuredContent"]["status"] == "done"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    link = result["content"][1]
    assert link["type"] == "resource_link"
    assert link["uri"] == "lyre://take/proj1/take1"
    assert link["mimeType"] == "application/json"
    assert calls["n"] == 2


def test_wait_timeout_and_job_error(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    def queued(request: httpx2.Request) -> httpx2.Response:
        return _json(200 if request.method == "GET" else 202, _job())

    _patch_transport(monkeypatch, queued)
    waiting = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "wait_timeout_sec": 0},
                "_meta": _meta(),
            },
            name="create_song",
        )
    )
    assert waiting["isError"] is False
    assert "still queued" in waiting["content"][1]["text"]

    def failed(request: httpx2.Request) -> httpx2.Response:
        return _json(
            200 if request.method == "GET" else 202, _job(status="error", error="render failed")
        )

    _patch_transport(monkeypatch, failed)
    errored = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "wait_timeout_sec": 1},
                "_meta": _meta(),
            },
            name="create_song",
        )
    )
    assert errored["isError"] is True
    assert errored["structuredContent"]["error"] == "render failed"


def test_wait_rejects_a_bad_timeout(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _patch_transport(monkeypatch, lambda request: _json(202, _job()))
    result = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "wait_timeout_sec": -1},
                "_meta": _meta(),
            },
            name="create_song",
        )
    )
    assert result["isError"] is True
    assert "wait_timeout_sec" in result["content"][0]["text"]
    assert scripted.requests == []

    huge = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "wait_timeout_sec": 86401},
                "_meta": _meta({TASKS: {}}),
            },
            name="create_song",
        )
    )
    assert huge["isError"] is True
    assert scripted.requests == []


def test_transform_take_and_invalid_action(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.raw_path == b"/v1/projects/proj%2F1/jobs"
        assert request.headers["idempotency-key"]
        assert json.loads(request.content)["action"] == "repaint"
        return _json(202, _job(action="repaint", project_id="proj/1"))

    _patch_transport(monkeypatch, handler)
    result = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "transform_take",
                "arguments": {
                    "project_id": "proj/1",
                    "action": "repaint",
                    "source_take_id": "take1",
                    "repainting_start": 1.5,
                },
                "_meta": _meta({TASKS: {}}),
            },
            name="transform_take",
        )
    )
    assert result["resultType"] == "task"
    assert result["taskId"] == "job1"

    bad = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "transform_take",
                "arguments": {"project_id": "proj1", "action": "train_lora"},
                "_meta": _meta({TASKS: {}}),
            },
            name="transform_take",
        )
    )
    assert bad["isError"] is True
    assert "action" in bad["content"][0]["text"]


def test_tasks_follow_the_job(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"job": _job(stage="rendering")}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            state["job"] = {**state["job"], "status": "cancelled", "stage": None}
            return _json(200, state["job"])
        if state["job"]["id"] == "missing":
            return _json(
                404,
                {
                    "error": {
                        "code": "not_found",
                        "message": "job not found: missing",
                        "param": None,
                    }
                },
            )
        return _json(200, state["job"])

    scripted = _patch_transport(monkeypatch, handler)
    meta = _meta({TASKS: {}})

    working = _result(_rpc(api_client, "tasks/get", {"taskId": "job1", "_meta": meta}, name="job1"))
    assert working["status"] == "working"
    assert working["statusMessage"] == "rendering"

    state["job"] = _job(
        status="done", take_id="take1", artifacts=["/v1/artifacts/proj1/take1/mix.mp3"]
    )
    done = _result(_rpc(api_client, "tasks/get", {"taskId": "job1", "_meta": meta}, name="job1"))
    assert done["status"] == "completed"
    assert done["result"]["structuredContent"]["take_id"] == "take1"
    assert done["result"]["content"][1]["type"] == "resource_link"

    state["job"] = _job(status="error", error="nope")
    failed = _result(_rpc(api_client, "tasks/get", {"taskId": "job1", "_meta": meta}, name="job1"))
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "nope"

    state["job"] = _job(status="cancelled", stage=None, error=None, take_id=None)
    cancelled = _result(
        _rpc(api_client, "tasks/get", {"taskId": "job1", "_meta": meta}, name="job1")
    )
    assert cancelled["status"] == "cancelled"

    updated = _result(
        _rpc(api_client, "tasks/update", {"taskId": "job1", "_meta": meta}, name="job1")
    )
    assert updated["resultType"] == "complete"
    assert all(
        request.method == "GET"
        for request in scripted.requests
        if request.url.path.endswith("/job1")
    )

    acked = _result(
        _rpc(api_client, "tasks/cancel", {"taskId": "job1", "_meta": meta}, name="job1")
    )
    assert acked["resultType"] == "complete"
    assert any(
        request.method == "POST" and request.url.path.endswith("/cancel")
        for request in scripted.requests
    )

    state["job"] = _job(id="missing")
    missing = _rpc(api_client, "tasks/get", {"taskId": "missing", "_meta": meta}, name="missing")
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == -32602

    bare = _rpc(api_client, "tasks/get", {"taskId": "job1", "_meta": _meta()}, name="job1")
    assert bare.status_code == 400
    assert bare.json()["error"]["code"] == -32021


def test_task_name_header_must_match(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _patch_transport(monkeypatch, lambda request: _json(500, {}))
    response = _rpc(
        api_client,
        "tasks/get",
        {"taskId": "job1", "_meta": _meta({TASKS: {}})},
        name="other",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32020
    assert scripted.requests == []


def test_sync_tools_and_resources(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    project = {
        "project": {"id": "proj1", "title": "Rain"},
        "plan": {"query": "rain"},
        "takes": [{"id": "take1", "caption": "night"}],
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/v1/projects" and request.method == "GET":
            assert request.url.params["limit"] == "2"
            assert request.url.params["cursor"] == "abc"
            return _json(200, {"data": [{"id": "proj1"}], "next_cursor": None})
        if path == "/v1/projects/proj1" and request.method == "GET":
            return _json(200, project)
        if path == "/v1/projects/proj1/plan" and request.method == "PUT":
            return _json(200, {"query": "updated"})
        if path.endswith("/takes"):
            return _json(200, {"data": [{"id": "take1"}], "next_cursor": "next"})
        if path == "/v1/jobs/job1" and request.method == "GET":
            return _json(200, _job(status="running", stage="planning"))
        if request.method == "HEAD":
            if path.endswith("lyrics.lrc"):
                return httpx2.Response(404)
            return httpx2.Response(200, headers={"content-length": "12"})
        return _json(500, {"error": {"code": "nope", "message": path, "param": None}})

    _patch_transport(monkeypatch, handler)
    monkeypatch.setenv("LYRE_URL", "http://127.0.0.1:9999")
    projects = _result(
        _rpc(
            api_client,
            "tools/call",
            {"name": "list_projects", "arguments": {"limit": 2, "cursor": "abc"}, "_meta": _meta()},
            name="list_projects",
        )
    )
    assert projects["structuredContent"]["data"][0]["id"] == "proj1"

    got = _result(
        _rpc(
            api_client,
            "tools/call",
            {"name": "get_project", "arguments": {"project_id": "proj1"}, "_meta": _meta()},
            name="get_project",
        )
    )
    assert got["structuredContent"]["project"]["title"] == "Rain"

    plan = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "update_plan",
                "arguments": {"project_id": "proj1", "plan": {"query": "updated"}},
                "_meta": _meta(),
            },
            name="update_plan",
        )
    )
    assert plan["structuredContent"]["query"] == "updated"

    takes = _result(
        _rpc(
            api_client,
            "tools/call",
            {"name": "list_takes", "arguments": {"project_id": "proj1"}, "_meta": _meta()},
            name="list_takes",
        )
    )
    assert takes["structuredContent"]["next_cursor"] == "next"

    job = _result(
        _rpc(
            api_client,
            "tools/call",
            {"name": "get_job", "arguments": {"job_id": "job1"}, "_meta": _meta()},
            name="get_job",
        )
    )
    assert job["structuredContent"]["stage"] == "planning"

    artifact = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "get_artifact",
                "arguments": {"project_id": "proj1", "take_id": "take1"},
                "_meta": _meta(),
            },
            name="get_artifact",
        )
    )
    files = {item["name"]: item for item in artifact["structuredContent"]["files"]}
    assert files["mix.wav"]["available"] is True
    assert files["mix.wav"]["content_length"] == 12
    assert (
        files["mix.wav"]["absolute_url"] == "http://127.0.0.1:9999/v1/artifacts/proj1/take1/mix.wav"
    )
    assert files["lyrics.lrc"]["available"] is False
    assert files["lyrics.lrc"]["content_length"] is None
    assert "data" not in json.dumps(artifact)

    listed = _result(_rpc(api_client, "resources/templates/list", {"_meta": _meta()}))
    templates = {item["uriTemplate"] for item in listed["resourceTemplates"]}
    assert "lyre://project/{project_id}" in templates
    assert "lyre://take/{project_id}/{take_id}" in templates

    project_text = _result(
        _rpc(
            api_client,
            "resources/read",
            {"uri": "lyre://project/proj1", "_meta": _meta()},
            name="lyre://project/proj1",
        )
    )
    assert json.loads(project_text["contents"][0]["text"])["plan"]["query"] == "rain"
    take_text = _result(
        _rpc(
            api_client,
            "resources/read",
            {"uri": "lyre://take/proj1/take1", "_meta": _meta()},
            name="lyre://take/proj1/take1",
        )
    )
    assert json.loads(take_text["contents"][0]["text"])["id"] == "take1"
    missing = _rpc(
        api_client,
        "resources/read",
        {"uri": "lyre://take/proj1/nope", "_meta": _meta()},
        name="lyre://take/proj1/nope",
    )
    assert missing.json()["error"]["code"] == -32602


def test_bad_idempotency_key_and_unreachable_url(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted = _patch_transport(monkeypatch, lambda request: _json(202, _job()))
    bad_key = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "rain", "idempotency_key": "has space"},
                "_meta": _meta({TASKS: {}}),
            },
            name="create_song",
        )
    )
    assert bad_key["isError"] is True
    assert scripted.requests == []

    monkeypatch.setattr(mcp_client, "TRANSPORT", None)
    monkeypatch.setenv("LYRE_URL", "ftp://lyre.example")

    async def blow_up() -> None:
        with pytest.raises(Exception, match="LYRE_URL"):
            async with mcp_service._session():
                pass

    anyio.run(blow_up)


def test_origin_is_rejected_and_missing_origin_is_allowed(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = api_client.get("/v1/jobs").json()["data"]
    rejected = _rpc(
        api_client,
        "tools/call",
        {
            "name": "create_song",
            "arguments": {"prompt": "from a web page"},
            "_meta": _meta({TASKS: {}}),
        },
        name="create_song",
        origin="https://evil.example",
    )
    assert rejected.status_code == 403
    assert rejected.json() == {
        "error": {"code": "forbidden", "message": "origin not allowed", "param": "Origin"}
    }
    assert api_client.get("/v1/jobs").json()["data"] == before

    for origin in (
        "http://127.0.0.1:9",
        "http://localhost:3000",
        "https://[::1]:8443",
        "http://LOCALHOST",
        "http://127.0.0.1:80",
    ):
        allowed = _rpc(api_client, "tools/list", {"_meta": _meta()}, origin=origin)
        assert allowed.status_code == 200, origin

    for origin in (
        "http://127.0.0.2:8421",
        "http://127.0.0.1:8421/mcp",
        "null",
        "http://evil.example",
    ):
        blocked = _rpc(api_client, "tools/list", {"_meta": _meta()}, origin=origin)
        assert blocked.status_code == 403, origin

    missing = _rpc(api_client, "tools/list", {"_meta": _meta()})
    assert missing.status_code == 200

    monkeypatch.setenv(
        "LYRE_ALLOWED_ORIGINS",
        "https://studio.example/, http://203.0.113.5:8421",
    )
    listed = _rpc(
        api_client,
        "tools/list",
        {"_meta": _meta()},
        origin="https://studio.example",
    )
    assert listed.status_code == 200
    other_port = _rpc(
        api_client,
        "tools/list",
        {"_meta": _meta()},
        origin="http://203.0.113.5:9",
    )
    assert other_port.status_code == 403
    same = _rpc(
        api_client,
        "tools/list",
        {"_meta": _meta()},
        origin="http://203.0.113.5:8421",
    )
    assert same.status_code == 200


def test_streamable_http_enqueues_the_same_generate_job(api_client: TestClient) -> None:
    result = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "a rainy night jazz ballad", "title": "Night"},
                "_meta": _meta({TASKS: {}}),
            },
            name="create_song",
        )
    )
    assert result["resultType"] == "task"
    job = api_client.get(f"/v1/jobs/{result['taskId']}")
    assert job.status_code == 200
    body = job.json()
    assert body["action"] == "generate"
    assert body["status"] == "queued"
    assert body["project_id"]
    project = api_client.get(f"/v1/projects/{body['project_id']}")
    assert project.json()["plan"]["query"] == "a rainy night jazz ballad"


def test_auth_on_post_mcp_uses_the_bearer(
    api_client: TestClient, lyre_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.api_keys import create_key

    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    _record, secret = create_key("agent", ["generate"])
    _reader, read_secret = create_key("reader", ["read"])
    missing = _rpc(
        api_client,
        "tools/call",
        {"name": "create_song", "arguments": {"prompt": "rain"}, "_meta": _meta({TASKS: {}})},
        name="create_song",
    )
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert api_client.get("/").status_code == 404
    opened = _rpc(api_client, "tools/list", {"_meta": _meta()}, token=secret)
    assert opened.status_code == 200
    created = _result(
        _rpc(
            api_client,
            "tools/call",
            {
                "name": "create_song",
                "arguments": {"prompt": "auth song"},
                "_meta": _meta({TASKS: {}}),
            },
            name="create_song",
            token=secret,
        )
    )
    assert created["resultType"] == "task"
    hidden = api_client.get(f"/v1/jobs/{created['taskId']}")
    assert hidden.status_code == 401
    visible = api_client.get(
        f"/v1/jobs/{created['taskId']}", headers={"authorization": f"Bearer {secret}"}
    )
    assert visible.status_code == 200
    denied = _result(
        _rpc(
            api_client,
            "tools/call",
            {"name": "list_projects", "arguments": {}, "_meta": _meta()},
            name="list_projects",
            token=secret,
        )
    )
    assert denied["isError"] is True
    assert "read" in denied["content"][0]["text"]
    assert _rpc(api_client, "tools/list", {"_meta": _meta()}, token=read_secret).status_code == 200


def test_gateway_503_and_misconfigured_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server.mcp.http._inner", None)

    async def run_503() -> httpx2.Response:
        transport = httpx2.ASGITransport(app=gateway)
        async with httpx2.AsyncClient(transport=transport, base_url="http://lyre") as http:
            return await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})

    down = anyio.run(run_503)
    assert down.status_code == 503

    from server import config

    def broken() -> bool:
        raise config.BindConfigError("secret path should stay in the log")

    monkeypatch.setattr("server.mcp.http._inner", object())
    monkeypatch.setattr(config, "auth_required", broken)

    async def run_500() -> httpx2.Response:
        transport = httpx2.ASGITransport(app=gateway)
        async with httpx2.AsyncClient(transport=transport, base_url="http://lyre") as http:
            return await http.get("/mcp")

    failed = anyio.run(run_500)
    assert failed.status_code == 500
    assert "secret path" not in failed.text
    assert failed.json()["error"]["code"] == "misconfigured"


def test_stdio_frames_a_jsonrpc_response(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = _patch_transport(monkeypatch, lambda request: _json(500, {}))
    message = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "create_song",
            "arguments": {"prompt": "stdio"},
            "_meta": _meta(),
        },
    }

    async def roundtrip() -> str:
        with anyio.fail_after(5):
            return await _speak()

    async def _speak() -> str:
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        stdin = anyio.wrap_file(os.fdopen(in_r, "r", encoding="utf-8"))
        stdout = anyio.wrap_file(os.fdopen(out_w, "w", encoding="utf-8"))
        to_server = os.fdopen(in_w, "w", encoding="utf-8")
        from_server = os.fdopen(out_r, "r", encoding="utf-8")
        from mcp.server.stdio import stdio_server

        async def serve() -> None:
            async with stdio_server(stdin, stdout) as streams:
                await mcp_server._lowlevel_server.run(
                    streams[0],
                    streams[1],
                    mcp_server._lowlevel_server.create_initialization_options(),
                )

        line = ""

        async def speak() -> None:
            nonlocal line
            to_server.write(json.dumps(message) + "\n")
            to_server.flush()
            line = await anyio.to_thread.run_sync(from_server.readline)
            to_server.close()

        async with anyio.create_task_group() as tg:
            tg.start_soon(serve)
            tg.start_soon(speak)
        from_server.close()
        return line

    raw = anyio.run(roundtrip)
    payload = json.loads(raw)
    assert payload["id"] == 7
    assert payload["error"]["code"] == -32021
    assert scripted.requests == []
    assert raw.endswith("\n")


def test_main_uses_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_run(transport: str = "stdio", **kwargs: Any) -> None:
        called["transport"] = transport
        called["kwargs"] = kwargs

    monkeypatch.setattr(mcp_server, "run", fake_run)
    main()
    assert called == {"transport": "stdio", "kwargs": {}}


def test_content_length_ignores_junk() -> None:
    response = httpx2.Response(200, headers={"content-length": "nope"})
    assert mcp_client._content_length(response) is None
