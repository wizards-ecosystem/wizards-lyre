"""`Idempotency-Key` for every `/v1` POST (SPEC.md sec 15.3).

The key, the request hash, and the response live in the jobs SQLite
database for 24 hours. The same key and the same raw body replay the
stored response and do not run the handler again. The same key and a
different body is `409`. A replay of a POST that queued a job
(`POST /v1/songs` and `POST /v1/projects/{id}/jobs`) is `200`; the first
response stays `202`.

The key is optional. Callers that omit it get ordinary, non-replayed
behavior. When auth is on, the row is scoped to the API key id so one
agent's key cannot replay another's response. The secret itself is not
stored. When auth is off, every caller on this machine shares one scope,
which matches the single-operator studio.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.requests import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from server import config
from server.api_keys import ApiKeyStoreError, authenticate
from server.jobs.db import _connect

# Visible ASCII, no spaces. Long enough for a UUID, short enough to index.
_KEY_RE = re.compile(r"^[!-~]{1,255}$")
# These POSTs answer 202 because they queued a job. The stored replay is 200.
_QUEUED_JOB_POST = re.compile(r"^/v1/(?:songs|projects/[^/]+/jobs)$")
IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60

# Auth-off requests share this owner. It is not an API key id.
_ANONYMOUS = ""


class IdempotencyConflict(Exception):
    """Same key, different body, or the first request is still running."""


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _error(status: int, code: str, message: str, param: str | None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "param": param}},
    )


def request_hash(method: str, path: str, query: bytes, body: bytes) -> str:
    """Hash the raw request. Two JSON objects that differ only by a space
    are different bodies, which is what `same body` means here."""
    digest = hashlib.sha256()
    digest.update(method.encode("ascii"))
    digest.update(b"\0")
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(query)
    digest.update(b"\0")
    digest.update(body)
    return digest.hexdigest()


def owner_for(headers: Headers) -> str | None:
    """Who this key belongs to.

    `None` means idempotency does not apply and the route should answer
    on its own: auth is required and this request is not a known bearer,
    or the bind configuration is broken. `""` means auth is off.
    """
    try:
        required = config.auth_required()
    except config.BindConfigError:
        return None
    if not required:
        return _ANONYMOUS
    header = headers.get("authorization")
    if header is None:
        return None
    scheme, _, secret = header.partition(" ")
    token = secret.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    try:
        record = authenticate(token)
    except ApiKeyStoreError:
        return None
    if record is None:
        return None
    return str(record["id"])


def _purge_expired(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM idempotency_keys WHERE expires_at <= ?", (_now(),))


def reserve(owner_id: str, key: str, digest: str) -> Response | None:
    """Insert a pending row, or return the stored response.

    Raises `IdempotencyConflict` when the key is in use with another body
    or the first request has not finished. The caller deletes the pending
    row if the handler raises or returns 5xx, so a retry can run again.
    """
    with closing(_connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _purge_expired(conn)
            row = conn.execute(
                "SELECT request_hash, state, status_code, body, location, content_type "
                "FROM idempotency_keys WHERE owner_id = ? AND idempotency_key = ?",
                (owner_id, key),
            ).fetchone()
            if row is None:
                now = _now()
                conn.execute(
                    "INSERT INTO idempotency_keys ("
                    "owner_id, idempotency_key, request_hash, state, status_code, body, "
                    "location, content_type, created_at, expires_at"
                    ") VALUES (?, ?, ?, 'pending', 0, ?, NULL, NULL, ?, ?)",
                    (owner_id, key, digest, b"", now, now + IDEMPOTENCY_TTL_SECONDS),
                )
                conn.commit()
                return None
            same = row["request_hash"] == digest and row["state"] == "done"
            replay = _replay(row) if same else None
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
    if row is not None and row["request_hash"] != digest:
        raise IdempotencyConflict("Idempotency-Key was already used with a different body")
    if row is not None and row["state"] != "done":
        raise IdempotencyConflict("Idempotency-Key is already in progress")
    return replay


def release(owner_id: str, key: str) -> None:
    """Drop a pending row so a failed request can be retried with the same key."""
    with closing(_connect()) as conn:
        conn.execute(
            "DELETE FROM idempotency_keys "
            "WHERE owner_id = ? AND idempotency_key = ? AND state = 'pending'",
            (owner_id, key),
        )
        conn.commit()


def remember(
    owner_id: str,
    key: str,
    path: str,
    status: int,
    body: bytes,
    location: str | None,
    content_type: str | None,
) -> None:
    """Store a finished response. A queued job replays as `200`."""
    stored = 200 if status == 202 and _QUEUED_JOB_POST.fullmatch(path) else status
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE idempotency_keys SET state = 'done', status_code = ?, body = ?, "
            "location = ?, content_type = ? "
            "WHERE owner_id = ? AND idempotency_key = ? AND state = 'pending'",
            (stored, body, location, content_type, owner_id, key),
        )
        conn.commit()


def _replay(row) -> Response:
    headers: dict[str, str] = {}
    if row["location"]:
        headers["Location"] = row["location"]
    media = "application/json"
    raw_type = row["content_type"]
    if isinstance(raw_type, str) and raw_type:
        media = raw_type.split(";", 1)[0].strip() or media
    return Response(
        content=bytes(row["body"]),
        status_code=int(row["status_code"]),
        media_type=media,
        headers=headers,
    )


def _header(headers: list[tuple[bytes, bytes]], name: str) -> str | None:
    needle = name.lower().encode("ascii")
    for key, value in headers:
        if key.lower() == needle:
            return value.decode("latin-1")
    return None


def _as_bytes(chunk: bytes | memoryview) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    return bytes(chunk)


async def _buffer_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            continue
        chunks.append(_as_bytes(message.get("body") or b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


def _http_error(exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    if exc.status_code == 413:
        code = "payload_too_large"
    elif exc.status_code == 404:
        code = "not_found"
    else:
        code = "invalid_request"
    return _error(exc.status_code, code, message, None)


class IdempotencyMiddleware:
    """Replay or record `/v1` POSTs that send `Idempotency-Key`.

    Sits inside the body-size limit and outside routing, so the hash is of
    the bytes the handler will see, and a stored `400` is the response the
    exception handlers already produced.
    """

    def __init__(self, asgi_app: ASGIApp) -> None:
        self.app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        if not path.startswith("/v1"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if "idempotency-key" not in headers:
            await self.app(scope, receive, send)
            return

        owner = owner_for(headers)
        if owner is None:
            # Missing, unknown, or unreadable credentials. The route returns
            # 401/500 itself; do not consume the key on the way there.
            await self.app(scope, receive, send)
            return

        key = (headers.get("idempotency-key") or "").strip()
        if _KEY_RE.fullmatch(key) is None:
            response = _error(
                400,
                "invalid_request",
                "Idempotency-Key must be 1-255 visible ASCII characters with no spaces",
                "Idempotency-Key",
            )
            await response(scope, receive, send)
            return

        try:
            body = await _buffer_body(receive)
        except HTTPException as exc:
            # The body-limit middleware raises while we read. That raise is
            # outside the router's exception handlers, so answer it here.
            await _http_error(exc)(scope, receive, send)
            return

        digest = request_hash(
            scope.get("method") or "POST", path, scope.get("query_string") or b"", body
        )
        try:
            replay = reserve(owner, key, digest)
        except IdempotencyConflict as exc:
            await _error(409, "idempotency_conflict", str(exc), "Idempotency-Key")(
                scope, receive, send
            )
            return
        if replay is not None:
            await replay(scope, receive, send)
            return

        sent = False

        async def replay_receive() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        captured_status = 0
        captured_headers: list[tuple[bytes, bytes]] = []
        captured_body: list[bytes] = []
        finished = False

        async def capture_send(message: Message) -> None:
            nonlocal captured_status, finished
            if message["type"] == "http.response.start":
                captured_status = int(message["status"])
                captured_headers[:] = list(message.get("headers") or [])
                await send(message)
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            captured_body.append(_as_bytes(message.get("body") or b""))
            await send(message)
            if message.get("more_body", False) or finished:
                return
            finished = True
            if captured_status >= 500 or captured_status == 0:
                release(owner, key)
                return
            remember(
                owner,
                key,
                path,
                captured_status,
                b"".join(captured_body),
                _header(captured_headers, "location"),
                _header(captured_headers, "content-type"),
            )

        try:
            await self.app(scope, replay_receive, capture_send)
        except Exception:
            release(owner, key)
            raise
