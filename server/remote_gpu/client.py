"""The HTTP protocol between Lyre and a Remote GPU host (`worker.remote_host`).

Standard library only: this lane is optional, and it should not add a runtime
dependency to an app that is local-first by default. Every call goes through
`TRANSPORT`, which tests replace to route requests into an in-process host or a
fake provider without touching the network.

Every error message that can reach a log or the UI has the shared secret
scrubbed out of it first.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from server.remote_gpu.connection import Connection

SECRET_HEADER = "X-Lyre-Secret"
# Runpod's API and pod proxy sit behind Cloudflare, which refuses Python's
# default "Python-urllib/x.y" agent outright (error 1010, a 403 that looks
# like a permissions problem). Every request names itself instead.
USER_AGENT = "wizards-lyre/0.2 (+https://github.com/wizards-ecosystem/wizards-lyre)"

# (method, url, headers, body, timeout, max_bytes) -> (status, body)
Transport = Callable[[str, str, dict[str, str], bytes | None, float, int], tuple[int, bytes]]

CONTROL_TIMEOUT_SEC = 30.0
# Audio can be ~100 MB; allow a slow link time to move it.
AUDIO_TIMEOUT_SEC = 600.0
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_AUDIO_BYTES = 512 * 1024 * 1024


class RemoteGpuError(RuntimeError):
    """A Remote GPU call failed. `status` is the HTTP status, or None when the
    host could not be reached at all."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect would replay the secret header to wherever it points.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
    max_bytes: int,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status = response.status
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        status, data = exc.code, exc.read(64 * 1024)
    if len(data) > max_bytes:
        raise RemoteGpuError(f"{url} returned more than {max_bytes} bytes", status)
    return status, data


TRANSPORT: Transport = urllib_transport


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: Any = None,
    timeout: float = CONTROL_TIMEOUT_SEC,
    max_bytes: int = MAX_JSON_BYTES,
    redact: tuple[str, ...] = (),
) -> tuple[int, bytes]:
    """One HTTP exchange through `TRANSPORT`, with `redact` scrubbed from any
    error it raises. Shared by the host client and the provisioners."""
    all_headers = dict(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        all_headers.setdefault("Content-Type", "application/json")
    all_headers.setdefault("Accept", "application/json")
    all_headers.setdefault("User-Agent", USER_AGENT)

    def _scrub(text: str) -> str:
        for value in redact:
            if value:
                text = text.replace(value, "***")
        return text

    try:
        return TRANSPORT(method, url, all_headers, body, timeout, max_bytes)
    except RemoteGpuError as exc:
        raise RemoteGpuError(_scrub(str(exc)), exc.status) from None
    except (OSError, ValueError) as exc:
        # URLError, timeouts and connection resets are all OSErrors.
        raise RemoteGpuError(_scrub(f"{url} is unreachable: {exc}")) from None


def _json(status: int, data: bytes, what: str) -> Any:
    try:
        return json.loads(data or b"{}")
    except ValueError:
        raise RemoteGpuError(f"{what} returned a response that is not JSON", status) from None


def _detail(status: int, data: bytes) -> str:
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict) and parsed.get("detail"):
            return str(parsed["detail"])
    except ValueError:
        pass
    return data[:300].decode("utf-8", errors="replace") or f"HTTP {status}"


class RemoteClient:
    """Calls to one host. Cheap to construct; holds no connection open."""

    def __init__(self, connection: Connection) -> None:
        self.base_url = connection.base_url
        self._secret = connection.secret

    def _call(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        timeout: float = CONTROL_TIMEOUT_SEC,
        max_bytes: int = MAX_JSON_BYTES,
    ) -> tuple[int, bytes]:
        status, data = request(
            method,
            f"{self.base_url}{path}",
            headers={SECRET_HEADER: self._secret},
            payload=payload,
            timeout=timeout,
            max_bytes=max_bytes,
            redact=(self._secret,),
        )
        if status == 401:
            raise RemoteGpuError("the Remote GPU rejected the shared secret", status)
        if status >= 400:
            detail = _detail(status, data).replace(self._secret, "***")
            raise RemoteGpuError(f"Remote GPU error {status}: {detail}", status)
        return status, data

    def health(self) -> dict[str, Any]:
        status, data = self._call("GET", "/health", timeout=15.0)
        return _json(status, data, "the Remote GPU health check")

    def submit(self, body: dict[str, Any]) -> str:
        status, data = self._call("POST", "/jobs", body, timeout=AUDIO_TIMEOUT_SEC)
        token = _json(status, data, "the Remote GPU").get("token")
        if not isinstance(token, str) or not token:
            raise RemoteGpuError("the Remote GPU accepted the job but returned no token", status)
        return token

    def job(self, token: str) -> dict[str, Any]:
        status, data = self._call("GET", f"/jobs/{token}")
        return _json(status, data, "the Remote GPU")

    def audio(self, token: str) -> bytes:
        _, data = self._call(
            "GET", f"/jobs/{token}/audio", timeout=AUDIO_TIMEOUT_SEC, max_bytes=MAX_AUDIO_BYTES
        )
        return data

    def ack(self, token: str) -> None:
        self._call("POST", f"/jobs/{token}/ack")

    def cancel(self, token: str) -> None:
        self._call("POST", f"/jobs/{token}/cancel")


# The UI polls status every few seconds; the worker checks readiness on its
# own heartbeat. Neither needs a fresh round trip to the host every time.
HEALTH_CACHE_SEC = 5.0
_HEALTH_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def probe(connection: Connection, *, max_age: float = HEALTH_CACHE_SEC) -> dict[str, Any]:
    """The host's `/health`, folded into `{"connected": bool, ...}`. Never raises:
    an unreachable host is a status to show, not an error."""
    key = (connection.base_url, connection.secret)
    cached = _HEALTH_CACHE.get(key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < max_age:
        return cached[1]
    try:
        result = {"connected": True, "error": None, **RemoteClient(connection).health()}
    except RemoteGpuError as exc:
        result = {"connected": False, "error": str(exc)}
    _HEALTH_CACHE.clear()
    _HEALTH_CACHE[key] = (now, result)
    return result


def forget_probe() -> None:
    _HEALTH_CACHE.clear()
