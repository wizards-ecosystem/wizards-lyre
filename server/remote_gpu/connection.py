"""The Remote GPU this install talks to, and any rental it is running.

State, not configuration: the UI writes a pasted address here, a provisioner
writes the pod it just started, and the worker process reads it before each
job, so connecting a GPU never needs a restart. It sits beside the jobs
database and is readable only by its owner, because it holds the shared secret.

A rental is recorded here *before* a provisioner reports it started. The app
can die in ways that run no shutdown code, and a pod nobody remembers keeps
billing; on the next start this record is how it gets found and stopped.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from server import config

FILE_NAME = "remote_gpu.json"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Connection:
    base_url: str
    secret: str


def path() -> Path:
    return config.db_path().parent / FILE_NAME


def _read() -> dict[str, Any]:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Missing is the normal case; a corrupt file must not stop the app.
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict[str, Any]) -> None:
    target = path()
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.replace(tmp, target)


def _update(key: str, value: Any) -> None:
    with _LOCK:
        data = _read()
        if value is None:
            if key not in data:
                return
            data.pop(key)
        else:
            data[key] = value
        _write(data)


def normalize_base_url(raw: str) -> str:
    """A host address the app is willing to send the secret to.

    HTTPS only, except to this machine itself (a host reached through an SSH
    tunnel, or one run locally to try it out). No credentials, query or
    fragment, and no path: the protocol's routes hang off the root.
    """
    url = (raw or "").strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ValueError("the Remote GPU address must be an https:// URL")
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "the Remote GPU address must use https:// (plain http is only allowed to "
            "this machine), because every request carries the shared secret"
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("the Remote GPU address must not contain credentials, a query, or #")
    if parts.path not in ("", "/"):
        raise ValueError("the Remote GPU address must be the host's root, without a path")
    return f"{parts.scheme}://{parts.netloc}"


def get_connection() -> Connection | None:
    raw = _read().get("connection")
    if not isinstance(raw, dict):
        return None
    base_url, secret = raw.get("base_url"), raw.get("secret")
    if not isinstance(base_url, str) or not isinstance(secret, str) or not base_url or not secret:
        return None
    return Connection(base_url=base_url, secret=secret)


def set_connection(base_url: str, secret: str) -> Connection:
    secret = (secret or "").strip()
    if len(secret) < 24:
        raise ValueError("the shared secret must be at least 24 characters")
    connection = Connection(base_url=normalize_base_url(base_url), secret=secret)
    _update("connection", {"base_url": connection.base_url, "secret": connection.secret})
    return connection


def clear_connection() -> None:
    _update("connection", None)


def get_session() -> dict[str, Any] | None:
    raw = _read().get("session")
    return raw if isinstance(raw, dict) and raw.get("id") else None


def save_session(session: dict[str, Any]) -> None:
    _update("session", session)


def clear_session() -> None:
    _update("session", None)
