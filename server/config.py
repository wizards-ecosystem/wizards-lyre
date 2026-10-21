"""Runtime paths and bind settings. No CUDA here. See SPEC.md sec 5, 6, 8."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PORT = 8421
# The source default stays this loopback address. Operators who want another
# interface set LYRE_BIND themselves. Do not replace this with a public bind.
HOST = "127.0.0.1"

# SPEC.md sec 15.2. Anything else, including an operator-supplied
# all-interfaces address, is a network bind.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class BindConfigError(RuntimeError):
    """The process must not listen. `server.app.main` prints this and exits."""


def lyre_port() -> int:
    return int(os.environ.get("LYRE_PORT", DEFAULT_PORT))


def bind_host() -> str:
    """`LYRE_BIND`, or `127.0.0.1` when it is unset. Blank is the default too,
    so an empty assignment in `.env` does not become a network bind."""
    raw = os.environ.get("LYRE_BIND")
    if raw is None:
        return HOST
    host = raw.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host or HOST


def is_loopback(host: str | None = None) -> bool:
    return (host if host is not None else bind_host()).lower() in _LOOPBACK_HOSTS


def _require_auth_flag() -> str | None:
    raw = os.environ.get("LYRE_REQUIRE_AUTH")
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip()
    if value not in {"0", "1"}:
        raise BindConfigError("LYRE_REQUIRE_AUTH must be 0 or 1")
    return value


def auth_required() -> bool:
    """True when requests must carry an API key and the studio is hidden.

    Loopback with the flag unset is the studio install: auth off.
    `LYRE_REQUIRE_AUTH=1` turns auth on even on loopback (a tunnel or
    reverse proxy makes remote traffic look local). A non-loopback bind
    always requires auth. `LYRE_REQUIRE_AUTH=0` on a network bind is a
    configuration error, not a way to turn the check off.
    """
    flag = _require_auth_flag()
    if not is_loopback():
        if flag == "0":
            raise BindConfigError(
                "LYRE_REQUIRE_AUTH=0 is not allowed when LYRE_BIND is not a loopback address"
            )
        return True
    return flag == "1"


def validate_startup() -> str:
    """Return the bind host, or raise `BindConfigError` before any socket
    is opened. A network bind, and loopback with `LYRE_REQUIRE_AUTH=1`,
    refuse to start when no API key is stored."""
    host = bind_host()
    if any(char.isspace() for char in host):
        raise BindConfigError("LYRE_BIND must be a single address")
    required = auth_required()
    if required:
        # Imported here so config stays free of the key file format.
        from server.api_keys import ApiKeyStoreError, has_any_key

        try:
            stored = has_any_key()
        except ApiKeyStoreError as exc:
            raise BindConfigError(str(exc)) from exc
        if not stored:
            raise BindConfigError(
                f"refusing to listen on {host}: no API key is stored. "
                "Create one with ./scripts/lyre api-key create --name <name>"
            )
    return host


def projects_dir() -> Path:
    raw = os.environ.get("LYRE_PROJECTS_DIR")
    path = Path(raw) if raw else REPO_ROOT / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def output_dir() -> Path:
    raw = os.environ.get("LYRE_OUTPUT_DIR")
    path = Path(raw) if raw else REPO_ROOT / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def db_path() -> Path:
    raw = os.environ.get("LYRE_DB_PATH")
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    return projects_dir() / "lyre.db"
