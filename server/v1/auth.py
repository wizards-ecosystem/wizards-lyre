"""Bearer-key dependency for `/v1` (SPEC.md sec 15.2).

Auth off (the default loopback install) accepts every request. Auth on
requires `Authorization: Bearer <secret>`. A missing or unknown secret is
401. A known secret without the scope is 403.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from server import config
from server.api_keys import ApiKeyStoreError, authenticate

_bearer = HTTPBearer(auto_error=False)


class ApiAuthError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def require_scope(*scopes: str):
    """Allow the request when auth is off, or when the key has any of
    `scopes`. `generate` does not imply `read` or `write` except where a
    route lists both (job and artifact reads accept either)."""
    if not scopes:
        raise RuntimeError("require_scope needs at least one scope")

    def dependency(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> dict[str, Any] | None:
        try:
            required = config.auth_required()
        except config.BindConfigError as exc:
            raise ApiAuthError(500, "misconfigured", str(exc)) from exc
        if not required:
            return None
        secret = credentials.credentials if credentials is not None else ""
        if credentials is None or credentials.scheme.lower() != "bearer" or not secret:
            raise ApiAuthError(401, "unauthorized", "missing or unknown API key")
        try:
            record = authenticate(secret)
        except ApiKeyStoreError as exc:
            raise ApiAuthError(500, "misconfigured", str(exc)) from exc
        if record is None:
            raise ApiAuthError(401, "unauthorized", "missing or unknown API key")
        held = set(record.get("scopes") or [])
        if not any(scope in held for scope in scopes):
            needed = " or ".join(scopes)
            raise ApiAuthError(403, "forbidden", f"API key does not include the {needed} scope")
        return record

    return dependency
