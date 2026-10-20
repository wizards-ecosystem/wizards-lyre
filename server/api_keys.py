"""Operator-minted API keys for the opt-in `/v1` surface (SPEC.md sec 15.2).

There is no HTTP endpoint that creates a key. `./scripts/lyre api-key`
prints the secret once. The file beside the jobs database stores the
sha256, the id, the label, the scopes, and `created_at`, mode `0600`.
Comparison is constant-time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from server import config

SCOPES = ("read", "generate", "write")
_DEFAULT_SCOPES = ("read", "generate", "write")


class ApiKeyStoreError(RuntimeError):
    """The key file cannot be read. Callers fail closed."""


def keys_path() -> Path:
    return config.db_path().with_name("api-keys.json")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def load_keys() -> list[dict[str, Any]]:
    path = keys_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiKeyStoreError(f"API key file cannot be read: {path}") from exc
    keys = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(keys, list):
        raise ApiKeyStoreError(f"API key file is corrupt: {path}")
    return keys


def has_any_key() -> bool:
    return len(load_keys()) > 0


def _write_keys(keys: list[dict[str, Any]]) -> None:
    path = keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"keys": keys}, indent=2).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _parse_scopes(raw: str) -> list[str]:
    scopes: list[str] = []
    for piece in raw.split(","):
        scope = piece.strip()
        if not scope:
            continue
        if scope not in SCOPES:
            raise ValueError(f"unknown scope: {scope} (expected {', '.join(SCOPES)})")
        if scope not in scopes:
            scopes.append(scope)
    if not scopes:
        raise ValueError("at least one scope is required")
    return scopes


def create_key(name: str, scopes: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """Store a new key and return `(public record, secret)`. The secret is
    not written to disk."""
    label = name.strip()
    if not label or any(char in label for char in "\r\n"):
        raise ValueError("name must be a single non-empty line")
    chosen = list(scopes) if scopes is not None else list(_DEFAULT_SCOPES)
    unknown = [scope for scope in chosen if scope not in SCOPES]
    if unknown:
        raise ValueError(f"unknown scope: {unknown[0]}")
    if not chosen:
        raise ValueError("at least one scope is required")
    secret = "lyre_" + secrets.token_urlsafe(32)
    record = {
        "id": secrets.token_hex(8),
        "name": label,
        "sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        "scopes": chosen,
        "created_at": _now(),
    }
    _write_keys([*load_keys(), record])
    public = {key: value for key, value in record.items() if key != "sha256"}
    return public, secret


def revoke_key(key_id: str) -> None:
    keys = load_keys()
    kept = [key for key in keys if key.get("id") != key_id]
    if len(kept) == len(keys):
        raise ValueError(f"no API key with id {key_id}")
    _write_keys(kept)


def authenticate(secret: str) -> dict[str, Any] | None:
    """Return the matching key record, or None. Every stored hash is
    compared, so a match is not faster than a miss."""
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    found: dict[str, Any] | None = None
    for key in load_keys():
        stored = key.get("sha256")
        if not isinstance(stored, str):
            continue
        if secrets.compare_digest(digest, stored):
            found = key
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyre api-key")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="mint a key and print its secret once")
    create.add_argument("--name", required=True, help="label stored with the key")
    create.add_argument(
        "--scopes",
        default=",".join(_DEFAULT_SCOPES),
        help="comma-separated scopes (default: read,generate,write)",
    )

    revoke = sub.add_parser("revoke", help="delete a key by id")
    revoke.add_argument("key_id")

    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            public, secret = create_key(args.name, _parse_scopes(args.scopes))
            print(f"id: {public['id']}")
            print(f"name: {public['name']}")
            print(f"scopes: {', '.join(public['scopes'])}")
            print(f"secret: {secret}")
            return 0
        revoke_key(args.key_id)
    except (ValueError, ApiKeyStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"revoked {args.key_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
