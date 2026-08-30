"""Atomic JSON and text writes.

Both the HTTP server and the worker process read and write these files
concurrently (SPEC.md sec 5/10), so every write lands via a temp file plus a
single `os.replace`.

Callers reach these through the module (`from server.storage import jsonio`
... `jsonio._write_json(...)`) rather than importing the functions by name,
so that a test patching `jsonio._write_json` affects every call site.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def _write_json(path: Path, data: dict) -> None:
    """Write atomically: serialize to a temp file in the same directory,
    then `os.replace` over the destination. A plain `write_text` can be
    observed mid-write by a concurrent reader (the worker loading a plan
    the HTTP server is saving, or vice versa) -- `os.replace` is a single
    filesystem-level rename, so readers only ever see the old or the new
    content, never a partial file (SPEC.md sec 5/10)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    """Same atomic temp-file-then-`os.replace` approach as `_write_json`,
    for plain-text files (SPEC.md sec 7: `lyrics.lrc` is plain text, not
    JSON, so `json.dumps` doesn't apply). Writes raw utf-8 bytes rather than
    `Path.write_text` so `\\n` in `text` (e.g. ACE-Step's own `lrc_text`
    line breaks) round-trips exactly -- `write_text`'s default text mode
    translates `\\n` to `os.linesep` on write, silently turning every LRC
    line into CRLF on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_bytes(text.encode("utf-8"))
    os.replace(tmp_path, path)
