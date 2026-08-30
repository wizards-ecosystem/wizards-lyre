"""Drag-dropped WAV/MP3 ingest as a cover/repaint source (SPEC.md sec 12 Phase 6).

The body is streamed to a temp file by `server.app.upload_audio` and only
published at its final path once it has been accepted whole, so an oversized
or failed upload never appears at a resolvable path.
"""

from __future__ import annotations

import os
from pathlib import Path

from server.storage.errors import PathJailError
from server.storage.paths import jailed_path, new_id, uploads_dir
from server.storage.projects import load_project


def resolve_upload_path(project_id: str, upload_path: str) -> Path:
    """Resolve a job's `upload_path` relative to its project dir, enforcing
    the same jail as every other write (SPEC.md sec 8.1 / sec 11)."""
    rel = Path(upload_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise PathJailError(f"upload_path must be a relative path under the project: {upload_path}")
    return jailed_path(project_id, *rel.parts)


# SPEC.md sec 12 Phase 6 scopes drag-drop ingest to "a local WAV/MP3"
# specifically -- not the broader set of audio extensions the repo's
# .gitignore treats as generated/ingested audio, which would silently widen
# this feature's contract (and pass formats to the worker it never promised
# to support) without SPEC.md saying so.
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
# Read/write granularity while streaming an upload to disk (see
# open_upload_destination / finalize_upload below) -- small enough that a
# single chunk is never a meaningful memory spike, large enough to not
# dominate upload time with per-chunk overhead.
UPLOAD_CHUNK_BYTES = 1024 * 1024


def open_upload_destination(project_id: str, filename: str) -> tuple[Path, Path]:
    """Validate the extension and allocate a jailed (temp_path, dest_path)
    pair under `projects/<id>/uploads/` for a drag-dropped cover/repaint
    source (SPEC.md sec 12 Phase 6), without touching the request body.

    The caller (server.app.upload_audio) streams the multipart body into
    temp_path in bounded chunks, enforcing MAX_UPLOAD_BYTES as it goes, and
    only calls `finalize_upload` -- an atomic rename -- once the whole body
    has been accepted. This keeps an oversized upload from ever being
    buffered fully in memory, and keeps a rejected/failed upload from ever
    appearing at a resolvable path.

    The client's original filename is discarded entirely in favor of a
    generated id + the (validated) extension, rather than sanitized and
    kept: that sidesteps path-traversal/weird-character concerns completely
    instead of trying to enumerate every dangerous character.
    """
    load_project(project_id)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"unsupported upload extension: {suffix or '(none)'}")

    udir = uploads_dir(project_id)
    udir.mkdir(parents=True, exist_ok=True)
    dest = udir / f"{new_id()}{suffix}"
    tmp = dest.with_name(f"{dest.name}.part")
    return tmp, dest


def finalize_upload(tmp_path: Path, dest_path: Path) -> str:
    """Atomically publish a fully-streamed, size-validated upload at its
    final path (`os.replace`, same single-rename approach as `_write_json` --
    see its docstring) and return the path relative to the project dir --
    exactly the string shape `JobBody.upload_path` / `resolve_upload_path`
    already expect."""
    os.replace(tmp_path, dest_path)
    return f"uploads/{dest_path.name}"
