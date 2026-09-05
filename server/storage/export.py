"""Project export as a zip (SPEC.md sec 12 Phase 5 / sec 9.2).

Built entirely in memory, so nothing new touches disk; member audio is read
through the jailed take helpers, and every member name is sanitized and then
re-checked for zip-slip.
"""

from __future__ import annotations

import io
import json
import posixpath
import zipfile

from server.storage.errors import TakeNotFound
from server.storage.paths import _sanitize_component
from server.storage.projects import load_plan, load_project
from server.storage.takes import list_takes, take_audio_path

# SPEC.md sec 7's data-model diagram labels the stems/ directory "extract /
# lego outputs" specifically (sec 12 Phase 5 line: "stems/  extract / lego
# outputs") -- `complete` ("Fill arrangement", sec 4.2) instead produces a
# full alternate mix, the same shape as the active take, not an isolated or
# added track. No separate stems/ directory is ever written on disk (see
# storage.take_dir / write_take_meta), so "optional stems" in an export
# means "optionally include these takes" -- but only the extract/lego ones,
# else `complete` outputs (and the active mix, when it happens to be a
# complete take) would get packaged as if they were stems (reviewer-flagged).
STEM_TASK_TYPES = {"extract", "lego"}


def _assert_safe_zip_member(name: str) -> str:
    """Defense in depth on top of `_sanitize_component`: normalize the fully
    composed archive member name and reject it outright if it's still
    absolute or escapes the archive root via a `..` segment. Every dynamic
    piece of a member name is sanitized before this runs, so this should
    never actually trigger -- it exists so a future component that forgets
    to sanitize its own input fails loudly instead of producing a
    zip-slip-able archive (reviewer-flagged)."""
    normalized = posixpath.normpath(name)
    if normalized in ("..", ".") or normalized.startswith("../") or normalized.startswith("/"):
        raise ValueError(f"unsafe zip member name: {name!r}")
    return name


def build_export_zip(project_id: str, include_stems: bool = True) -> bytes:
    """SPEC.md sec 12 Phase 5 / sec 9.2: zip project.json, plan.json, the
    active take's audio, and (when `include_stems`) every extract/lego
    take's audio. Built entirely in memory (`io.BytesIO` +
    `zipfile.ZipFile`) -- nothing new touches disk, so there's no path-jail
    concern for the archive itself; the member audio is read via the
    existing jailed `take_audio_path`/`take_dir` helpers.

    Audio members are added with `ZipFile.write()` against the source path
    rather than `writestr(name, path.read_bytes())`: `write()` streams the
    file into the archive in chunks internally, so a project with several
    long WAV stems never needs the *entire* raw file held as a second
    in-memory `bytes` object alongside the zip buffer while it's being
    compressed (reviewer-flagged: peak memory on large exports).
    """
    project = load_project(project_id)
    plan = load_plan(project_id)
    takes = list_takes(project_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project, indent=2))
        zf.writestr("plan.json", json.dumps(plan, indent=2))

        # The active take's audio always lands at the fixed "mix<ext>" path
        # -- consumers (a DAW, a script) need one predictable, unconditional
        # place to find it, regardless of whether that same take also
        # qualifies as a stem below. Deduplicating away this entry when the
        # active take happens to be an extract/lego result breaks that
        # contract (reviewer-flagged): the two arcnames serve different
        # semantic roles (active mix vs. named stem) even when their bytes
        # happen to be identical.
        active_take_id = project.get("active_take_id")
        if active_take_id:
            try:
                active_path = take_audio_path(project_id, active_take_id)
            except TakeNotFound:
                active_path = None
            if active_path is not None:
                arcname = _assert_safe_zip_member(f"mix{active_path.suffix}")
                zf.write(active_path, arcname=arcname)

        if include_stems:
            for take in takes:
                if take.get("task_type") not in STEM_TASK_TYPES:
                    continue
                try:
                    stem_path = take_audio_path(project_id, take["id"])
                except TakeNotFound:
                    continue
                track = _sanitize_component(take.get("track_name") or "track", fallback="track")
                arcname = _assert_safe_zip_member(
                    f"{take['id']}-{take['task_type']}-{track}{stem_path.suffix}"
                )
                zf.write(stem_path, arcname=arcname)

    return buf.getvalue()
