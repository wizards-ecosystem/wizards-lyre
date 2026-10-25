"""Project export: a delivery ZIP with a versioned manifest (SPEC.md sec 16.6).

The ZIP carries the selected take's full-resolution mix (the active take
unless one is named), its lyrics.lrc when ACE-Step produced one, optional
extract/lego stems, project.json and plan.json, and `manifest.json`
(`lyre.export-manifest/1`) describing every member and every optional file
that is not there. It is a delivery bundle, not a backup or an import
format: docs/COMPLETION_MIGRATION.md describes a consistent backup.

Member audio is read through the jailed take helpers, every member name is
sanitized and then re-checked for zip-slip, and every member's size and
sha256 in the manifest are computed from the bytes actually written.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from server.generation import artifacts
from server.storage.errors import TakeNotFound
from server.storage.paths import _sanitize_component, take_dir
from server.storage.projects import load_plan, load_project
from server.storage.takes import get_take, list_takes, take_audio_path

MANIFEST_SCHEMA = "lyre.export-manifest/1"

# SPEC.md sec 7's data-model diagram labels the stems/ directory "extract /
# lego outputs" specifically -- `complete` ("Fill arrangement", sec 4.2)
# instead produces a full alternate mix, the same shape as the active take,
# not an isolated or added track. No separate stems/ directory is ever
# written on disk, so "optional stems" in an export means "optionally
# include these takes" -- but only the extract/lego ones, else `complete`
# outputs (and the active mix, when it happens to be a complete take) would
# get packaged as if they were stems (reviewer-flagged).
STEM_TASK_TYPES = {"extract", "lego"}

_NOTES = {
    "plan": (
        "project.json and plan.json are the project as it is now. The plan can have changed "
        "since the take was rendered; the take's own frozen brief and actual settings are in "
        "this manifest under take."
    ),
    "scope": (
        "A delivery bundle for one take, not a backup and not an import format. To back up "
        "Lyre, see docs/COMPLETION_MIGRATION.md."
    ),
}


class ExportError(ValueError):
    """The requested export cannot be built: an unknown or failed take, or a
    kept take whose audio is missing. Mapped onto HTTP 400 with `param`."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


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


def _selected_take(project_id: str, project: dict, take_id: str | None) -> dict | None:
    """The take to deliver: the named one, else the active one, else None
    (a project nothing has been kept in yet)."""
    if take_id is not None:
        try:
            take = get_take(project_id, take_id)
        except TakeNotFound as exc:
            raise ExportError(f"take not found: {take_id}", param="take_id") from exc
        if take.get("error") is not None:
            raise ExportError(f"take {take_id} failed and has no audio", param="take_id")
        return take
    active = project.get("active_take_id")
    if not active:
        return None
    try:
        take = get_take(project_id, active)
    except TakeNotFound as exc:
        raise ExportError(
            f"the active take {active} is missing; choose another take to export", param="take_id"
        ) from exc
    return take


def _audio_file(project_id: str, take_id: str) -> Path:
    try:
        path = take_audio_path(project_id, take_id)
    except TakeNotFound as exc:
        raise ExportError(f"take {take_id} has no audio file to export", param="take_id") from exc
    if path.stat().st_size == 0:
        raise ExportError(f"take {take_id}'s audio file is empty", param="take_id")
    return path


def _plan_members(project_id: str, take: dict | None, include_stems: bool) -> dict[str, Any]:
    """What goes into the archive, before anything is written."""
    files: list[tuple[str, str, Path, dict]] = []
    unavailable: list[dict[str, str]] = []
    if take is None:
        unavailable.append(
            {"role": "mix", "reason": "no take is kept yet: set an active take or pass take_id"}
        )
    else:
        mix = _audio_file(project_id, take["id"])
        files.append(("mix", _assert_safe_zip_member(f"mix{mix.suffix}"), mix, take))
        if mix.suffix == ".mp3":
            unavailable.append(
                {"role": "mix.wav", "reason": "only an MP3 was stored for this take"}
            )
        lrc = take_dir(project_id, take["id"]) / "lyrics.lrc"
        if lrc.is_file():
            files.append(("lrc", "lyrics.lrc", lrc, take))
        else:
            unavailable.append(
                {"role": "lrc", "reason": "ACE-Step produced no lyric timestamps for this take"}
            )
    if include_stems:
        stems = 0
        for stem_take in list_takes(project_id):
            if stem_take.get("task_type") not in STEM_TASK_TYPES or stem_take.get("error"):
                continue
            try:
                stem_path = take_audio_path(project_id, stem_take["id"])
            except TakeNotFound:
                unavailable.append(
                    {"role": "stem", "take_id": stem_take["id"], "reason": "audio file missing"}
                )
                continue
            track = _sanitize_component(stem_take.get("track_name") or "track", fallback="track")
            name = _assert_safe_zip_member(
                f"{stem_take['id']}-{stem_take['task_type']}-{track}{stem_path.suffix}"
            )
            files.append(("stem", name, stem_path, stem_take))
            stems += 1
        if stems == 0:
            unavailable.append({"role": "stem", "reason": "no extract or lego takes"})
    else:
        unavailable.append({"role": "stem", "reason": "not requested"})
    return {"files": files, "unavailable": unavailable}


def _take_block(take: dict | None) -> dict[str, Any] | None:
    if take is None:
        return None
    provenance = take.get("provenance") if isinstance(take.get("provenance"), dict) else None
    return {
        "id": take["id"],
        "task_type": take.get("task_type"),
        "created_at": take.get("created_at"),
        "parent_take_id": take.get("parent_take_id"),
        "job_id": take.get("job_id"),
        "seed": take.get("seed"),
        "provenance_complete": bool(take.get("provenance_complete")),
        # A take written before provenance existed says so; nothing here is
        # filled in from the current plan.
        "requested": provenance.get("requested") if provenance else None,
        "brief": provenance.get("brief") if provenance else None,
        "actual": provenance.get("actual") if provenance else None,
        "actual_origins": provenance.get("actual_origins") if provenance else None,
        "model": provenance.get("model") if provenance else None,
        "app": provenance.get("app") if provenance else None,
        "timestamps": provenance.get("timestamps") if provenance else None,
        "favorite": take.get("favorite", False),
        "notes": take.get("notes", ""),
    }


def _member(
    role: str, name: str, take_id: str | None, *, path: Path | None = None, data: bytes = b""
) -> dict:
    """A manifest entry. A file member is hashed in chunks from disk -- the
    archive member is that file byte for byte -- so a long WAV is never held
    in memory twice."""
    entry: dict[str, Any] = {
        "path": name,
        "role": role,
        "take_id": take_id,
        "bytes": path.stat().st_size if path is not None else len(data),
        "sha256": artifacts.sha256_file(path)
        if path is not None
        else "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    if path is not None and path.suffix in (".wav", ".mp3"):
        described = artifacts.describe_file(path)
        entry.update(
            format=described["format"],
            media_type=described["media_type"],
            sample_rate=described["sample_rate"],
            channels=described["channels"],
            frames=described["frames"],
            duration_sec=described["duration_sec"],
        )
    return entry


def _manifest(
    project: dict, take: dict | None, members: list[dict], unavailable: list[dict]
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "app": {"version": artifacts.app_version(), "revision": artifacts.app_revision()},
        "ace_step_revision": artifacts.ace_step_revision(),
        "project": {"id": project["id"], "title": project.get("title")},
        "take": _take_block(take),
        "files": members,
        "unavailable": unavailable,
        "notes": _NOTES,
    }


def build_export_manifest(
    project_id: str, include_stems: bool = True, take_id: str | None = None
) -> dict[str, Any]:
    """The manifest the ZIP would carry, without building the ZIP. Member
    hashes are computed from the files on disk (project.json and plan.json
    as they would be serialized)."""
    project = load_project(project_id)
    take = _selected_take(project_id, project, take_id)
    plan = _plan_members(project_id, take, include_stems)
    members = [
        _member(role, name, member_take["id"], path=path)
        for role, name, path, member_take in plan["files"]
    ]
    members.append(_member("project", "project.json", None, data=_json_bytes(project)))
    members.append(_member("plan", "plan.json", None, data=_json_bytes(load_plan(project_id))))
    return _manifest(project, take, members, plan["unavailable"])


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, indent=2).encode("utf-8")


def build_export_zip(
    project_id: str, include_stems: bool = True, take_id: str | None = None
) -> bytes:
    """The delivery ZIP. `mix<ext>` is always the selected take's stored
    full-resolution file at that fixed path -- consumers (a DAW, a script)
    need one predictable place for it, even when the same take also appears
    as a stem (reviewer-flagged). Audio members are added with
    `ZipFile.write()`, which streams each file in chunks rather than holding
    a second full copy in memory."""
    project = load_project(project_id)
    take = _selected_take(project_id, project, take_id)
    plan = _plan_members(project_id, take, include_stems)
    members: list[dict] = []

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for role, name, path, member_take in plan["files"]:
            zf.write(path, arcname=name)
            members.append(_member(role, name, member_take["id"], path=path))
        project_bytes = _json_bytes(project)
        plan_bytes = _json_bytes(load_plan(project_id))
        zf.writestr("project.json", project_bytes)
        zf.writestr("plan.json", plan_bytes)
        members.append(_member("project", "project.json", None, data=project_bytes))
        members.append(_member("plan", "plan.json", None, data=plan_bytes))
        manifest = _manifest(project, take, members, plan["unavailable"])
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue()
