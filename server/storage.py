"""Project / plan / take persistence on disk. Enforces the projects/ path jail.

Data model matches SPEC.md sec 7. Audio and weights never live in git; JSON
files under projects/<id>/ are the source of truth.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from server import config

# SPEC.md sec 12 Phase 5 / sec 9.2: task_types produced by extract/lego/
# complete jobs are the closest thing this codebase has to the "stems"
# mentioned in the sec 6/7 data-model diagram -- no separate stems/
# directory is ever written (see storage.take_dir / write_take_meta), so
# "optional stems" in an export means "optionally include these takes".
STEM_TASK_TYPES = {"extract", "lego", "complete"}

VALID_DIT_PROFILES = {"iterate", "polish", "quality", "studio_ops"}


class PathJailError(ValueError):
    """A resolved filesystem path escaped its allowed root."""


class ProjectNotFound(LookupError):
    pass


class TakeNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _jail(base: Path, target: Path) -> Path:
    base_r = base.resolve()
    target_r = target.resolve()
    try:
        target_r.relative_to(base_r)
    except ValueError as exc:
        raise PathJailError(f"path '{target_r}' escapes jail root '{base_r}'") from exc
    return target_r


def jailed_path(*parts: str) -> Path:
    """Resolve a path under projects_dir(), rejecting any escape (.., symlink, etc.)."""
    base = config.projects_dir()
    target = base.joinpath(*parts)
    return _jail(base, target)


def jailed_output_path(*parts: str) -> Path:
    """Resolve a path under output_dir(), rejecting any escape. SPEC.md sec
    8.1/11: generated audio may only be written under projects/ or output/
    -- never a bare OS temp directory (e.g. scripts/smoke-gpu.py)."""
    base = config.output_dir()
    target = base.joinpath(*parts)
    return _jail(base, target)


def project_dir(project_id: str) -> Path:
    return jailed_path(project_id)


def takes_dir(project_id: str) -> Path:
    return jailed_path(project_id, "takes")


def take_dir(project_id: str, take_id: str) -> Path:
    return jailed_path(project_id, "takes", take_id)


def loras_dir(project_id: str) -> Path:
    return jailed_path(project_id, "loras")


def lora_dir(project_id: str, lora_id: str) -> Path:
    return jailed_path(project_id, "loras", lora_id)


def project_json_path(project_id: str) -> Path:
    return jailed_path(project_id, "project.json")


def plan_json_path(project_id: str) -> Path:
    return jailed_path(project_id, "plan.json")


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


# project.json and plan.json are both read-modify-written by the HTTP
# server (save_plan, patch_project) and by the worker process (merging a
# simple-mode plan_patch, and set_active_take, after a generation
# completes) -- see SPEC.md sec 5/10. Without serialization, one process
# can read a stale copy and overwrite the other's concurrent change (e.g. a
# plan PUT landing between the worker's read and write of a plan_patch
# merge silently disappears; a plan save racing generation completion can
# clobber the newly assigned active_take_id). One lock per project_id
# guards *all* metadata mutations for that project, project.json and
# plan.json alike -- they're small, infrequent writes, so sharing a single
# lock is simpler than two independent ones and still lets different
# projects proceed fully in parallel. This is a plain lock-file mutex
# (atomic exclusive create, `os.O_CREAT | os.O_EXCL`) rather than an
# in-process lock, since it must work *across* the server/worker process
# boundary, not just across threads within one of them.
_PROJECT_LOCK_TIMEOUT_SEC = 10.0
_PROJECT_LOCK_POLL_SEC = 0.05
# A lock file older than this is assumed abandoned by a crashed process
# (e.g. the worker was killed mid-write) and is reclaimed rather than
# blocking every future update to this project forever.
_PROJECT_LOCK_STALE_SEC = 30.0


@contextmanager
def _project_lock(project_id: str) -> Iterator[None]:
    lock_path = jailed_path(project_id, ".project.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _PROJECT_LOCK_TIMEOUT_SEC
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # Someone else holds the lock. On Windows, a concurrent
            # O_CREAT|O_EXCL attempt against a file another handle
            # currently has open can raise PermissionError instead of
            # FileExistsError (a sharing-violation quirk, not a real
            # permissions problem) -- treat both as "still locked".
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between our open() and stat(); retry immediately
            except PermissionError:
                time.sleep(_PROJECT_LOCK_POLL_SEC)  # transient Windows sharing violation
                continue
            if age > _PROJECT_LOCK_STALE_SEC:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for project lock: {project_id}")
            time.sleep(_PROJECT_LOCK_POLL_SEC)
    try:
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _update_project(project_id: str, mutate: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write project.json: load it, apply `mutate`
    in place, bump `updated_at`, and write it back, all under the
    cross-process lock so a concurrent mutation from the other process
    can't be lost."""
    with _project_lock(project_id):
        project = load_project(project_id)
        mutate(project)
        project["updated_at"] = _now()
        _write_json(project_json_path(project_id), project)
        return project


def _read_plan_or_default(project_id: str) -> dict:
    path = plan_json_path(project_id)
    if not path.exists():
        return default_plan()
    return _read_json(path)


def _update_plan(project_id: str, mutate: Callable[[dict], dict]) -> dict:
    """Atomically read-modify-write plan.json: load the project (raises if
    missing), load the *current* plan.json, apply `mutate` to get the new
    plan, write it back, and bump project.json's `updated_at` -- all in one
    locked critical section. `save_plan` (a PUT /plan) and
    `merge_plan_patch` (the worker merging a simple-mode plan_patch after a
    job finishes) both go through here, so the two can never interleave and
    silently drop one side's change (SPEC.md sec 5/7.2)."""
    with _project_lock(project_id):
        project = load_project(project_id)
        current_plan = _read_plan_or_default(project_id)
        new_plan = mutate(current_plan)
        _write_json(plan_json_path(project_id), new_plan)
        project["updated_at"] = _now()
        _write_json(project_json_path(project_id), project)
        return new_plan


def default_plan() -> dict:
    return {
        "query": "",
        "caption": "",
        "negative": [],
        "lyrics": "",
        "instrumental": False,
        "vocal_language": "en",
        "bpm": None,
        "keyscale": None,
        "timesignature": "4/4",
        "duration_sec": 120,
        "sections": [],
    }


def create_project(
    title: str | None = None,
    query: str | None = None,
    dit_profile: str = "iterate",
) -> dict:
    project_id = new_id()
    now = _now()
    project = {
        "id": project_id,
        "title": title or "Untitled",
        "created_at": now,
        "updated_at": now,
        "dit_profile": dit_profile,
        "lm_model": "acestep-5Hz-lm-1.7B",
        "active_take_id": None,
    }
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "takes").mkdir(parents=True, exist_ok=True)
    _write_json(project_json_path(project_id), project)

    plan = default_plan()
    if query:
        plan["query"] = query
    _write_json(plan_json_path(project_id), plan)
    return project


def list_projects() -> list[dict]:
    base = config.projects_dir()
    out: list[dict] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pj = entry / "project.json"
        if not pj.exists():
            continue
        try:
            data = _read_json(pj)
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at"),
                "active_take_id": data.get("active_take_id"),
            }
        )
    out.sort(key=lambda p: p.get("updated_at") or "", reverse=True)
    return out


def load_project(project_id: str) -> dict:
    path = project_json_path(project_id)
    if not path.exists():
        raise ProjectNotFound(project_id)
    return _read_json(path)


def load_plan(project_id: str) -> dict:
    load_project(project_id)
    return _read_plan_or_default(project_id)


def save_plan(project_id: str, plan: dict) -> dict:
    """Replace plan.json outright (PUT /api/projects/{id}/plan). Locked and
    atomic -- see the module-level note above `_project_lock`."""
    return _update_plan(project_id, lambda _current: plan)


def merge_plan_patch(project_id: str, patch: dict) -> dict:
    """Merge `patch` onto whatever plan is current on disk, atomically: the
    read of "current" and the write both happen inside the same lock, so a
    plan PUT landing between the read and the write can never be silently
    clobbered (SPEC.md sec 7.2). Used by `server.jobs` to apply a
    simple-mode `plan_patch` after a job finishes."""
    return _update_plan(project_id, lambda current: {**current, **patch})


def touch_project(project_id: str) -> dict:
    return _update_project(project_id, lambda project: None)


def patch_project(project_id: str, patch: dict) -> dict:
    def mutate(project: dict) -> None:
        if patch.get("title") is not None:
            project["title"] = patch["title"]
        if patch.get("dit_profile") is not None:
            if patch["dit_profile"] not in VALID_DIT_PROFILES:
                raise ValueError(f"invalid dit_profile: {patch['dit_profile']}")
            project["dit_profile"] = patch["dit_profile"]

    return _update_project(project_id, mutate)


def set_active_take(project_id: str, take_id: str) -> dict:
    def mutate(project: dict) -> None:
        project["active_take_id"] = take_id

    return _update_project(project_id, mutate)


def list_takes(project_id: str) -> list[dict]:
    load_project(project_id)
    tdir = takes_dir(project_id)
    out: list[dict] = []
    if tdir.exists():
        for entry in sorted(tdir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                out.append(_read_json(meta_path))
    out.sort(key=lambda t: t.get("created_at") or "", reverse=True)
    return out


def list_loras(project_id: str) -> list[dict]:
    load_project(project_id)
    ldir = loras_dir(project_id)
    out: list[dict] = []
    if ldir.exists():
        for entry in sorted(ldir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                out.append(_read_json(meta_path))
    out.sort(key=lambda t: t.get("created_at") or "", reverse=True)
    return out


def get_take(project_id: str, take_id: str) -> dict:
    path = take_dir(project_id, take_id) / "meta.json"
    if not path.exists():
        raise TakeNotFound(take_id)
    return _read_json(path)


def take_audio_path(project_id: str, take_id: str) -> Path:
    tdir = take_dir(project_id, take_id)
    for name in ("mix.wav", "mix.mp3"):
        candidate = tdir / name
        if candidate.exists():
            return _jail(config.projects_dir(), candidate)
    raise TakeNotFound(take_id)


def take_lrc_path(project_id: str, take_id: str) -> Path:
    """SPEC.md sec 7: `lyrics.lrc` is optional (phase 4, conditional on
    ACE-Step providing timestamps) -- raises `TakeNotFound` the same way
    `take_audio_path` does when there's nothing to serve, so the HTTP layer
    can 404 instead of the UI having to guess from a missing file."""
    candidate = take_dir(project_id, take_id) / "lyrics.lrc"
    if candidate.exists():
        return _jail(config.projects_dir(), candidate)
    raise TakeNotFound(take_id)


def resolve_upload_path(project_id: str, upload_path: str) -> Path:
    """Resolve a job's `upload_path` relative to its project dir, enforcing
    the same jail as every other write (SPEC.md sec 8.1 / sec 11)."""
    rel = Path(upload_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise PathJailError(f"upload_path must be a relative path under the project: {upload_path}")
    return jailed_path(project_id, *rel.parts)


def allocate_take_dir(project_id: str) -> tuple[str, Path]:
    load_project(project_id)
    take_id = new_id()
    tdir = take_dir(project_id, take_id)
    tdir.mkdir(parents=True, exist_ok=False)
    return take_id, tdir


def write_take_meta(project_id: str, take_id: str, meta: dict) -> None:
    path = take_dir(project_id, take_id) / "meta.json"
    _write_json(path, meta)


def allocate_lora_dir(project_id: str) -> tuple[str, Path]:
    load_project(project_id)
    lora_id = new_id()
    ldir = lora_dir(project_id, lora_id)
    ldir.mkdir(parents=True, exist_ok=False)
    return lora_id, ldir


def write_lora_meta(project_id: str, lora_id: str, meta: dict) -> None:
    path = lora_dir(project_id, lora_id) / "meta.json"
    _write_json(path, meta)


def write_take_lrc(project_id: str, take_id: str, lrc_text: str) -> None:
    """SPEC.md sec 7 `lyrics.lrc`; `worker.acestep_worker.run_job` only
    returns `lrc_text` (see its docstring) -- this is what actually persists
    it, same division of labor as `write_take_meta` above."""
    path = take_dir(project_id, take_id) / "lyrics.lrc"
    _write_text(path, lrc_text)


def sanitize_filename(name: str) -> str:
    """Strip anything that isn't alnum/space/hyphen/underscore from a
    user-controlled string (project title) before using it in a
    Content-Disposition filename (SPEC.md sec 12 Phase 5)."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip()
    return cleaned or "project"


def build_export_zip(project_id: str, include_stems: bool = True) -> bytes:
    """SPEC.md sec 12 Phase 5 / sec 9.2: zip project.json, plan.json, the
    active take's audio, and (when `include_stems`) every extract/lego/
    complete take's audio. Built entirely in memory (`io.BytesIO` +
    `zipfile.ZipFile`) -- nothing new touches disk, so there's no path-jail
    concern for the archive itself; the member audio is read via the
    existing jailed `take_audio_path`/`take_dir` helpers.
    """
    project = load_project(project_id)
    plan = load_plan(project_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(project, indent=2))
        zf.writestr("plan.json", json.dumps(plan, indent=2))

        active_take_id = project.get("active_take_id")
        if active_take_id:
            try:
                active_path = take_audio_path(project_id, active_take_id)
            except TakeNotFound:
                active_path = None
            if active_path is not None:
                zf.writestr(f"mix{active_path.suffix}", active_path.read_bytes())

        if include_stems:
            for take in list_takes(project_id):
                if take.get("task_type") not in STEM_TASK_TYPES:
                    continue
                try:
                    stem_path = take_audio_path(project_id, take["id"])
                except TakeNotFound:
                    continue
                track = take.get("track_name") or "track"
                arcname = f"{take['id']}-{take['task_type']}-{track}{stem_path.suffix}"
                zf.writestr(arcname, stem_path.read_bytes())

    return buf.getvalue()
