"""Project / plan / take persistence on disk. Enforces the projects/ path jail.

Data model matches SPEC.md sec 7. Audio and weights never live in git; JSON
files under projects/<id>/ are the source of truth.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from server import config

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


def project_json_path(project_id: str) -> Path:
    return jailed_path(project_id, "project.json")


def plan_json_path(project_id: str) -> Path:
    return jailed_path(project_id, "plan.json")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# project.json is read-modify-written by both the HTTP server (save_plan,
# patch_project) and the worker process (set_active_take, after a
# generation completes) -- see SPEC.md sec 5/10. Without serialization, one
# process can read a stale copy and overwrite the other's concurrent change
# (e.g. a plan save racing generation completion can clobber the newly
# assigned active_take_id, or vice versa). This is a plain lock-file mutex
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
    path = plan_json_path(project_id)
    if not path.exists():
        return default_plan()
    return _read_json(path)


def save_plan(project_id: str, plan: dict) -> dict:
    # Writing plan.json doesn't need the project.json lock, but bumping
    # updated_at does -- do it via _update_project (a no-op mutation) so
    # this can never race set_active_take/patch_project's read-modify-write
    # of project.json (SPEC.md sec 5/10).
    load_project(project_id)
    _write_json(plan_json_path(project_id), plan)
    _update_project(project_id, lambda project: None)
    return plan


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
