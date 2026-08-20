"""Project / plan / take persistence on disk. Enforces the projects/ path jail.

Data model matches SPEC.md sec 7. Audio and weights never live in git; JSON
files under projects/<id>/ are the source of truth.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
    load_project(project_id)
    _write_json(plan_json_path(project_id), plan)
    touch_project(project_id)
    return plan


def touch_project(project_id: str) -> dict:
    project = load_project(project_id)
    project["updated_at"] = _now()
    _write_json(project_json_path(project_id), project)
    return project


def patch_project(project_id: str, patch: dict) -> dict:
    project = load_project(project_id)
    if patch.get("title") is not None:
        project["title"] = patch["title"]
    if patch.get("dit_profile") is not None:
        if patch["dit_profile"] not in VALID_DIT_PROFILES:
            raise ValueError(f"invalid dit_profile: {patch['dit_profile']}")
        project["dit_profile"] = patch["dit_profile"]
    project["updated_at"] = _now()
    _write_json(project_json_path(project_id), project)
    return project


def set_active_take(project_id: str, take_id: str) -> dict:
    project = load_project(project_id)
    project["active_take_id"] = take_id
    project["updated_at"] = _now()
    _write_json(project_json_path(project_id), project)
    return project


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
