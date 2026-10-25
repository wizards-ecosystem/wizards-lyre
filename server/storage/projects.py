"""project.json and plan.json lifecycle: create, list, load, patch, delete.

Every mutation is a read-modify-write inside the project's cross-process
lock, so a change from the HTTP server and one from the worker can never
clobber each other (SPEC.md sec 5).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable

from server import config
from server.generation import brief as creation
from server.storage import jsonio
from server.storage.errors import PlanRevisionConflict, ProjectNotFound
from server.storage.locks import _project_lock
from server.storage.paths import (
    _now,
    new_id,
    plan_json_path,
    project_dir,
    project_json_path,
)
from server.storage.plan import _read_plan_or_default, default_plan, validate_plan

VALID_DIT_PROFILES = {"iterate", "polish", "quality", "studio_ops"}


def _normalize_title(title: str | None) -> str:
    """Strip display padding and keep project titles visibly non-empty."""
    cleaned = (title or "").strip()
    return cleaned or "Untitled"


def _update_project(project_id: str, mutate: Callable[[dict], None]) -> dict:
    """Atomically read-modify-write project.json: load it, apply `mutate`
    in place, bump `updated_at`, and write it back, all under the
    cross-process lock so a concurrent mutation from the other process
    can't be lost."""
    with _project_lock(project_id):
        project = load_project(project_id)
        mutate(project)
        project["updated_at"] = _now()
        jsonio._write_json(project_json_path(project_id), project)
        return project


def _update_plan(
    project_id: str,
    mutate: Callable[[dict], dict],
    write_if: Callable[[], bool] | None = None,
) -> dict:
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
        if write_if is not None and not write_if():
            return new_plan
        jsonio._write_json(plan_json_path(project_id), new_plan)
        project["updated_at"] = _now()
        jsonio._write_json(project_json_path(project_id), project)
        return new_plan


def create_project(
    title: str | None = None,
    query: str | None = None,
    dit_profile: str = "iterate",
) -> dict:
    project_id = new_id()
    now = _now()
    project = {
        "id": project_id,
        "title": _normalize_title(title),
        "created_at": now,
        "updated_at": now,
        "dit_profile": dit_profile,
        "lm_model": "acestep-5Hz-lm-1.7B",
        "active_take_id": None,
        "favorite": False,
    }
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "takes").mkdir(parents=True, exist_ok=True)
    jsonio._write_json(project_json_path(project_id), project)

    plan = default_plan()
    if query:
        plan["query"] = query
    jsonio._write_json(plan_json_path(project_id), plan)
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
            data = jsonio._read_json(pj)
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at"),
                "active_take_id": data.get("active_take_id"),
                "favorite": data.get("favorite", False),
            }
        )
    out.sort(key=lambda p: p.get("updated_at") or "", reverse=True)
    return out


def load_project(project_id: str) -> dict:
    path = project_json_path(project_id)
    try:
        if not path.exists():
            raise ProjectNotFound(project_id)
        return jsonio._read_json(path)
    except FileNotFoundError as exc:
        # A concurrent delete_project can rmtree the directory in the gap
        # between the exists() check and the read -- same outcome as the
        # check itself failing.
        raise ProjectNotFound(project_id) from exc
    except PermissionError as exc:
        # On Windows, the same delete-project race above can surface as
        # PermissionError (WinError 5) instead of FileNotFoundError, for a
        # read that starts while the file is mid-delete (reviewer-flagged
        # flake under test_project_delete.py's concurrent enqueue test --
        # jobs.enqueue_job's unlocked load_project call races
        # delete_project's locked rmtree directly). But a PermissionError can
        # also be a genuine ACL/filesystem problem on a project that still
        # exists -- only normalize it to ProjectNotFound once the path has
        # actually disappeared (confirming the race), otherwise propagate the
        # real error instead of masking it as a 404.
        if path.exists():
            raise
        raise ProjectNotFound(project_id) from exc


def delete_project(project_id: str) -> None:
    """Remove `projects/<id>` entirely (SPEC.md sec 9.1 "Delete (confirm)").

    `project_dir()` resolves through `jailed_path`, so this can never be
    tricked (via a crafted/traversal id) into `rmtree`-ing anything outside
    `projects_dir()` -- same jail every other write in this module goes
    through. `load_project` first so an unknown id raises `ProjectNotFound`
    (-> 404) rather than `rmtree` silently no-oping on a path that never
    existed, matching every other lookup in this module.

    The rmtree itself retries briefly on PermissionError: on Windows a
    concurrent lock-free reader (e.g. jobs.enqueue_job's load_project) can
    still hold a file handle open for a moment, which surfaces as a sharing
    violation (WinError 32 / PermissionError) instead of the delete
    completing. Such handles drop almost immediately, so a short retry turns
    an avoidable failure into a successful deletion. The attempt count/total
    budget (~1.5s) is sized for many concurrent readers hammering the same
    project (reviewer-flagged flake under test_project_delete.py's
    concurrent-enqueue test: 4 threads x 50 enqueues can keep re-opening
    project.json for longer than a couple of retries cover).
    """
    with _project_lock(project_id):
        load_project(project_id)
        pdir = project_dir(project_id)
        attempts = 15
        for attempt in range(attempts):
            try:
                shutil.rmtree(pdir)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.1)


def load_plan(project_id: str) -> dict:
    load_project(project_id)
    return _read_plan_or_default(project_id)


def replace_plan(project_id: str, plan: dict) -> dict:
    """Write `plan` as plan.json without `validate_plan`.

    `POST /v1/songs` builds the complete plan itself, including its field
    origins and creation mode (SPEC.md sec 15.4/16.3), and stores a provided
    duration as given. The revision still advances, so a snapshot taken
    before this write is recognizably older. Unknown projects still raise
    `ProjectNotFound`."""

    def mutate(current: dict) -> dict:
        snapshot = dict(plan)
        snapshot["revision"] = int(current.get("revision") or 0) + 1
        snapshot.setdefault("brief", None)
        return snapshot

    return _update_plan(project_id, mutate)


def save_plan(project_id: str, plan: dict, expected_revision: int | None = None) -> dict:
    """Replace plan.json outright (PUT /api/projects/{id}/plan). The body is
    validated and normalized first (`validate_plan`): well-formed plans pass
    through unchanged, missing keys are filled from `default_plan()`, unknown
    keys are dropped, and invalid types raise `ValueError` (-> HTTP 400 via
    server/app.py) before anything is written. Validation runs inside the
    locked read-modify-write so an unknown project_id still surfaces
    ProjectNotFound (-> 404) ahead of a malformed body. Locked and atomic --
    see the module-level note above `_project_lock`.

    SPEC.md sec 16.3: `apply_write_semantics` derives field origins, the
    creation mode, and the next revision from what changed against the plan
    on disk. `expected_revision`, when given, makes the write conditional:
    a plan that moved on since the client read it raises
    `PlanRevisionConflict` (-> 409) and nothing is written."""

    def mutate(current: dict) -> dict:
        if expected_revision is not None and int(current.get("revision") or 0) != expected_revision:
            raise PlanRevisionConflict(
                f"plan revision is {current.get('revision', 0)}, not {expected_revision}: "
                "it changed since this edit started"
            )
        return creation.apply_write_semantics(current, validate_plan(plan), plan)

    return _update_plan(project_id, mutate)


def merge_plan_patch(
    project_id: str, patch: dict, expected_revision: int | None = None
) -> dict | None:
    """Merge `patch` onto whatever plan is current on disk, atomically: the
    read of "current" and the write both happen inside the same lock, so a
    plan PUT landing between the read and the write can never be silently
    clobbered (SPEC.md sec 7.2). Used by `server.jobs` to apply a planned
    render's brief after a job finishes.

    `field_origins` in the patch merges key by key. With
    `expected_revision`, a draft that changed since the job was submitted
    is left alone and this returns None: the newer text wins and the take
    keeps its own brief (SPEC.md sec 16.3)."""
    skipped = False

    def mutate(current: dict) -> dict:
        nonlocal skipped
        if expected_revision is not None and int(current.get("revision") or 0) != expected_revision:
            skipped = True
            return current
        merged = {**current, **patch}
        if isinstance(patch.get("field_origins"), dict):
            merged["field_origins"] = {**current.get("field_origins", {}), **patch["field_origins"]}
        merged["revision"] = int(current.get("revision") or 0) + 1
        return merged

    result = _update_plan(project_id, mutate, write_if=lambda: not skipped)
    return None if skipped else result


def patch_project(project_id: str, patch: dict) -> dict:
    def mutate(project: dict) -> None:
        if patch.get("title") is not None:
            project["title"] = _normalize_title(patch["title"])
        if patch.get("dit_profile") is not None:
            if patch["dit_profile"] not in VALID_DIT_PROFILES:
                raise ValueError(f"invalid dit_profile: {patch['dit_profile']}")
            project["dit_profile"] = patch["dit_profile"]
        if patch.get("favorite") is not None:
            project["favorite"] = patch["favorite"]

    return _update_project(project_id, mutate)


def set_active_take(project_id: str, take_id: str) -> dict:
    def mutate(project: dict) -> None:
        project["active_take_id"] = take_id

    return _update_project(project_id, mutate)
