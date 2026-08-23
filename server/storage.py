"""Project / plan / take persistence on disk. Enforces the projects/ path jail.

Data model matches SPEC.md sec 7. Audio and weights never live in git; JSON
files under projects/<id>/ are the source of truth.
"""

from __future__ import annotations

import io
import json
import math
import os
import posixpath
import re
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from server import config

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

VALID_DIT_PROFILES = {"iterate", "polish", "quality", "studio_ops"}


class PathJailError(ValueError):
    """A resolved filesystem path escaped its allowed root."""


class ProjectNotFound(LookupError):
    pass


class TakeNotFound(LookupError):
    pass


class LoraNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _strip_win_extended_prefix(path: Path) -> Path:
    # Windows quirk: Path.resolve() goes through ntpath.realpath, which gets
    # an extended-length "\\?\"-prefixed result from GetFinalPathNameByHandle
    # and only strips the prefix after re-resolving the stripped form. If the
    # path is deleted in that gap (delete_project's rmtree racing a reader),
    # the prefixed form leaks out -- so two resolves of the same tree can
    # differ in prefix only, and relative_to() below would misread that as a
    # jail escape. The stripped form names the same file, so this changes no
    # containment outcome, only its string spelling. (\\?\UNC\ maps back to
    # the \\server\share spelling the same code path uses.)
    s = str(path)
    if s.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + s[len("\\\\?\\UNC\\"):])
    if s.startswith("\\\\?\\"):
        return Path(s[len("\\\\?\\"):])
    return path


def _jail(base: Path, target: Path) -> Path:
    base_r = _strip_win_extended_prefix(base.resolve())
    target_r = _strip_win_extended_prefix(target.resolve())
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


def uploads_dir(project_id: str) -> Path:
    return jailed_path(project_id, "uploads")


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
_PROJECT_LOCK_TIMEOUT_SEC = 24 * 60 * 60.0
_PROJECT_LOCK_POLL_SEC = 0.05
# A lock file older than this is assumed abandoned by a crashed process
# (e.g. the worker was killed mid-write) and is reclaimed rather than
# blocking every future update to this project forever.
_PROJECT_LOCK_STALE_SEC = 30.0
_PROJECT_LOCK_STATE = threading.local()


@contextmanager
def _project_lock(project_id: str) -> Iterator[None]:
    # Keep lifecycle locks outside the project directory.  A lock inside
    # projects/<id> disappears during rmtree, allowing a waiting writer to
    # create a fresh lock (and project directory) while deletion still owns
    # the old, unlinked lock file.
    lock_path = jailed_path(".project-locks", f"{project_id}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = getattr(_PROJECT_LOCK_STATE, "held", {})
    if held.get(project_id, 0):
        held[project_id] += 1
        try:
            yield
        finally:
            held[project_id] -= 1
        return
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
        held[project_id] = 1
        _PROJECT_LOCK_STATE.held = held
        stop_refresh = threading.Event()

        def _refresh_lock() -> None:
            while not stop_refresh.wait(_PROJECT_LOCK_STALE_SEC / 3):
                try:
                    lock_path.touch()
                except OSError:
                    return

        refresh_thread = threading.Thread(target=_refresh_lock, daemon=True)
        refresh_thread.start()
        yield
    finally:
        stop_refresh.set()
        refresh_thread.join()
        del held[project_id]
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


# Public lifecycle guard for operations (worker execution and streamed
# uploads) that perform several filesystem writes over an extended period.
# Deletion takes this same cross-process lock and therefore waits until the
# complete operation, not merely one metadata write, has finished.
project_lifecycle_lock = _project_lock


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


def _normalize_plan(plan: dict) -> dict:
    """Fill in plan fields added after some plan.json files were already
    written to disk (SPEC.md sec 9.2 `caption_rewrite`), same pattern as
    `_normalize_take_meta` for takes: a plan.json saved before this field
    existed has no key at all, and must keep behaving exactly like it did
    then -- worker.acestep_worker.run_job (and the frontend's Plan type /
    checkbox) would otherwise have to special-case a missing key at every
    read site instead of once here."""
    plan.setdefault("caption_rewrite", False)
    return plan


def _read_plan_or_default(project_id: str) -> dict:
    path = plan_json_path(project_id)
    if not path.exists():
        return default_plan()
    return _normalize_plan(_read_json(path))


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
        # SPEC.md sec 9.2/7.2: Custom-mode checkbox controlling whether
        # ACE-Step's LM ("thinking") is allowed to rewrite the user's caption.
        # New plans allow rewriting until the user disables it, as required
        # by SPEC.md sec 7.2. _normalize_plan separately keeps legacy plans
        # without this field on their historical False behavior.
        "caption_rewrite": True,
    }


# plan.json field types per SPEC.md sec 7.2. `bool` is a subclass of `int` in
# Python, so every numeric check must reject it explicitly -- a JSON `true`
# would otherwise pass `isinstance(..., int)` and reach the worker as a
# number. `_plan_number` additionally rejects NaN/+-inf, which `json.loads`
# accepts but which would serialize back out as non-standard JSON tokens and
# confuse downstream consumers.
_PLAN_STRING_FIELDS = ("query", "caption", "lyrics", "vocal_language", "timesignature")
_PLAN_BOOL_FIELDS = ("instrumental", "caption_rewrite")
# The only keys a section dict may carry (SPEC.md sec 7.2 sections[]). Unknown
# keys inside a section are dropped for the same reason unknown top-level keys
# are -- they'd otherwise be persisted verbatim and flow to worker/SPA.
_PLAN_SECTION_KEYS = ("name", "start_sec", "end_sec", "lyrics")


def _plan_type_name(value: object) -> str:
    return type(value).__name__


def _plan_type_error(field: str, expected: str, value: object) -> ValueError:
    """ValueError naming the offending field (server/app.py maps it to 400)."""
    return ValueError(
        f"invalid plan field '{field}': expected {expected}, got {_plan_type_name(value)}"
    )


def _plan_string(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise _plan_type_error(field, "a string", value)
    return value


def _plan_bool(field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise _plan_type_error(field, "a boolean", value)
    return value


def _plan_number(field: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _plan_type_error(field, "a number", value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _plan_type_error(field, "a finite number", value)
    return value


def validate_plan(plan: object) -> dict:
    """Validate and normalize a client-submitted plan for `PUT /plan`
    (SPEC.md sec 7.2), returning a fresh dict safe to persist:

    - missing keys are filled from `default_plan()`, so a partial body can
      never produce a plan the worker (`.get()` reads) or the SPA (Plan type
      assumes exact shapes) has to special-case;
    - field types are enforced -- `query`/`caption`/`lyrics`/
      `vocal_language`/`timesignature` strings, `negative` a list of strings,
      `instrumental`/`caption_rewrite` booleans, `bpm` int-or-null,
      `keyscale` string-or-null, `duration_sec` a number, `sections` a list
      of dicts each with `name` (str), `start_sec`/`end_sec` (numbers) and an
      optional `lyrics` (str). Any invalid value raises `ValueError` (->
      HTTP 400 via server/app.py) naming the offending field;
    - unknown top-level keys (and unknown keys inside a section) are dropped
      rather than persisted.

    Only the PUT write path runs this (`save_plan`). Reads stay on the
    lenient `_normalize_plan`, so plan.json files written before this
    validation existed keep loading exactly as before.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"invalid plan: expected a JSON object, got {_plan_type_name(plan)}")

    defaults = default_plan()
    # Copy only known keys (drops unknown top-level keys), filling any that
    # are missing from default_plan().
    normalized: dict = {key: plan.get(key, defaults[key]) for key in defaults}

    for field in _PLAN_STRING_FIELDS:
        _plan_string(field, normalized[field])
    for field in _PLAN_BOOL_FIELDS:
        _plan_bool(field, normalized[field])

    negative = normalized["negative"]
    if not isinstance(negative, list):
        raise _plan_type_error("negative", "a list of strings", negative)
    for i, item in enumerate(negative):
        _plan_string(f"negative[{i}]", item)

    bpm = normalized["bpm"]
    if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, int)):
        raise _plan_type_error("bpm", "an integer or null", bpm)

    keyscale = normalized["keyscale"]
    if keyscale is not None and not isinstance(keyscale, str):
        raise _plan_type_error("keyscale", "a string or null", keyscale)

    _plan_number("duration_sec", normalized["duration_sec"])

    sections = normalized["sections"]
    if not isinstance(sections, list):
        raise _plan_type_error("sections", "a list", sections)
    cleaned_sections: list[dict] = []
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise _plan_type_error(f"sections[{i}]", "an object", section)
        cleaned = {key: section[key] for key in _PLAN_SECTION_KEYS if key in section}
        if "name" not in cleaned:
            raise ValueError(f"invalid plan field 'sections[{i}].name': missing required key")
        _plan_string(f"sections[{i}].name", cleaned["name"])
        for bound in ("start_sec", "end_sec"):
            if bound not in cleaned:
                raise ValueError(
                    f"invalid plan field 'sections[{i}].{bound}': missing required key"
                )
            _plan_number(f"sections[{i}].{bound}", cleaned[bound])
        if "lyrics" in cleaned:
            _plan_string(f"sections[{i}].lyrics", cleaned["lyrics"])
        cleaned_sections.append(cleaned)
    normalized["sections"] = cleaned_sections
    return normalized


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
        "favorite": False,
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
        return _read_json(path)
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


def save_plan(project_id: str, plan: dict) -> dict:
    """Replace plan.json outright (PUT /api/projects/{id}/plan). The body is
    validated and normalized first (`validate_plan`): well-formed plans pass
    through unchanged, missing keys are filled from `default_plan()`, unknown
    keys are dropped, and invalid types raise `ValueError` (-> HTTP 400 via
    server/app.py) before anything is written. Validation runs inside the
    locked read-modify-write so an unknown project_id still surfaces
    ProjectNotFound (-> 404) ahead of a malformed body. Locked and atomic --
    see the module-level note above `_project_lock`."""
    return _update_plan(project_id, lambda _current: validate_plan(plan))


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
        if patch.get("favorite") is not None:
            project["favorite"] = patch["favorite"]

    return _update_project(project_id, mutate)


def set_active_take(project_id: str, take_id: str) -> dict:
    def mutate(project: dict) -> None:
        project["active_take_id"] = take_id

    return _update_project(project_id, mutate)


def _normalize_take_meta(meta: dict) -> dict:
    """Fill in fields added after some takes were already written to disk
    (SPEC.md sec 12 Phase 6: favorite/notes). New takes always get these from
    worker.mock_worker/acestep_worker, but a take created before that change
    has neither key -- every read site funnels through here instead of each
    caller (and the frontend's Take type / textarea, which assume both
    always exist) having to special-case missing keys."""
    meta.setdefault("favorite", False)
    meta.setdefault("notes", "")
    # SPEC.md sec 4.4 "LoRA train / load": added after some takes were
    # already written -- a take created before this exists has no key at
    # all, same reasoning as favorite/notes above.
    meta.setdefault("lora_id", None)
    return meta


def list_takes(project_id: str) -> list[dict]:
    load_project(project_id)
    tdir = takes_dir(project_id)
    out: list[dict] = []
    if tdir.exists():
        for entry in sorted(tdir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                out.append(_normalize_take_meta(_read_json(meta_path)))
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
    return _normalize_take_meta(_read_json(path))


def get_lora(project_id: str, lora_id: str) -> dict:
    """Mirrors `get_take` above -- `train_lora`'s meta.json is only ever
    written once training actually finished (successfully or not, see
    `server.jobs._run_train_lora_job`/`_error_lora_meta`), so a missing file
    here means either an unknown lora_id or one still mid-training (its
    directory exists via `allocate_lora_dir` but no meta.json yet)."""
    path = lora_dir(project_id, lora_id) / "meta.json"
    if not path.exists():
        raise LoraNotFound(lora_id)
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


def allocate_take_dir(project_id: str) -> tuple[str, Path]:
    with _project_lock(project_id):
        load_project(project_id)
        take_id = new_id()
        tdir = take_dir(project_id, take_id)
        tdir.mkdir(parents=True, exist_ok=False)
        return take_id, tdir


def write_take_meta(project_id: str, take_id: str, meta: dict) -> None:
    with _project_lock(project_id):
        load_project(project_id)
        path = take_dir(project_id, take_id) / "meta.json"
        _write_json(path, meta)


def update_take_annotations(
    project_id: str, take_id: str, favorite: bool | None = None, notes: str | None = None
) -> dict:
    """Patch the user-annotation fields on a take's meta.json (SPEC.md sec
    12 Phase 6: 'free-text take notes'). `favorite`/`notes` are annotations
    layered on top of a take, not generation state -- unlike every other
    write site, which persists a *complete*, freshly-generated meta dict via
    `write_take_meta`, this only ever touches the field(s) actually passed
    in, so it can never clobber the take's immutable generation data
    (SPEC.md sec 7.3) with a stale copy.

    Runs the read-modify-write under the project's cross-process lock
    (reviewer-flagged: an unlocked read-modify-write here lets a favorite
    PATCH and a notes PATCH racing each other both read the same on-disk
    meta and then overwrite one another's write, silently losing whichever
    landed second) -- the same lock `_update_project`/`_update_plan` already
    use for every other metadata mutation under this project."""
    with _project_lock(project_id):
        meta = get_take(project_id, take_id)
        if favorite is not None:
            meta["favorite"] = favorite
        if notes is not None:
            meta["notes"] = notes
        write_take_meta(project_id, take_id, meta)
        return meta


def allocate_lora_dir(project_id: str) -> tuple[str, Path]:
    with _project_lock(project_id):
        load_project(project_id)
        lora_id = new_id()
        ldir = lora_dir(project_id, lora_id)
        ldir.mkdir(parents=True, exist_ok=False)
        return lora_id, ldir


def write_lora_meta(project_id: str, lora_id: str, meta: dict) -> None:
    with _project_lock(project_id):
        load_project(project_id)
        path = lora_dir(project_id, lora_id) / "meta.json"
        _write_json(path, meta)


def write_take_lrc(project_id: str, take_id: str, lrc_text: str) -> None:
    """SPEC.md sec 7 `lyrics.lrc`; `worker.acestep_worker.run_job` only
    returns `lrc_text` (see its docstring) -- this is what actually persists
    it, same division of labor as `write_take_meta` above."""
    with _project_lock(project_id):
        load_project(project_id)
        path = take_dir(project_id, take_id) / "lyrics.lrc"
        _write_text(path, lrc_text)


def _sanitize_component(value: str, fallback: str) -> str:
    """Strip anything that isn't alnum/space/hyphen/underscore from a
    user-controlled string before using it as a filesystem path component --
    either a Content-Disposition filename (project title) or a member name
    inside the export archive (take `track_name`). This removes path
    separators (`/`, `\\`), `.`/`..` segments (dots aren't in the allowed
    set at all), control characters, and platform-special characters
    (`:`, `*`, `?`, `"`, `<`, `>`, `|`) alike, so a track name like
    `../../evil` or `a/..\\b` can't turn into a path-traversing zip entry
    (zip-slip risk, reviewer-flagged; SPEC.md sec 12 Phase 5)."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", value).strip()
    return cleaned or fallback


def sanitize_filename(name: str) -> str:
    return _sanitize_component(name, fallback="project")


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
