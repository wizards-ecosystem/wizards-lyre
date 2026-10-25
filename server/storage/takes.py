"""Takes, LoRA style packs, and their on-disk metadata.

Every take is immutable once written (SPEC.md sec 7.3): cover/repaint/extract
produce a *new* take with `parent_take_id` set. The one exception is
`update_take_annotations`, which patches only the user-authored
favorite/notes fields.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from server import config
from server.storage import jsonio
from server.storage.errors import LoraNotFound, TakeNotFound
from server.storage.locks import _project_lock
from server.storage.paths import (
    _jail,
    jailed_path,
    lora_dir,
    loras_dir,
    new_id,
    take_dir,
    takes_dir,
)
from server.storage.projects import load_project


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
    # SPEC.md sec 16.4: a take written before provenance existed is legacy.
    # Its unknown values stay unknown; nothing is backfilled from the plan.
    meta.setdefault("job_id", None)
    meta.setdefault("provenance", None)
    provenance = meta["provenance"]
    meta["provenance_complete"] = bool(isinstance(provenance, dict) and provenance.get("complete"))
    return meta


def list_takes(project_id: str) -> list[dict]:
    load_project(project_id)
    tdir = takes_dir(project_id)
    out: list[dict] = []
    if tdir.exists():
        for entry in sorted(tdir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                out.append(_normalize_take_meta(jsonio._read_json(meta_path)))
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
                out.append(jsonio._read_json(meta_path))
    out.sort(key=lambda t: t.get("created_at") or "", reverse=True)
    return out


def get_take(project_id: str, take_id: str) -> dict:
    path = take_dir(project_id, take_id) / "meta.json"
    if not path.exists():
        raise TakeNotFound(take_id)
    return _normalize_take_meta(jsonio._read_json(path))


def get_lora(project_id: str, lora_id: str) -> dict:
    """Mirrors `get_take` above -- `train_lora`'s meta.json is only ever
    written once training actually finished (successfully or not, see
    `server.jobs._run_train_lora_job`/`_error_lora_meta`), so a missing file
    here means either an unknown lora_id or one still mid-training (its
    directory exists via `allocate_lora_dir` but no meta.json yet)."""
    path = lora_dir(project_id, lora_id) / "meta.json"
    if not path.exists():
        raise LoraNotFound(lora_id)
    return jsonio._read_json(path)


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


def discard_unfinished_take(project_id: str, take_id: str) -> None:
    """Delete a take that must not be kept. Cooperative cancel allocates a
    directory before generation and then drops it (SPEC.md sec 15.5: write
    no take). If the project was already pointed at this id, clear that
    pointer too."""
    from server.storage.errors import ProjectNotFound
    from server.storage.projects import _update_project, load_project

    tdir = take_dir(project_id, take_id)
    if tdir.exists():
        shutil.rmtree(tdir)
    try:
        project = load_project(project_id)
    except ProjectNotFound:
        return
    if project.get("active_take_id") == take_id:

        def mutate(current: dict) -> None:
            current["active_take_id"] = None

        _update_project(project_id, mutate)


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
        jsonio._write_json(path, meta)


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
        jsonio._write_json(path, meta)


def write_take_lrc(project_id: str, take_id: str, lrc_text: str) -> None:
    """SPEC.md sec 7 `lyrics.lrc`; `worker.acestep_worker.run_job` only
    returns `lrc_text` (see its docstring) -- this is what actually persists
    it, same division of labor as `write_take_meta` above."""
    with _project_lock(project_id):
        load_project(project_id)
        path = take_dir(project_id, take_id) / "lyrics.lrc"
        jsonio._write_text(path, lrc_text)


# SPEC.md sec 16.5: a take is rendered into a private staging directory
# outside the project, then promoted into takes/ with one rename. Staging
# lives under projects/.staging/<project>/<take> -- the same filesystem as
# takes/, so the rename is atomic -- and outside projects/<project>/, so a
# render that outlives a project deletion can never recreate that project's
# directory. A staged take is never listed as a take.
STAGING_MARKER = ".lyre-staging.json"


def staging_dir(project_id: str, take_id: str) -> Path:
    return jailed_path(".staging", project_id, take_id)


def allocate_staged_take(project_id: str, owner: dict) -> tuple[str, Path]:
    """Allocate a take id and its staging directory. `owner` (job id and
    attempt token) is written beside the files so cleanup can tell whose
    work it is."""
    with _project_lock(project_id):
        load_project(project_id)
        take_id = new_id()
        sdir = staging_dir(project_id, take_id)
        sdir.mkdir(parents=True, exist_ok=False)
        jsonio._write_json(sdir / STAGING_MARKER, {**owner, "take_id": take_id})
        return take_id, sdir


def write_staged_meta(project_id: str, take_id: str, meta: dict) -> None:
    jsonio._write_json(staging_dir(project_id, take_id) / "meta.json", meta)


def write_staged_lrc(project_id: str, take_id: str, lrc_text: str) -> None:
    jsonio._write_text(staging_dir(project_id, take_id) / "lyrics.lrc", lrc_text)


def promote_staged_take(project_id: str, take_id: str) -> None:
    """Move a fully written staged take into takes/ in one rename. The
    caller holds the project lock and has checked the project exists."""
    source = staging_dir(project_id, take_id)
    if not (source / "meta.json").is_file():
        raise RuntimeError(f"staged take {take_id} has no meta.json; refusing to promote it")
    (source / STAGING_MARKER).unlink(missing_ok=True)
    target = take_dir(project_id, take_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, target)


def discard_staged_take(project_id: str, take_id: str) -> None:
    shutil.rmtree(staging_dir(project_id, take_id), ignore_errors=True)
    try:
        jailed_path(".staging", project_id).rmdir()
    except OSError:
        pass  # other staged takes remain, or it is already gone


def list_staged_takes() -> list[tuple[str, str, dict | None]]:
    """Every staged take as `(project_id, take_id, owner marker or None)`."""
    root = config.projects_dir() / ".staging"
    out: list[tuple[str, str, dict | None]] = []
    if not root.is_dir():
        return out
    for project in sorted(root.iterdir()):
        if not project.is_dir():
            continue
        for take in sorted(project.iterdir()):
            marker = take / STAGING_MARKER
            try:
                owner = jsonio._read_json(marker) if marker.is_file() else None
            except (OSError, ValueError):
                owner = None
            out.append((project.name, take.name, owner))
    return out


def write_error_take(project_id: str, take_id: str, meta: dict) -> None:
    """A failed generation is still recorded as a take with `error` set
    (SPEC.md sec 10 point 5), but holds no audio: only its meta.json."""
    with _project_lock(project_id):
        load_project(project_id)
        jsonio._write_json(take_dir(project_id, take_id) / "meta.json", meta)


def set_active_take_if_unset(project_id: str, take_id: str) -> bool:
    """Make `take_id` active only when the project has no playable active
    take (SPEC.md sec 16.5): a finished job never replaces the user's kept
    take. Returns whether it became active. Caller holds the project lock
    or accepts the race with an explicit set."""
    from server.storage.projects import _update_project

    became = False

    def mutate(project: dict) -> None:
        nonlocal became
        current = project.get("active_take_id")
        if current and (take_dir(project_id, current) / "meta.json").is_file():
            meta = jsonio._read_json(take_dir(project_id, current) / "meta.json")
            if meta.get("error") is None:
                return
        project["active_take_id"] = take_id
        became = True

    _update_project(project_id, mutate)
    return became
