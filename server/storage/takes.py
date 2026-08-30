"""Takes, LoRA style packs, and their on-disk metadata.

Every take is immutable once written (SPEC.md sec 7.3): cover/repaint/extract
produce a *new* take with `parent_take_id` set. The one exception is
`update_take_annotations`, which patches only the user-authored
favorite/notes fields.
"""

from __future__ import annotations

from pathlib import Path

from server import config
from server.storage import jsonio
from server.storage.errors import LoraNotFound, TakeNotFound
from server.storage.locks import _project_lock
from server.storage.paths import (
    _jail,
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
