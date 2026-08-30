"""Path construction and the projects/ path jail (SPEC.md sec 8.1, sec 11).

Every filesystem path the rest of the package touches is built here, so the
jail is enforced in exactly one place: nothing may be written outside
`projects/` or `output/`.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from server import config
from server.storage.errors import PathJailError


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
        return Path("\\\\" + s[len("\\\\?\\UNC\\") :])
    if s.startswith("\\\\?\\"):
        return Path(s[len("\\\\?\\") :])
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
