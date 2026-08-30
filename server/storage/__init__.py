"""Project / plan / take persistence on disk. Enforces the projects/ path jail.

Data model matches SPEC.md sec 7. Audio and weights never live in git; JSON
files under projects/<id>/ are the source of truth.

This package is the single import surface: callers do `from server import
storage` and reach everything below as `storage.<name>`, exactly as they did
when this was one module. The split is by concern:

- `errors`    -- the exception types `server/app.py` maps onto HTTP statuses
- `paths`     -- path construction and the jail itself
- `jsonio`    -- atomic JSON/text writes
- `locks`     -- the cross-process per-project lock
- `plan`      -- plan.json defaults, normalization, and validation
- `projects`  -- project.json + plan.json lifecycle
- `takes`     -- takes, LoRA packs, and their metadata
- `uploads`   -- drag-dropped audio ingest
- `export`    -- the project export zip
"""

from __future__ import annotations

# Re-exported so `storage.config` keeps resolving, as it did when this
# package was a single module that imported config at the top.
from server import config
from server.storage.errors import (
    LoraNotFound,
    PathJailError,
    ProjectNotFound,
    TakeNotFound,
)
from server.storage.export import STEM_TASK_TYPES, build_export_zip
# The private IO/jail helpers are part of the historical surface: tests
# and scripts reach them as storage.<name>. Internal callers deliberately
# go through the defining module instead (see jsonio's docstring).
from server.storage.jsonio import _read_json, _write_json, _write_text
from server.storage.locks import project_lifecycle_lock
from server.storage.paths import (
    _jail,
    jailed_output_path,
    jailed_path,
    lora_dir,
    loras_dir,
    new_id,
    plan_json_path,
    project_dir,
    project_json_path,
    sanitize_filename,
    take_dir,
    takes_dir,
    uploads_dir,
)
from server.storage.plan import default_plan, validate_plan
from server.storage.projects import (
    VALID_DIT_PROFILES,
    create_project,
    delete_project,
    list_projects,
    load_plan,
    load_project,
    merge_plan_patch,
    patch_project,
    save_plan,
    set_active_take,
    touch_project,
)
from server.storage.takes import (
    allocate_lora_dir,
    allocate_take_dir,
    get_lora,
    get_take,
    list_loras,
    list_takes,
    take_audio_path,
    take_lrc_path,
    update_take_annotations,
    write_lora_meta,
    write_take_lrc,
    write_take_meta,
)
from server.storage.uploads import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    UPLOAD_CHUNK_BYTES,
    finalize_upload,
    open_upload_destination,
    resolve_upload_path,
)

__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "_jail",
    "_read_json",
    "_write_json",
    "_write_text",
    "config",
    "MAX_UPLOAD_BYTES",
    "STEM_TASK_TYPES",
    "UPLOAD_CHUNK_BYTES",
    "VALID_DIT_PROFILES",
    "LoraNotFound",
    "PathJailError",
    "ProjectNotFound",
    "TakeNotFound",
    "allocate_lora_dir",
    "allocate_take_dir",
    "build_export_zip",
    "create_project",
    "default_plan",
    "delete_project",
    "finalize_upload",
    "get_lora",
    "get_take",
    "jailed_output_path",
    "jailed_path",
    "list_loras",
    "list_projects",
    "list_takes",
    "load_plan",
    "load_project",
    "lora_dir",
    "loras_dir",
    "merge_plan_patch",
    "new_id",
    "open_upload_destination",
    "patch_project",
    "plan_json_path",
    "project_dir",
    "project_json_path",
    "project_lifecycle_lock",
    "resolve_upload_path",
    "sanitize_filename",
    "save_plan",
    "set_active_take",
    "take_audio_path",
    "take_dir",
    "take_lrc_path",
    "takes_dir",
    "touch_project",
    "update_take_annotations",
    "uploads_dir",
    "validate_plan",
    "write_lora_meta",
    "write_take_lrc",
    "write_take_meta",
]
