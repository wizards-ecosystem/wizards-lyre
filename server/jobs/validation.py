"""Enqueue-time validation: what a job body may ask for, and what it resolves to.

Everything here runs before a row reaches the queue, so an invalid request
fails as an HTTP 400 rather than as a job that errors later on the GPU.
"""

from __future__ import annotations

from typing import Any

from server import storage
from server.jobs.errors import JobError

# SPEC.md sec 12 (phase order): phase 1 is generate; phase 2 adds cover and
# repaint (now that the web UI has a waveform with drag-to-select region
# feeding repainting_start/repainting_end). Phase 3 adds extract, lego, and
# complete, now that the web UI has a base-model-swap confirmation/loading
# workflow (SPEC.md sec 4.3/9.2) reused by all three. PHASE_GATED_ACTIONS is
# the one-line lever (move an action here, out of VALID_ACTIONS) for any
# action that needs to land ahead of its UI.
#
# `train_lora` (SPEC.md sec 4.4 style pack) is live: worker/acestep_worker.py
# wraps ACE-Step's DatasetBuilder -> preprocess_to_tensors -> LoRATrainer
# pipeline, and worker/mock_worker.py implements the same call shape for
# tests. It doesn't fit STUDIO_OPS_ACTIONS/SOURCE_REQUIRED_ACTIONS'
# single-`source_take_id` shape -- it takes a `source_take_ids` list instead
# (see _resolve_lora_sources) and routes to a dedicated worker entry point
# rather than the generate-shaped run_job path (see run_claimed_job).
VALID_ACTIONS = {"generate", "cover", "repaint", "extract", "lego", "complete", "train_lora"}
PHASE_GATED_ACTIONS: set[str] = set()
STUDIO_OPS_ACTIONS = {"extract", "lego", "complete"}
SOURCE_REQUIRED_ACTIONS = {"cover", "repaint", "extract", "lego", "complete"}

# SPEC.md sec 4.4 "LoRA train / load" -- the load half. A LoRA's weight
# deltas are only valid against the exact base checkpoint it was trained
# on, which worker/acestep_worker.py's LORA_BASE_DIT_PROFILE pins to
# "studio_ops" (turbo/xl-turbo are distilled few-step checkpoints the real
# trainer can't run a full diffusion training loop against -- see that
# module's LoRA docstring section). extract/lego/complete already force
# studio_ops for an unrelated reason (they're structural editing ops, not
# style-pack generation) and always did -- generate/cover/repaint are the
# actions a style-pack lora is actually for (SPEC.md's "Style pack" row),
# so _resolve_dit_profile allows studio_ops on exactly these three,
# specifically when a valid lora_id for the project is attached; an
# ordinary generate/cover/repaint with no lora attached still rejects
# studio_ops exactly as before.
LORA_ELIGIBLE_ACTIONS = {"generate", "cover", "repaint"}

# SPEC.md sec 4.4: "Style pack | LoRA train / load | 8+ songs".
MIN_LORA_SOURCE_TAKES = 8


def _resolve_dit_profile(
    action: str,
    dit_profile: str | None,
    project_dit_profile: str,
    lora_attached: bool = False,
) -> str:
    """`dit_profile` is the job body's explicit override, if any;
    `project_dit_profile` is project.json's persisted default (PATCH
    /api/projects/{id}) -- an omitted job-level profile must fall back to
    that, not silently to 'iterate', or a project switched to e.g. 'polish'
    keeps generating with 'iterate' the moment a client omits the field
    (reviewer-flagged: the included frontend always omits it).

    `lora_attached` is True when the job body carries a validated `lora_id`
    for this project (see `_resolve_lora`) -- SPEC.md sec 4.4 "LoRA train /
    load". Loading a trained LoRA is only architecturally valid against the
    exact studio_ops base checkpoint it was trained on (see
    LORA_ELIGIBLE_ACTIONS above and worker/acestep_worker.py's
    LORA_BASE_DIT_PROFILE), so a lora-attached generate/cover/repaint is
    coerced to studio_ops the same way extract/lego/complete always are --
    and, symmetrically, an explicit non-studio_ops profile on a lora-attached
    job is rejected as a conflict instead of silently ignoring the lora."""
    if dit_profile is not None and dit_profile not in storage.VALID_DIT_PROFILES:
        raise JobError(f"invalid dit_profile: {dit_profile}")
    if action in STUDIO_OPS_ACTIONS:
        # SPEC.md sec 8.1: reject extract/lego/complete unless dit_profile is
        # studio_ops. An unset profile is coerced; an explicit mismatch is rejected.
        if dit_profile is None:
            return "studio_ops"
        if dit_profile != "studio_ops":
            raise JobError(
                f"action '{action}' requires dit_profile='studio_ops' (got '{dit_profile}')"
            )
        return "studio_ops"
    if lora_attached and action in LORA_ELIGIBLE_ACTIONS:
        if dit_profile is None:
            return "studio_ops"
        if dit_profile != "studio_ops":
            raise JobError(
                f"action '{action}' with a lora_id attached requires dit_profile='studio_ops' "
                f"(got '{dit_profile}') -- a LoRA's weights are only valid against the "
                "studio_ops base checkpoint it was trained on (SPEC.md sec 4.4)"
            )
        return "studio_ops"
    # studio_ops is reserved for extract/lego/complete, or generate/cover/
    # repaint with a valid lora_id attached (SPEC.md sec 8.1/4.4) -- reject
    # it here for every other case instead of loading the base model for
    # ordinary generation, whether it came from an explicit override or
    # (reviewer-flagged) a project's persisted default.
    resolved = dit_profile or project_dit_profile
    if resolved == "studio_ops":
        raise JobError(
            f"action '{action}' cannot use dit_profile='studio_ops' -- that profile is "
            "reserved for extract/lego/complete, or generate/cover/repaint with a valid "
            "lora_id attached"
        )
    return resolved


def _resolve_source_audio(project_id: str, action: str, body: dict[str, Any]) -> str | None:
    """Resolve cover/repaint/extract/lego/complete's source to a real, jailed
    filesystem path (SPEC.md sec 8.1 / sec 11). Called both at enqueue time
    (fail fast) and again when the job runs (payload_json only stores the
    client's identifiers, not the resolved path)."""
    if action not in SOURCE_REQUIRED_ACTIONS:
        return None

    source_take_id = body.get("source_take_id")
    upload_path = body.get("upload_path")
    if not source_take_id and not upload_path:
        raise JobError(f"action '{action}' requires source_take_id or upload_path")

    if source_take_id:
        try:
            path = storage.take_audio_path(project_id, source_take_id)
        except storage.TakeNotFound as exc:
            raise JobError(f"source_take_id not found: {source_take_id}") from exc
        return str(path)

    path = storage.resolve_upload_path(project_id, upload_path)
    if not path.exists():
        raise JobError(f"upload_path not found: {upload_path}")
    return str(path)


def _distinct_lora_source_ids(body: dict[str, Any]) -> list[str]:
    """Order-preserving de-duplication of `train_lora`'s `source_take_ids`,
    shared by `_resolve_lora_sources` (validation/path resolution) and
    `_error_lora_meta` below (reviewer-flagged: error metadata was computing
    `source_take_count` from the raw submitted list while success metadata
    used the deduplicated paths -- an accepted request with 8+ distinct ids
    plus extra duplicates reported inconsistent counts depending on whether
    training succeeded or failed). Single source of truth for what "the
    source songs" means for this job."""
    seen: set[str] = set()
    distinct_ids: list[str] = []
    for take_id in body.get("source_take_ids") or []:
        if take_id not in seen:
            seen.add(take_id)
            distinct_ids.append(take_id)
    return distinct_ids


def _resolve_lora_sources(project_id: str, body: dict[str, Any]) -> list[str]:
    """Resolve `train_lora`'s `source_take_ids` to real, jailed filesystem
    paths (SPEC.md sec 4.4 / sec 11), sibling to `_resolve_source_audio`
    above -- that helper resolves a single `source_take_id`, while
    `train_lora` instead takes a list and requires SPEC's '8+ songs' floor.
    Deduplicates first (reviewer-flagged: repeating one take_id N times must
    not satisfy the floor) so both the count check and the paths actually
    handed to the worker reflect distinct songs, not raw list length. Called
    both at enqueue time (fail fast, before any job row exists) and again
    when the job runs (payload_json only stores the client's identifiers,
    not the resolved paths)."""
    distinct_ids = _distinct_lora_source_ids(body)
    if len(distinct_ids) < MIN_LORA_SOURCE_TAKES:
        source_take_ids = body.get("source_take_ids") or []
        raise JobError(
            f"action 'train_lora' requires at least {MIN_LORA_SOURCE_TAKES} distinct "
            f"source_take_ids (got {len(distinct_ids)} distinct "
            f"of {len(source_take_ids)} submitted)"
        )
    paths: list[str] = []
    for take_id in distinct_ids:
        try:
            paths.append(str(storage.take_audio_path(project_id, take_id)))
        except storage.TakeNotFound as exc:
            raise JobError(f"source_take_ids entry not found: {take_id}") from exc
    return paths


def _resolve_track_name(action: str, body: dict[str, Any]) -> str | None:
    """extract/lego/complete route their target track through `track_name`,
    which the worker forwards onto ACE-Step's task-specific `instruction`
    field (SPEC.md sec 4.4) -- missing, non-string, or blank input would
    otherwise reach ACE-Step as a meaningless instruction. The web UI
    disables its Extract button until a track name is typed, but that's a
    client-side convenience only; a request posted straight to the HTTP API
    must be rejected the same way. Returns the trimmed name for actions that
    require one, else None."""
    if action not in STUDIO_OPS_ACTIONS:
        return None
    track_name = body.get("track_name")
    if not isinstance(track_name, str) or not track_name.strip():
        raise JobError(f"action '{action}' requires a non-empty track_name")
    return track_name.strip()


def _resolve_lora(project_id: str, lora_id: str) -> dict:
    """Resolve and validate `lora_id` (SPEC.md sec 4.4 "LoRA train / load"),
    sibling to `_resolve_source_audio`/`_resolve_track_name` above. Called
    both at enqueue time (fail fast) and again when the job runs
    (payload_json only stores the client's lora_id, not the resolved
    adapter path -- see `enqueue_job`'s `lora_adapter_path` payload field).

    A lora's meta.json is only ever written once training actually finished
    (`_run_train_lora_job` writes either the real success meta
    `worker.acestep_worker.train_lora`/`worker.mock_worker.train_lora`
    return, or `_error_lora_meta` on failure -- see `storage.get_lora`) --
    so there is no "still training" state to special-case here: a lora_id
    either doesn't exist yet (LoraNotFound), finished with an error (a
    non-null `error`), or finished successfully (`error` is None and
    `status` is a truthy value the training pipeline actually reported).
    ACE-Step's own training generator reports free-form progress strings as
    `status` (e.g. "epoch 2/10", see worker/acestep_worker.py's LoRA
    docstring section) rather than a fixed "completed" sentinel, so
    "successful" is checked structurally -- present and truthy, not equal to
    a specific literal -- instead of guessing at upstream's exact wording.
    """
    try:
        lora = storage.get_lora(project_id, lora_id)
    except storage.LoraNotFound as exc:
        raise JobError(f"lora_id not found: {lora_id}") from exc
    if lora.get("error"):
        raise JobError(f"lora_id '{lora_id}' failed training: {lora['error']}")
    if not lora.get("status"):
        raise JobError(f"lora_id '{lora_id}' has not finished training successfully")
    return lora
