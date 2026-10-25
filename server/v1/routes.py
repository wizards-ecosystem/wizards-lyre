"""`/v1` routes (SPEC.md sec 15.3).

Reads, the job object, cancel, and stable artifact GETs landed in the first
slice. This module adds the studio-loop writes: projects, plan, jobs,
uploads, annotations, active take, delete, style packs, and export. `/api`
keeps its own routes and its `{"detail": ...}` errors.

`Idempotency-Key` on POSTs is enforced by `IdempotencyMiddleware`, which
sees every `/v1` POST. The header parameter below only publishes it in
OpenAPI.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Header, Response, UploadFile
from fastapi.responses import FileResponse

from server import capabilities as capability_report
from server import creation, jobs, storage
from server.uploads import receive_audio_upload
from server.v1.auth import require_scope
from server.v1.models import (
    ActiveTake,
    Capabilities,
    Error,
    Health,
    Job,
    JobCreate,
    JobPage,
    Lora,
    Plan,
    PlanWrite,
    PresetRequest,
    PresetResult,
    Project,
    ProjectCreate,
    ProjectDetail,
    ProjectPage,
    ProjectPatch,
    SongCreate,
    Take,
    TakePage,
    TakePatch,
    UploadCreated,
)
from server.v1.present import job_object
from server.v1.songs import create_song as queue_song

router = APIRouter(prefix="/v1")


def _accepted_job(queued: dict[str, Any]) -> dict[str, Any]:
    """The job this call accepted, still `queued`.

    `read_job` can already see `running` when the worker claims the row
    before the response is built. The 202 body is the accepted job.
    `GET /v1/jobs/{id}` is the live status.
    """
    row = jobs.read_job(queued["id"])
    row["status"] = "queued"
    row["stage"] = None
    row["take_id"] = None
    row["error"] = None
    return job_object(row)


_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": Error, "description": "Invalid request"},
    401: {"model": Error, "description": "Missing or unknown API key"},
    403: {"model": Error, "description": "Known API key without the required scope"},
    404: {"model": Error, "description": "Unknown project, take, or job"},
}

_PLAN_ERRORS: dict[int | str, dict[str, Any]] = {
    **_ERRORS,
    409: {"model": Error, "description": "expected_revision no longer matches the plan"},
}

_POST_ERRORS: dict[int | str, dict[str, Any]] = {
    **_ERRORS,
    409: {
        "model": Error,
        "description": "Idempotency-Key reused with a different body",
    },
}

_PAGE_DEFAULT = 50
_PAGE_MAX = 100


def declare_idempotency_key(
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            description=(
                "Optional. The same key and the same raw body replay the original "
                "response for 24 hours. The same key and a different body is 409. "
                "A replay of a queued job is 200."
            ),
        ),
    ] = None,
) -> None:
    """Publish the header in OpenAPI. The middleware is what enforces it."""
    del idempotency_key


_IDEMPOTENT = [Depends(declare_idempotency_key)]


class BadPage(ValueError):
    def __init__(self, message: str, param: str) -> None:
        super().__init__(message)
        self.param = param


def _page(limit: int, cursor: str | None) -> tuple[int, int]:
    if limit < 1 or limit > _PAGE_MAX:
        raise BadPage(f"limit must be between 1 and {_PAGE_MAX}", "limit")
    if cursor is None or cursor == "":
        return limit, 0
    if not cursor.isdigit():
        raise BadPage("cursor is not valid", "cursor")
    return limit, int(cursor)


def _next_cursor(offset: int, limit: int, more: bool) -> str | None:
    if not more:
        return None
    return str(offset + limit)


@router.get("/health", response_model=Health, responses=_ERRORS)
def health(_auth: Annotated[Any, Depends(require_scope("read"))]) -> dict[str, Any]:
    backend = os.environ.get("LYRE_WORKER", "acestep")
    status = jobs.get_worker_status()
    if status is None:
        gpu = f"worker backend: {backend} (not reported yet -- is worker.run_worker running?)"
        ready = False
        dit_loaded = None
    else:
        gpu = status["message"] or f"worker backend: {backend}"
        ready = bool(status["ready"])
        if not ready:
            gpu = f"unavailable: {gpu}"
        dit_loaded = status["loaded_dit_profile"]
    counts = jobs.queue_counts()
    return {
        "ok": True,
        "backend": backend,
        "ready": ready,
        "gpu": gpu,
        "dit_loaded": dit_loaded,
        "queue": counts,
    }


@router.get(
    "/capabilities",
    response_model=Capabilities,
    response_model_by_alias=True,
    responses=_ERRORS,
)
def get_capabilities(_auth: Annotated[Any, Depends(require_scope("read"))]) -> dict[str, Any]:
    """Readiness, profiles, presets, and request limits (SPEC.md sec 16.6)."""
    return capability_report.capabilities()


@router.post(
    "/projects/{project_id}/preset",
    response_model=PresetResult,
    responses={**_PLAN_ERRORS, **_POST_ERRORS},
    dependencies=_IDEMPOTENT,
)
def project_preset(
    project_id: str,
    body: PresetRequest,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    """Preview a preset (`apply: false`, the default) or apply it. Applying
    keeps every field the user set and never touches lyrics."""
    return creation.preset_on_project(
        project_id, body.preset_id, apply=body.apply, expected_revision=body.expected_revision
    )


@router.post(
    "/projects",
    response_model=Project,
    responses=_POST_ERRORS,
    dependencies=_IDEMPOTENT,
)
def create_project(
    body: ProjectCreate,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    return storage.create_project(title=body.title, query=body.query)


@router.get("/projects", response_model=ProjectPage, responses=_ERRORS)
def list_projects(
    _auth: Annotated[Any, Depends(require_scope("read"))],
    limit: int = _PAGE_DEFAULT,
    cursor: str | None = None,
) -> dict[str, Any]:
    limit, offset = _page(limit, cursor)
    projects = storage.list_projects()
    page = projects[offset : offset + limit]
    more = offset + limit < len(projects)
    return {"data": page, "next_cursor": _next_cursor(offset, limit, more)}


@router.get("/projects/{project_id}", response_model=ProjectDetail, responses=_ERRORS)
def get_project(
    project_id: str, _auth: Annotated[Any, Depends(require_scope("read"))]
) -> dict[str, Any]:
    return {
        "project": storage.load_project(project_id),
        "plan": storage.load_plan(project_id),
        "takes": storage.list_takes(project_id),
    }


@router.patch(
    "/projects/{project_id}",
    response_model=Project,
    responses=_ERRORS,
)
def patch_project(
    project_id: str,
    body: ProjectPatch,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    return storage.patch_project(project_id, body.model_dump(exclude_unset=True))


@router.delete(
    "/projects/{project_id}",
    status_code=204,
    responses={
        204: {"description": "Deleted. Queued jobs for this project are cancelled."},
        **_ERRORS,
    },
)
def delete_project(
    project_id: str,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> Response:
    # Same tombstone + filesystem removal as DELETE /api/projects/{id}.
    jobs.remove_project(project_id)
    return Response(status_code=204)


@router.put(
    "/projects/{project_id}/plan",
    response_model=Plan,
    responses=_PLAN_ERRORS,
)
def put_plan(
    project_id: str,
    body: PlanWrite,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    sent = body.model_dump(exclude_unset=True)
    expected = sent.pop("expected_revision", None)
    return storage.save_plan(project_id, sent, expected_revision=expected)


@router.get("/projects/{project_id}/takes", response_model=TakePage, responses=_ERRORS)
def list_takes(
    project_id: str,
    _auth: Annotated[Any, Depends(require_scope("read"))],
    limit: int = _PAGE_DEFAULT,
    cursor: str | None = None,
) -> dict[str, Any]:
    limit, offset = _page(limit, cursor)
    takes = storage.list_takes(project_id)
    page = takes[offset : offset + limit]
    return {"data": page, "next_cursor": _next_cursor(offset, limit, offset + limit < len(takes))}


@router.post(
    "/projects/{project_id}/active_take",
    response_model=Project,
    responses=_POST_ERRORS,
    dependencies=_IDEMPOTENT,
)
def set_active_take(
    project_id: str,
    body: ActiveTake,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    # An active take must be playable. A failed take stays in the history.
    take = storage.get_take(project_id, body.take_id)
    if take.get("error") is not None:
        raise ValueError(f"take has an error, cannot set active: {body.take_id}")
    return storage.set_active_take(project_id, body.take_id)


@router.patch(
    "/projects/{project_id}/takes/{take_id}",
    response_model=Take,
    responses=_ERRORS,
)
def patch_take(
    project_id: str,
    take_id: str,
    body: TakePatch,
    _auth: Annotated[Any, Depends(require_scope("write"))],
) -> dict[str, Any]:
    return storage.update_take_annotations(
        project_id, take_id, favorite=body.favorite, notes=body.notes
    )


@router.post(
    "/projects/{project_id}/uploads",
    response_model=UploadCreated,
    responses={
        **_POST_ERRORS,
        413: {"model": Error, "description": "Body over the upload cap"},
    },
    dependencies=_IDEMPOTENT,
)
async def upload_audio(
    project_id: str,
    _auth: Annotated[Any, Depends(require_scope("write"))],
    file: Annotated[UploadFile, File()],
) -> dict[str, str]:
    return await receive_audio_upload(project_id, file)


@router.get(
    "/projects/{project_id}/loras",
    response_model=list[Lora],
    responses=_ERRORS,
)
def list_loras(
    project_id: str,
    _auth: Annotated[Any, Depends(require_scope("read"))],
) -> list[dict[str, Any]]:
    return storage.list_loras(project_id)


_ZIP: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Zip of project.json, plan.json, the active mix, and optional stems.",
        "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
    },
    **_ERRORS,
}


@router.get("/projects/{project_id}/export", responses=_ZIP)
def export_project(
    project_id: str,
    _auth: Annotated[Any, Depends(require_scope("read"))],
    include_stems: bool = True,
) -> Response:
    project = storage.load_project(project_id)
    zip_bytes = storage.build_export_zip(project_id, include_stems=include_stems)
    filename = f"{storage.sanitize_filename(project['title'])}-export.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


_JOB_CREATED: dict[int | str, dict[str, Any]] = {
    200: {"model": Job, "description": "Idempotent replay of the queued job."},
    202: {
        "model": Job,
        "description": "Job queued.",
        "headers": {
            "Location": {
                "description": "Path of the job object.",
                "schema": {"type": "string"},
            }
        },
    },
    **_POST_ERRORS,
}


@router.post(
    "/songs",
    status_code=202,
    response_model=Job,
    responses=_JOB_CREATED,
    dependencies=_IDEMPOTENT,
)
def create_song(
    body: SongCreate,
    response: Response,
    _auth: Annotated[Any, Depends(require_scope("generate"))],
) -> dict[str, Any]:
    """Create a project, write its plan, and queue `generate`.

    Scope `generate` (SPEC.md sec 15.4). The response is the job object.
    Poll `GET /v1/jobs/{id}`, then fetch `artifacts`.
    """
    job = _accepted_job(queue_song(body))
    response.headers["Location"] = f"/v1/jobs/{job['id']}"
    return job


@router.post(
    "/projects/{project_id}/jobs",
    status_code=202,
    response_model=Job,
    responses=_JOB_CREATED,
    dependencies=_IDEMPOTENT,
)
def create_job(
    project_id: str,
    body: JobCreate,
    response: Response,
    _auth: Annotated[Any, Depends(require_scope("generate"))],
) -> dict[str, Any]:
    # One enqueue function for /api and /v1. The row is queued; the worker
    # is the only thing that runs it.
    job = _accepted_job(jobs.enqueue_job(project_id, body.model_dump()))
    response.headers["Location"] = f"/v1/jobs/{job['id']}"
    return job


@router.get("/jobs", response_model=JobPage, responses=_ERRORS)
def list_jobs(
    _auth: Annotated[Any, Depends(require_scope("read", "generate"))],
    limit: int = _PAGE_DEFAULT,
    cursor: str | None = None,
    project_id: str | None = None,
    action: str | None = None,
    active: bool = False,
) -> dict[str, Any]:
    limit, offset = _page(limit, cursor)
    rows, more = jobs.list_jobs_page(
        limit=limit,
        offset=offset,
        project_id=project_id,
        action=action,
        active=active,
    )
    # List rows have no payload. Seed falls back to null unless the worker
    # has already recorded one on a finished take, which `job_object` reads.
    data = []
    for row in rows:
        full = jobs.read_job(row["id"])
        data.append(job_object(full))
    return {"data": data, "next_cursor": _next_cursor(offset, limit, more)}


@router.get("/jobs/{job_id}", response_model=Job, responses=_ERRORS)
def get_job(
    job_id: str, _auth: Annotated[Any, Depends(require_scope("read", "generate"))]
) -> dict[str, Any]:
    return job_object(jobs.read_job(job_id))


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=Job,
    responses=_POST_ERRORS,
    dependencies=_IDEMPOTENT,
)
def cancel_job(
    job_id: str, _auth: Annotated[Any, Depends(require_scope("generate"))]
) -> dict[str, Any]:
    jobs.cancel_job(job_id)
    return job_object(jobs.read_job(job_id))


@router.post(
    "/jobs/{job_id}/retry",
    status_code=202,
    response_model=Job,
    responses=_JOB_CREATED,
    dependencies=_IDEMPOTENT,
)
def retry_job(
    job_id: str,
    response: Response,
    _auth: Annotated[Any, Depends(require_scope("generate"))],
) -> dict[str, Any]:
    """Queue an explicit retry of a failed, cancelled, or held job (SPEC.md
    sec 16.5). The new job's `retry_of` names this one. A job that failed
    while a Remote GPU was rendering may repeat paid compute."""
    job = _accepted_job(jobs.retry_job(job_id))
    response.headers["Location"] = f"/v1/jobs/{job['id']}"
    return job


def _artifact(project_id: str, take_id: str, name: str, media_type: str) -> FileResponse:
    storage.load_project(project_id)
    candidate = storage.take_dir(project_id, take_id) / name
    if not candidate.is_file():
        raise storage.TakeNotFound(take_id)
    # Resolve through the jail so a symlink inside the take cannot escape.
    path = storage._jail(storage.config.projects_dir(), candidate)
    return FileResponse(path, media_type=media_type, filename=name)


_FILE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "The take file.",
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
    },
    **_ERRORS,
}


@router.get("/artifacts/{project_id}/{take_id}/mix.wav", responses=_FILE)
def artifact_wav(
    project_id: str,
    take_id: str,
    _auth: Annotated[Any, Depends(require_scope("read", "generate"))],
) -> FileResponse:
    return _artifact(project_id, take_id, "mix.wav", "audio/wav")


@router.get("/artifacts/{project_id}/{take_id}/mix.mp3", responses=_FILE)
def artifact_mp3(
    project_id: str,
    take_id: str,
    _auth: Annotated[Any, Depends(require_scope("read", "generate"))],
) -> FileResponse:
    return _artifact(project_id, take_id, "mix.mp3", "audio/mpeg")


@router.get("/artifacts/{project_id}/{take_id}/lyrics.lrc", responses=_FILE)
def artifact_lrc(
    project_id: str,
    take_id: str,
    _auth: Annotated[Any, Depends(require_scope("read", "generate"))],
) -> FileResponse:
    return _artifact(project_id, take_id, "lyrics.lrc", "text/plain")
