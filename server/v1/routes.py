"""`/v1` routes. Reads, the job object, cancel, and stable artifact GETs.

Writes, idempotency, one-shot songs, SSE, webhooks, and signed artifact
tokens are later phases (SPEC.md sec 12). `/api` is unchanged.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from server import jobs, storage
from server.v1.auth import require_scope
from server.v1.models import (
    Error,
    Health,
    Job,
    JobPage,
    ProjectDetail,
    ProjectPage,
    TakePage,
)
from server.v1.present import job_object

router = APIRouter(prefix="/v1")

_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": Error, "description": "Invalid request"},
    401: {"model": Error, "description": "Missing or unknown API key"},
    403: {"model": Error, "description": "Known API key without the required scope"},
    404: {"model": Error, "description": "Unknown project, take, or job"},
}

_PAGE_DEFAULT = 50
_PAGE_MAX = 100


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


@router.post("/jobs/{job_id}/cancel", response_model=Job, responses=_ERRORS)
def cancel_job(
    job_id: str, _auth: Annotated[Any, Depends(require_scope("generate"))]
) -> dict[str, Any]:
    jobs.cancel_job(job_id)
    return job_object(jobs.read_job(job_id))


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
