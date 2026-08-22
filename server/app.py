"""FastAPI app: health, projects, plan, takes, jobs. No CUDA here.

See SPEC.md sec 8 for the HTTP API and sec 14 for phase 1 definition of done.
Binds 127.0.0.1 only; port defaults to 8421, overridable via BARD_PORT.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from server import config, jobs, storage


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    jobs.init_db()
    yield


app = FastAPI(title="Wizard's Bard", version="0.1.0", lifespan=_lifespan)


class _MaxRequestBodyMiddleware:
    """Raw ASGI middleware -- not `BaseHTTPMiddleware`, which buffers the
    whole body itself -- that counts bytes as the ASGI server actually
    delivers them and aborts once a request body exceeds
    `storage.MAX_UPLOAD_BYTES` (plus a little slack for multipart
    boundary/header overhead around the file part). This covers a
    chunked-transfer-encoded body (no `Content-Length` to precheck) the same
    way it covers one with a declared length, since it never trusts the
    header, only what actually arrives.

    Sits *above* routing, so an oversized `POST .../uploads` is rejected
    while `python-multipart` is still asking `receive()` for more data --
    before it can spool an arbitrarily large body to a temp file on disk.
    The per-chunk counter inside `upload_audio` only bounded the copy *out*
    of that already-fully-spooled temp file, which was too late to protect
    disk/memory (reviewer-flagged).

    Applied to every request, not just the uploads endpoint -- simplest
    possible rule, and every other endpoint's JSON body is trivially small
    next to `storage.MAX_UPLOAD_BYTES`. Reads `storage.MAX_UPLOAD_BYTES`
    fresh on each request rather than capturing it once at startup, so it
    can never drift out of sync with the exact cap
    `storage.open_upload_destination` enforces on the file content itself.
    """

    def __init__(self, app: ASGIApp, extra_bytes: int = 64 * 1024) -> None:
        self.app = app
        self.extra_bytes = extra_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_bytes = storage.MAX_UPLOAD_BYTES + self.extra_bytes
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body") or b"")
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="request body too large")
            return message

        await self.app(scope, limited_receive, send)


app.add_middleware(_MaxRequestBodyMiddleware)


class CreateProjectBody(BaseModel):
    title: Optional[str] = None
    query: Optional[str] = None


class PatchProjectBody(BaseModel):
    title: Optional[str] = None
    dit_profile: Optional[str] = None
    favorite: Optional[bool] = None


class ActiveTakeBody(BaseModel):
    take_id: str


class JobBody(BaseModel):
    action: str
    dit_profile: Optional[str] = None
    source_take_id: Optional[str] = None
    source_take_ids: Optional[list[str]] = None
    upload_path: Optional[str] = None
    repainting_start: float = 0
    repainting_end: float = -1
    track_name: Optional[str] = None
    name: Optional[str] = None
    # SPEC.md sec 8.1: audio_cover_strength is a 0-1 mix ratio ACE-Step
    # expects; ge/le also reject NaN/+-inf (any comparison with NaN is
    # False, so it fails both bounds) instead of forwarding them to the
    # worker and causing an avoidable failure deep inside generation.
    audio_cover_strength: float = Field(0.7, ge=0.0, le=1.0)
    seed: int = -1
    batch_size: int = 1


@app.exception_handler(storage.PathJailError)
def _path_jail_handler(request, exc: storage.PathJailError):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(storage.ProjectNotFound)
def _project_not_found_handler(request, exc: storage.ProjectNotFound):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=404, content={"detail": f"project not found: {exc}"})


@app.exception_handler(storage.TakeNotFound)
def _take_not_found_handler(request, exc: storage.TakeNotFound):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=404, content={"detail": f"take not found: {exc}"})


@app.exception_handler(jobs.JobNotFound)
def _job_not_found_handler(request, exc: jobs.JobNotFound):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=404, content={"detail": f"job not found: {exc}"})


@app.exception_handler(jobs.JobError)
def _job_error_handler(request, exc: jobs.JobError):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _value_error_handler(request, exc: ValueError):  # noqa: ANN001, ARG001
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict[str, Any]:
    backend = os.environ.get("BARD_WORKER", "acestep")
    status = jobs.get_worker_status()
    if status is None:
        return {
            "ok": True,
            "gpu": f"worker backend: {backend} (not reported yet -- is worker.run_worker running?)",
            "dit_loaded": None,
        }
    gpu = status["message"] or f"worker backend: {backend}"
    if not status["ready"]:
        gpu = f"unavailable: {gpu}"
    return {
        "ok": True,
        "gpu": gpu,
        "dit_loaded": status["loaded_dit_profile"],
    }


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return storage.list_projects()


@app.post("/api/projects")
def create_project(body: CreateProjectBody) -> dict:
    return storage.create_project(title=body.title, query=body.query)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    project = storage.load_project(project_id)
    plan = storage.load_plan(project_id)
    takes = storage.list_takes(project_id)
    return {"project": project, "plan": plan, "takes": takes}


@app.patch("/api/projects/{project_id}")
def patch_project(project_id: str, body: PatchProjectBody) -> dict:
    return storage.patch_project(project_id, body.model_dump(exclude_unset=True))


@app.put("/api/projects/{project_id}/plan")
def put_plan(project_id: str, body: dict[str, Any]) -> dict:
    return storage.save_plan(project_id, body)


@app.get("/api/projects/{project_id}/takes")
def list_takes(project_id: str) -> list[dict]:
    return storage.list_takes(project_id)


@app.post("/api/projects/{project_id}/active_take")
def set_active_take(project_id: str, body: ActiveTakeBody) -> dict:
    # storage.get_take raises TakeNotFound (-> 404, see the exception handler
    # above) if the take doesn't exist. An active take must be playable, so
    # reject a take that failed generation (SPEC.md sec 7.3 `error`).
    take = storage.get_take(project_id, body.take_id)
    if take.get("error") is not None:
        raise ValueError(f"take has an error, cannot set active: {body.take_id}")
    return storage.set_active_take(project_id, body.take_id)


@app.get("/api/projects/{project_id}/takes/{take_id}/audio")
def get_take_audio(project_id: str, take_id: str) -> FileResponse:
    path = storage.take_audio_path(project_id, take_id)
    media_type = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/projects/{project_id}/takes/{take_id}/lrc")
def get_take_lrc(project_id: str, take_id: str) -> FileResponse:
    # SPEC.md sec 7 lyrics.lrc: optional, phase 4. take_lrc_path raises
    # storage.TakeNotFound (-> 404, see the exception handler above) when
    # this take has none, rather than the UI having to guess from a 404 on
    # a hypothetical always-present route.
    path = storage.take_lrc_path(project_id, take_id)
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: str, include_stems: bool = True) -> Response:
    # SPEC.md sec 12 Phase 5 / sec 9.2: project.json + plan.json + active
    # mix + optional stems, built in memory by storage.build_export_zip.
    project = storage.load_project(project_id)
    zip_bytes = storage.build_export_zip(project_id, include_stems=include_stems)
    filename = f"{storage.sanitize_filename(project['title'])}-export.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/projects/{project_id}/uploads")
async def upload_audio(project_id: str, file: UploadFile = File(...)) -> dict:
    # SPEC.md sec 12 Phase 6: drag-drop a local WAV/MP3 in as a cover/repaint
    # source, path-jailed under projects/. Streamed to disk in bounded
    # chunks -- never `await file.read()` in one shot -- so MAX_UPLOAD_BYTES
    # is enforced as bytes arrive; an oversized upload is aborted (and its
    # partial file deleted) mid-stream instead of only after the entire body
    # has already been buffered in memory (reviewer-flagged: an unbounded
    # single read is a memory-exhaustion vector). The final path is only
    # published (atomic rename) once the whole body has been accepted, and
    # is exactly what JobBody.upload_path already knows how to resolve
    # (SPEC.md sec 8.1).
    tmp_path, dest_path = storage.open_upload_destination(project_id, file.filename or "")
    total_bytes = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(storage.UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > storage.MAX_UPLOAD_BYTES:
                    raise ValueError(
                        f"upload too large: exceeds {storage.MAX_UPLOAD_BYTES} bytes"
                    )
                out.write(chunk)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    upload_path = storage.finalize_upload(tmp_path, dest_path)
    return {"upload_path": upload_path}


@app.get("/api/projects/{project_id}/loras")
def list_loras(project_id: str) -> list[dict]:
    return storage.list_loras(project_id)


@app.post("/api/projects/{project_id}/jobs")
def create_job(project_id: str, body: JobBody) -> dict:
    return jobs.enqueue_job(project_id, body.model_dump())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return jobs.get_job(job_id)


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> list[dict]:
    return jobs.list_recent_jobs(limit=limit)


# Prod: FastAPI serves the built SPA from web/dist (SPEC.md sec 5). Registered
# last so it never shadows the /api/* routes above. `npm run build` in web/
# produces dist/; until then, "/" reports how to build or run the dev server.
_WEB_DIST = config.REPO_ROOT / "web" / "dist"
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
else:

    @app.get("/")
    def _web_not_built() -> dict[str, Any]:
        return {
            "ok": True,
            "hint": "web/dist not found; run `npm install && npm run build` in web/, "
            "or `npm run dev` for the Vite dev server (proxies /api to this server).",
        }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.bard_port())


if __name__ == "__main__":
    main()
