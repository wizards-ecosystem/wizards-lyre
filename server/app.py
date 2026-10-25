"""FastAPI app: health, projects, plan, takes, jobs, Remote GPU, `/v1`. No CUDA here.

See SPEC.md sec 8 for `/api` and sec 15 for `/v1`. The default bind is
127.0.0.1 with auth off. `LYRE_BIND` and `LYRE_REQUIRE_AUTH` are the opt-in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from server import capabilities, config, creation, jobs, storage
from server.generation.presets import PresetNotFound
from server.mcp.http import gateway as mcp_gateway
from server.mcp.http import mcp_lifespan
from server.remote_gpu import service as remote_gpu
from server.remote_gpu.provisioners.base import AlreadyRunning, ProvisionerError
from server.uploads import receive_audio_upload
from server.v1.auth import ApiAuthError
from server.v1.idempotency import IdempotencyMiddleware
from server.v1.routes import BadPage
from server.v1.routes import router as v1_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    jobs.init_db()
    # Report (never start) a Remote GPU an earlier run left up, and terminate
    # ours on a clean exit. Both are no-ops for the default manual provisioner.
    await asyncio.to_thread(remote_gpu.on_startup)
    try:
        async with mcp_lifespan():
            yield
    finally:
        await asyncio.to_thread(remote_gpu.on_shutdown)


app = FastAPI(title="The Wizard's Lyre", version="0.3.0", lifespan=_lifespan)


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


# Innermost first. The body limit counts bytes, then idempotency hashes the
# buffered body for `/v1` POSTs that send a key, then the router runs.
# Exception handlers sit inside this stack, so a stored 400 is the response
# those handlers already built.
app.add_middleware(IdempotencyMiddleware)
app.add_middleware(_MaxRequestBodyMiddleware)


class _AgentSurfaceMiddleware:
    """When auth is required, serve `/v1` only. The studio and `/api` stay
    on the auth-off loopback server (SPEC.md sec 15.2)."""

    def __init__(self, asgi_app: ASGIApp) -> None:
        self.app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        try:
            restricted = config.auth_required()
        except config.BindConfigError as exc:
            response = _bind_misconfigured(exc)
            await response(scope, receive, send)
            return
        if restricted and not _agent_surface(path):
            response = JSONResponse(
                status_code=404,
                content={"error": {"code": "not_found", "message": "not found", "param": None}},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(_AgentSurfaceMiddleware)


def _agent_surface(path: str) -> bool:
    """Routes that stay mounted when auth is required (SPEC.md sec 15.2)."""
    return path.startswith("/v1") or path == "/mcp" or path.startswith("/mcp/")


def _bind_misconfigured(exc: config.BindConfigError) -> JSONResponse:
    """Generic body for a bad bind. The exception text stays in the log."""
    request_id = secrets.token_hex(8)
    logger.error("bind configuration error request_id=%s", request_id, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "misconfigured",
                "message": "server configuration error",
                "param": None,
                "request_id": request_id,
            }
        },
    )


def _error_body(
    request, status: int, message: str, *, code: str, param: str | None = None
) -> JSONResponse:
    if request.url.path.startswith("/v1"):
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message, "param": param}},
            headers=headers,
        )
    return JSONResponse(status_code=status, content={"detail": message})


class CreateProjectBody(BaseModel):
    title: str | None = None
    query: str | None = None


class PatchProjectBody(BaseModel):
    title: str | None = None
    dit_profile: str | None = None
    favorite: bool | None = None


class ActiveTakeBody(BaseModel):
    take_id: str


class TakeAnnotationBody(BaseModel):
    favorite: bool | None = None
    notes: str | None = None


class RemoteGpuConnectionBody(BaseModel):
    base_url: str = Field(max_length=300)
    secret: str = Field(max_length=300)


class JobBody(BaseModel):
    action: str
    dit_profile: str | None = None
    source_take_id: str | None = None
    source_take_ids: list[str] | None = None
    upload_path: str | None = None
    repainting_start: float = 0
    repainting_end: float = -1
    track_name: str | None = None
    name: str | None = None
    # SPEC.md sec 4.4 "LoRA train / load": a trained style-pack lora to
    # apply to this generate/cover/repaint (server.jobs.enqueue_job
    # resolves and validates it -- see _resolve_lora / _resolve_dit_profile).
    lora_id: str | None = None
    # SPEC.md sec 8.1: audio_cover_strength is a 0-1 mix ratio ACE-Step
    # expects; ge/le also reject NaN/+-inf (any comparison with NaN is
    # False, so it fails both bounds) instead of forwarding them to the
    # worker and causing an avoidable failure deep inside generation.
    audio_cover_strength: float = Field(0.7, ge=0.0, le=1.0)
    seed: int = -1
    batch_size: int = 1
    # SPEC.md sec 16.3: `from_idea` re-plans from the idea, `from_brief`
    # renders the stored brief. Omitted: the plan's own creation_mode.
    creation_mode: Literal["from_idea", "from_brief"] | None = None
    # SPEC.md sec 16.4: one submission per project. The same id and body
    # return the existing job; a different body is 409.
    submission_id: str | None = None


@app.exception_handler(storage.PathJailError)
def _path_jail_handler(request, exc: storage.PathJailError):
    return _error_body(request, 400, str(exc), code="invalid_request")


@app.exception_handler(storage.ProjectNotFound)
def _project_not_found_handler(request, exc: storage.ProjectNotFound):
    return _error_body(request, 404, f"project not found: {exc}", code="not_found")


@app.exception_handler(storage.TakeNotFound)
def _take_not_found_handler(request, exc: storage.TakeNotFound):
    return _error_body(request, 404, f"take not found: {exc}", code="not_found")


@app.exception_handler(storage.PlanRevisionConflict)
def _plan_revision_handler(request, exc: storage.PlanRevisionConflict):
    return _error_body(request, 409, str(exc), code="conflict", param="expected_revision")


@app.exception_handler(PresetNotFound)
def _preset_not_found_handler(request, exc: PresetNotFound):
    return _error_body(
        request, 404, f"preset not found: {exc}", code="not_found", param="preset_id"
    )


@app.exception_handler(storage.ExportError)
def _export_error_handler(request, exc: storage.ExportError):
    return _error_body(request, 400, str(exc), code="invalid_request", param=exc.param)


@app.exception_handler(storage.LoraNotFound)
def _lora_not_found_handler(request, exc: storage.LoraNotFound):
    return _error_body(request, 404, f"lora not found: {exc}", code="not_found")


@app.exception_handler(jobs.JobNotFound)
def _job_not_found_handler(request, exc: jobs.JobNotFound):
    return _error_body(request, 404, f"job not found: {exc}", code="not_found")


@app.exception_handler(jobs.SubmissionConflict)
def _submission_conflict_handler(request, exc: jobs.SubmissionConflict):
    return _error_body(request, 409, str(exc), code="conflict", param="submission_id")


@app.exception_handler(jobs.JobError)
def _job_error_handler(request, exc: jobs.JobError):
    return _error_body(
        request,
        400,
        str(exc),
        code=getattr(exc, "code", "invalid_request"),
        param=getattr(exc, "param", None),
    )


@app.exception_handler(ProvisionerError)
def _provisioner_error_handler(request, exc: ProvisionerError):
    # 409 for a state the user can fix (already running, stop it first);
    # anything else is the provider or its configuration refusing.
    status = 409 if isinstance(exc, AlreadyRunning | remote_gpu.Conflict) else 400
    code = "conflict" if status == 409 else "invalid_request"
    return _error_body(request, status, str(exc), code=code)


@app.exception_handler(ApiAuthError)
def _api_auth_handler(request, exc: ApiAuthError):
    return _error_body(request, exc.status_code, exc.message, code=exc.code)


@app.exception_handler(HTTPException)
def _http_exception_handler(request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    if exc.status_code == 401:
        code = "unauthorized"
    elif exc.status_code == 403:
        code = "forbidden"
    elif exc.status_code == 404:
        code = "not_found"
    elif exc.status_code == 413:
        code = "payload_too_large"
    else:
        code = "invalid_request"
    return _error_body(request, exc.status_code, message, code=code)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc: RequestValidationError):
    # `/api` keeps FastAPI's 422 `{"detail": [...]}` body. `/v1` uses the
    # shared error schema and 400 for an invalid body (SPEC.md sec 15.3).
    if not request.url.path.startswith("/v1"):
        return await request_validation_exception_handler(request, exc)
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = first.get("loc") or ()
    param = None
    for part in reversed(loc):
        if isinstance(part, str) and part not in {"body", "query", "path", "header"}:
            param = part
            break
    message = str(first.get("msg") or "invalid request")
    code = "invalid_plan" if request.url.path.rstrip("/").endswith("/plan") else "invalid_request"
    return _error_body(request, 400, message, code=code, param=param)


@app.exception_handler(ValueError)
def _value_error_handler(request, exc: ValueError):
    message = str(exc)
    param = getattr(exc, "param", None) if isinstance(exc, BadPage) else None
    code = "invalid_request"
    if message.startswith("invalid plan field"):
        code = "invalid_plan"
        if param is None and "'" in message:
            param = message.split("'", 2)[1]
    return _error_body(request, 400, message, code=code, param=param)


@app.get("/api/health")
def health() -> dict[str, Any]:
    backend = os.environ.get("LYRE_WORKER", "acestep")
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


@app.get("/api/capabilities")
def get_capabilities() -> dict[str, Any]:
    # SPEC.md sec 16.6: readiness, profiles, presets, and limits in one read.
    return capabilities.capabilities()


class PresetBody(BaseModel):
    preset_id: str
    apply: bool = False
    expected_revision: int | None = None


@app.post("/api/projects/{project_id}/preset")
def project_preset(project_id: str, body: PresetBody) -> dict[str, Any]:
    # Preview by default; `apply: true` writes the plan (SPEC.md sec 16.6).
    return creation.preset_on_project(
        project_id, body.preset_id, apply=body.apply, expected_revision=body.expected_revision
    )


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


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    # SPEC.md sec 9.1. `/v1` calls the same function, including the tombstone
    # that closes the enqueue race and the restore if the filesystem removal
    # fails. See `jobs.remove_project`.
    jobs.remove_project(project_id)
    return Response(status_code=204)


@app.put("/api/projects/{project_id}/plan")
def put_plan(project_id: str, body: dict[str, Any]) -> dict:
    # SPEC.md sec 16.3: an optional `expected_revision` makes the replace
    # conditional. It is a request field, not part of plan.json.
    expected = body.pop("expected_revision", None)
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
        raise ValueError("expected_revision must be an integer")
    return storage.save_plan(project_id, body, expected_revision=expected)


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


@app.patch("/api/projects/{project_id}/takes/{take_id}")
def patch_take(project_id: str, take_id: str, body: TakeAnnotationBody) -> dict:
    # Take-level counterpart to patch_project's favorite field (SPEC.md sec
    # 12 Phase 6). storage.update_take_annotations only touches the field(s)
    # actually provided, never the take's immutable generation data.
    return storage.update_take_annotations(
        project_id, take_id, favorite=body.favorite, notes=body.notes
    )


@app.get("/api/projects/{project_id}/takes/{take_id}/audio")
def get_take_audio(project_id: str, take_id: str) -> FileResponse:
    path = storage.take_audio_path(project_id, take_id)
    media_type = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    # The stored full-resolution file, named by take and its real extension
    # (SPEC.md sec 16.6), so a browser download is never a mislabelled MP3.
    return FileResponse(path, media_type=media_type, filename=f"{take_id}{path.suffix}")


@app.get("/api/projects/{project_id}/takes/{take_id}/lrc")
def get_take_lrc(project_id: str, take_id: str) -> FileResponse:
    # SPEC.md sec 7 lyrics.lrc: optional, phase 4. take_lrc_path raises
    # storage.TakeNotFound (-> 404, see the exception handler above) when
    # this take has none, rather than the UI having to guess from a 404 on
    # a hypothetical always-present route.
    path = storage.take_lrc_path(project_id, take_id)
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/api/projects/{project_id}/export")
def export_project(
    project_id: str, include_stems: bool = True, take_id: str | None = None
) -> Response:
    # SPEC.md sec 16.6: the selected take's full-resolution mix (the active
    # take unless take_id names one), its LRC, optional stems, project.json,
    # plan.json, and manifest.json, built by storage.build_export_zip.
    project = storage.load_project(project_id)
    zip_bytes = storage.build_export_zip(project_id, include_stems=include_stems, take_id=take_id)
    filename = f"{storage.sanitize_filename(project['title'])}-export.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/projects/{project_id}/uploads")
async def upload_audio(project_id: str, file: UploadFile = File(...)) -> dict:
    # SPEC.md sec 12 Phase 6. `/v1` calls the same streaming helper.
    return await receive_audio_upload(project_id, file)


@app.get("/api/projects/{project_id}/loras")
def list_loras(project_id: str) -> list[dict]:
    return storage.list_loras(project_id)


@app.post("/api/projects/{project_id}/jobs")
def create_job(project_id: str, body: JobBody) -> dict:
    return jobs.enqueue_job(project_id, body.model_dump())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return jobs.get_job(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    # Same cooperative cancel as `/v1` (SPEC.md sec 15.5); the studio needs it
    # to stop a job it started (sec 16.6).
    return jobs.cancel_job(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    # SPEC.md sec 16.5: a new job linked by `retry_of` that repeats the
    # frozen request.
    return jobs.retry_job(job_id)


@app.get("/api/jobs")
def list_jobs(
    limit: int = 20,
    project_id: str | None = None,
    action: str | None = None,
    active: bool = False,
) -> list[dict]:
    # SPEC.md sec 8 "GET /api/jobs | recent jobs". project_id/action are
    # optional narrowing filters (applied before the limit) -- the web UI
    # needs them to recover the open project's still-running train_lora job
    # after a refresh; without a project filter a long training job gets
    # pushed out of the unfiltered top-N by jobs enqueued behind it.
    # active=true returns the project's complete still-active (queued/
    # running) worklist with no recency truncation, so recovery finds an
    # old running training no matter how many newer or finished rows exist
    # behind it (see jobs.list_recent_jobs for why `limit` doesn't apply).
    return jobs.list_recent_jobs(limit=limit, project_id=project_id, action=action, active=active)


# Remote GPU (SPEC.md sec 3.1): opt-in. Nothing here rents hardware except
# POST /api/remote-gpu/session, which only the user's own click sends.
@app.get("/api/remote-gpu")
def get_remote_gpu() -> dict[str, Any]:
    return remote_gpu.status()


@app.put("/api/remote-gpu/connection")
def put_remote_gpu_connection(body: RemoteGpuConnectionBody) -> dict[str, Any]:
    return remote_gpu.connect(body.base_url, body.secret)


@app.delete("/api/remote-gpu/connection")
def delete_remote_gpu_connection() -> dict[str, Any]:
    return remote_gpu.disconnect()


@app.post("/api/remote-gpu/session")
def start_remote_gpu_session() -> dict[str, Any]:
    return remote_gpu.start()


@app.delete("/api/remote-gpu/session")
def stop_remote_gpu_session() -> dict[str, Any]:
    return remote_gpu.stop()


app.include_router(v1_router)

# Streamable HTTP MCP. Registered before the SPA so `/mcp` is not swallowed
# by the static mount. Not part of the OpenAPI document; see docs/MCP.md.
# A class instance is a raw ASGI app. `add_route` would type it as a request
# endpoint, and `mount` redirects `/mcp` to `/mcp/`.
app.router.routes.append(
    Route(
        "/mcp",
        mcp_gateway,
        methods=["GET", "POST", "DELETE"],
        name="mcp",
        include_in_schema=False,
    )
)
app.router.routes.append(
    Route(
        "/mcp/",
        mcp_gateway,
        methods=["GET", "POST", "DELETE"],
        name="mcp-slash",
        include_in_schema=False,
    )
)


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
    import sys

    import uvicorn

    try:
        host = config.validate_startup()
    except config.BindConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    uvicorn.run(app, host=host, port=config.lyre_port())


if __name__ == "__main__":
    main()
