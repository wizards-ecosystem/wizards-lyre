"""FastAPI app: health, projects, plan, takes, jobs. No CUDA here.

See SPEC.md sec 8 for the HTTP API and sec 14 for phase 1 definition of done.
Binds 127.0.0.1 only; port defaults to 8421, overridable via BARD_PORT.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server import config, jobs, storage


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    jobs.init_db()
    yield


app = FastAPI(title="Wizard's Bard", version="0.1.0", lifespan=_lifespan)


class CreateProjectBody(BaseModel):
    title: Optional[str] = None
    query: Optional[str] = None


class PatchProjectBody(BaseModel):
    title: Optional[str] = None
    dit_profile: Optional[str] = None


class JobBody(BaseModel):
    action: str
    dit_profile: Optional[str] = None
    source_take_id: Optional[str] = None
    upload_path: Optional[str] = None
    repainting_start: float = 0
    repainting_end: float = -1
    track_name: Optional[str] = None
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
