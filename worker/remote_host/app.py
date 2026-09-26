"""The Remote GPU host's HTTP surface and job protocol.

A render takes longer than the ~100 s a provider's proxy will hold a request
open, so nothing here blocks on the GPU. A submit returns a token at once, the
client polls the job's status, downloads the audio once it is done, and then
acknowledges it. Results are never dropped on read: a response the proxy lost
is recovered by polling again, and only an ack (or a generous TTL) frees one.

Every request is authenticated before its body is read, and bodies are capped,
because the address in front of this service is public.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from worker.remote_host import build_id

SECRET_HEADER = "X-Lyre-Secret"

# A source take travels base64-encoded inside the submit body. Lyre caps an
# upload at 100 MB and a ten-minute stereo WAV is ~115 MB, so this leaves room
# for both plus base64's third.
MAX_BODY_BYTES = 256 * 1024 * 1024

# What a remote host can render. Style packs (LoRA train/apply) need adapter
# weights moved to and from the host and are not offered remotely yet.
SUPPORTED_ACTIONS = ("generate", "cover", "repaint", "extract", "lego", "complete")
DIT_PROFILES = ("iterate", "polish", "quality", "studio_ops")
SOURCE_SUFFIXES = (".wav", ".mp3", ".flac")

# One GPU occupant: jobs run one at a time. The local worker never submits a
# second job before the first finishes, so a short queue only absorbs retries.
QUEUE_LIMIT = 8

# A delivered result is kept briefly so a client whose download was cut off can
# fetch it again; an undelivered one is kept long enough to survive a local
# restart mid-render.
DELIVERED_TTL_SEC = 300.0
UNDELIVERED_TTL_SEC = 3600.0

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class _IngressMiddleware:
    """Authenticate and cap every request before its body is parsed."""

    def __init__(self, app: ASGIApp, *, secret: str, max_body_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.secret = secret.encode()
        self.max_body_bytes = max_body_bytes

    async def _respond(self, send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(SECRET_HEADER.lower().encode(), b"")
        if not hmac.compare_digest(supplied, self.secret):
            await self._respond(send, 401, f"bad or missing {SECRET_HEADER}")
            return
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    await self._respond(send, 413, "request body too large")
                    return
            except ValueError:
                await self._respond(send, 400, "invalid Content-Length")
                return

        consumed = 0
        started = False

        class _TooLarge(Exception):
            pass

        async def limited_receive() -> Any:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > self.max_body_bytes:
                    raise _TooLarge
            return message

        async def tracked_send(message: Any) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _TooLarge:
            if not started:
                await self._respond(send, 413, "request body too large")


class JobRequest(BaseModel):
    client_id: str = Field(max_length=64)
    take_id: str = Field(max_length=64)
    job: dict[str, Any]
    plan: dict[str, Any]
    src_audio_b64: str | None = None
    src_audio_suffix: str | None = None


class Host:
    """The job queue and the one thread that runs it.

    `backend_loader` returns the worker backend module (`worker.acestep_worker`
    in production, `worker.mock_worker` for tests), resolved lazily so importing
    this module never imports ACE-Step or torch. `download` fetches a profile's
    weights before it is used; None skips downloading (the mock needs none).
    """

    def __init__(
        self,
        backend_loader: Callable[[], ModuleType],
        *,
        jobs_dir: Path,
        download: Callable[[str], None] | None = None,
        free_disk_gb: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self._backend_loader = backend_loader
        self._backend: ModuleType | None = None
        self._jobs_dir = jobs_dir
        self._download = download
        self._free_disk_gb = free_disk_gb
        self._jobs: dict[str, dict[str, Any]] = {}
        self._client_tokens: dict[str, str] = {}
        self._queue: queue.Queue[str] = queue.Queue(maxsize=QUEUE_LIMIT)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.ready = False
        self.message = "starting: fetching models and loading ACE-Step"
        self.gpu = ""
        self.capabilities: dict[str, dict[str, Any]] = {}

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Start the job thread. It prepares the backend first, so a job
        submitted during a first model download simply waits its turn."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._main, daemon=True, name="lyre-remote-jobs")
        self._thread.start()

    def _main(self) -> None:
        self._startup()
        while True:
            token = self._queue.get()
            try:
                self._run(token)
            except Exception:
                # _run records its own failures; this only keeps the one
                # thread that runs jobs alive through a bug in that bookkeeping.
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def _startup(self) -> None:
        try:
            if self._download is not None:
                self._download("iterate")
            backend = self._backend_loader()
            self._backend = backend
            describe_gpu = getattr(backend, "_log_cuda_status", None)
            self.gpu = str(describe_gpu()) if describe_gpu is not None else backend.__name__
            initialize = getattr(backend, "initialize_worker", None)
            ready, message = initialize() if initialize is not None else (True, "ready")
        except Exception as exc:
            ready, message = False, f"remote host startup failed: {exc}"
            traceback.print_exc()
        self.ready, self.message = ready, message
        self._refresh_capabilities()

    def _refresh_capabilities(self) -> None:
        backend = self._backend
        check = getattr(backend, "supports_dit_profile", None) if backend else None
        capabilities: dict[str, dict[str, Any]] = {}
        for profile in DIT_PROFILES:
            if check is None:
                supported, reason = backend is not None, None
            else:
                try:
                    supported, reason = check(profile)
                except Exception as exc:
                    supported, reason = False, str(exc)
            capabilities[profile] = {"supported": supported, "reason": reason}
        self.capabilities = capabilities

    # ---- reads ------------------------------------------------------------
    def loaded_dit_profile(self) -> str | None:
        get_loaded = getattr(self._backend, "get_loaded_dit_profile", None)
        return get_loaded() if get_loaded is not None else None

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "build": build_id(),
            "ready": self.ready,
            "message": self.message,
            "gpu": self.gpu,
            "loaded_dit_profile": self.loaded_dit_profile(),
            "capabilities": self.capabilities,
            "actions": list(SUPPORTED_ACTIONS),
            "queue_depth": self._queue.qsize(),
            "disk_free_gb": self._free_disk_gb(),
        }

    def _job(self, token: str) -> dict[str, Any]:
        record = self._jobs.get(token)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown job token")
        return record

    def status(self, token: str) -> dict[str, Any]:
        record = self._job(token)
        out: dict[str, Any] = {
            "status": record["status"],
            "error": record["error"],
            "dit_loaded": record["dit_loaded"],
        }
        if record["status"] == "done":
            out["result"] = record["result"]
            if record["delivered_at"] is None:
                record["delivered_at"] = time.monotonic()
        return out

    def audio_path(self, token: str) -> Path:
        record = self._job(token)
        if record["status"] != "done":
            raise HTTPException(status_code=409, detail="job has no audio yet")
        path = record["dir"] / "take" / record["result"]["audio_name"]
        if not path.is_file():
            raise HTTPException(status_code=410, detail="audio was already released")
        return path

    # ---- writes -----------------------------------------------------------
    def submit(self, request: JobRequest) -> dict[str, str]:
        if not _ID_PATTERN.match(request.client_id) or not _ID_PATTERN.match(request.take_id):
            raise HTTPException(status_code=422, detail="client_id and take_id must be simple ids")
        action = request.job.get("action")
        if action not in SUPPORTED_ACTIONS:
            raise HTTPException(
                status_code=422, detail=f"action '{action}' is not available on a Remote GPU"
            )
        if request.job.get("lora_id") or request.job.get("lora_adapter_path"):
            raise HTTPException(
                status_code=422, detail="style packs are not available on a Remote GPU yet"
            )
        if request.job.get("dit_profile") not in DIT_PROFILES:
            raise HTTPException(status_code=422, detail="unknown dit_profile")
        source: bytes | None = None
        if request.src_audio_b64 is not None:
            if request.src_audio_suffix not in SOURCE_SUFFIXES:
                raise HTTPException(status_code=422, detail="unsupported source audio format")
            try:
                source = base64.b64decode(request.src_audio_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=422, detail="source audio is not base64") from None

        with self._lock:
            self._prune()
            existing = self._client_tokens.get(request.client_id)
            if existing in self._jobs:
                # A retried submit whose first response was lost. Never render twice.
                return {"token": existing}
            token = uuid.uuid4().hex
            job_dir = self._jobs_dir / token
            job_dir.mkdir(parents=True)
            job = {k: v for k, v in request.job.items() if k != "src_audio"}
            if source is not None:
                source_path = job_dir / f"source{request.src_audio_suffix}"
                source_path.write_bytes(source)
                job["src_audio"] = str(source_path)
            self._jobs[token] = {
                "status": "queued",
                "error": None,
                "dit_loaded": None,
                "result": None,
                "cancel": False,
                "dir": job_dir,
                "job": job,
                "plan": request.plan,
                "take_id": request.take_id,
                "finished_at": None,
                "delivered_at": None,
            }
            try:
                self._queue.put_nowait(token)
            except queue.Full:
                self._forget(token)
                raise HTTPException(status_code=429, detail="Remote GPU queue is full") from None
            self._client_tokens[request.client_id] = token
        return {"token": token}

    def ack(self, token: str) -> dict[str, bool]:
        with self._lock:
            record = self._jobs.get(token)
            if record is None:
                return {"ok": False}
            if record["status"] not in ("done", "error"):
                # Its files are still in use; free it once it finishes instead.
                record["cancel"] = True
                record["delivered_at"] = -DELIVERED_TTL_SEC
                return {"ok": True}
            self._forget(token)
        return {"ok": True}

    def cancel(self, token: str) -> dict[str, bool]:
        """Cancel a job that has not started. ACE-Step has no way to interrupt
        a render in progress, so a running job finishes and is then discarded."""
        record = self._jobs.get(token)
        if record is None:
            return {"ok": False}
        record["cancel"] = True
        return {"ok": True}

    # ---- the job thread ---------------------------------------------------
    def _run(self, token: str) -> None:
        record = self._jobs.get(token)
        if record is None:
            return
        if record["cancel"]:
            self._finish(record, error="canceled")
            return
        record["status"] = "running"
        job = record["job"]

        def _on_dit_loaded(profile: str) -> None:
            record["dit_loaded"] = profile

        try:
            if self._download is not None:
                self._download(job["dit_profile"])
            backend = self._backend or self._backend_loader()
            self._backend = backend
            take_dir = record["dir"] / "take"
            meta, plan_patch, lrc_text = backend.run_job(
                job=job,
                plan=record["plan"],
                take_id=record["take_id"],
                take_dir=take_dir,
                on_dit_loaded=_on_dit_loaded,
            )
            audio = next(
                (take_dir / name for name in ("mix.wav", "mix.mp3") if (take_dir / name).is_file()),
                None,
            )
            if audio is None:
                raise RuntimeError("the worker finished without writing mix.wav or mix.mp3")
            if record["cancel"]:
                self._finish(record, error="canceled")
                return
            record["result"] = {
                "meta": meta,
                "plan_patch": plan_patch,
                "lrc_text": lrc_text,
                "audio_name": audio.name,
                "audio_bytes": audio.stat().st_size,
            }
            # A job that runs is proof the backend works, whatever startup said.
            self.ready, self.message = True, f"'{self.loaded_dit_profile()}' loaded and ready"
            self._finish(record)
        except Exception as exc:
            traceback.print_exc()
            self._finish(record, error=str(exc))
        finally:
            self._refresh_capabilities()

    def _finish(self, record: dict[str, Any], error: str | None = None) -> None:
        record["status"] = "error" if error else "done"
        record["error"] = error
        record["finished_at"] = time.monotonic()
        source = record["job"].get("src_audio")
        if source:
            Path(source).unlink(missing_ok=True)

    # ---- housekeeping (callers hold self._lock) ---------------------------
    def _forget(self, token: str) -> None:
        record = self._jobs.pop(token, None)
        if record is not None:
            shutil.rmtree(record["dir"], ignore_errors=True)
        for client_id in [c for c, t in self._client_tokens.items() if t == token]:
            self._client_tokens.pop(client_id, None)

    def _prune(self) -> None:
        now = time.monotonic()
        for token, record in list(self._jobs.items()):
            if record["status"] not in ("done", "error"):
                continue
            delivered = record["delivered_at"]
            since = delivered if delivered is not None else record["finished_at"] or now
            ttl = DELIVERED_TTL_SEC if delivered is not None else UNDELIVERED_TTL_SEC
            if now - since > ttl:
                self._forget(token)


def create_app(host: Host, secret: str) -> FastAPI:
    app = FastAPI(title="The Wizard's Lyre Remote GPU host", docs_url=None, redoc_url=None)
    app.add_middleware(_IngressMiddleware, secret=secret)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return host.health()

    @app.post("/jobs")
    def submit(request: JobRequest) -> dict[str, str]:
        return host.submit(request)

    @app.get("/jobs/{token}")
    def status(token: str) -> dict[str, Any]:
        return host.status(token)

    @app.get("/jobs/{token}/audio")
    def audio(token: str) -> FileResponse:
        path = host.audio_path(token)
        media = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
        return FileResponse(path, media_type=media)

    @app.post("/jobs/{token}/ack")
    def ack(token: str) -> dict[str, bool]:
        return host.ack(token)

    @app.post("/jobs/{token}/cancel")
    def cancel(token: str) -> dict[str, bool]:
        return host.cancel(token)

    return app
