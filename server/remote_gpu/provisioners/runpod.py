"""Rent a GPU pod from Runpod and point Lyre at it.

Plain REST v2 over the standard library, the same transport the host client
uses. A vendor SDK in a local-first app would be the wrong signal for a lane
that is off by default, and the whole surface used here is five calls.

The pod runs the same host image anyone can run anywhere; nothing Runpod-shaped
reaches the rest of Lyre, which only ever learns a base URL and a secret.

The API key never leaves this machine. The pod receives only a secret generated
for that one session. Pod environment can be read back through the Runpod API,
so anyone holding the key can read that session's secret; it is worthless once
the pod is terminated.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from typing import Any

from server.remote_gpu import client, connection
from server.remote_gpu.provisioners.base import (
    ERROR,
    READY,
    STARTING,
    STOPPING,
    AlreadyRunning,
    ProvisionerError,
    Session,
)
from worker.remote_host import build_id

API = "https://api.runpod.io/v2"
HOST_PORT = 8000

# Everything Lyre creates carries the marker, so an orphan is recognisable by
# eye in the Runpod console. A name is not attribution, though: only the
# session id recorded in `connection` makes a pod ours to stop.
NAME_PREFIX = "wizards-lyre"
MARKER_ENV = "WIZARDS_LYRE_MANAGED"

# The published host image, tagged with `build_id()`. With LYRE_RUNPOD_IMAGE
# unset, a pod runs the tag that matches this checkout's own host sources, so
# the pod and the checkout cannot disagree about the code.
REGISTRY = "ghcr.io"
PUBLISHED_REPO = "wizards-ecosystem/wizards-lyre-remote-gpu"
_MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)

# ACE-Step's pinned torch is built for CUDA 12.8; an older host driver cannot
# run it, so never let Runpod place the pod on one.
MIN_CUDA_VERSION = "12.8"

# 24 GB holds every Lyre profile, XL included, without CPU offload. Read from
# Runpod's catalog on 2026-09-26: the RTX A5000 ($0.27/hr) had no Secure stock
# at all on two reads, the RTX 4090 ($0.74/hr) had some. A default that
# cannot start is worse than a dearer one that can.
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
# Core weights are ~10 GB and every profile ~40 GB, plus caches and scratch.
DEFAULT_DISK_GB = 60
QUOTE_CACHE_SEC = 300.0


def _settings() -> dict[str, Any]:
    return {
        "api_key": os.environ.get("LYRE_RUNPOD_API_KEY", "").strip(),
        "image": os.environ.get("LYRE_RUNPOD_IMAGE", "").strip(),
        "gpu_type": os.environ.get("LYRE_RUNPOD_GPU_TYPE", "").strip() or DEFAULT_GPU_TYPE,
        "cloud": (os.environ.get("LYRE_RUNPOD_CLOUD", "").strip() or "SECURE").upper(),
        "disk_gb": int(os.environ.get("LYRE_RUNPOD_DISK_GB") or DEFAULT_DISK_GB),
        "volume_id": os.environ.get("LYRE_RUNPOD_NETWORK_VOLUME_ID", "").strip(),
        "data_centers": [
            c.strip()
            for c in os.environ.get("LYRE_RUNPOD_DATA_CENTER_IDS", "").split(",")
            if c.strip()
        ],
    }


def base_url_for(pod_id: str) -> str:
    return f"https://{pod_id}-{HOST_PORT}.proxy.runpod.net"


class RunpodProvisioner:
    id = "runpod"
    label = "Runpod"

    def __init__(self) -> None:
        self._quote: tuple[float, dict[str, Any]] | None = None

    # ---- helpers ----------------------------------------------------------
    def _key(self) -> str:
        key = _settings()["api_key"]
        if not key:
            raise ProvisionerError(
                "LYRE_RUNPOD_API_KEY is not set. Create a key at console.runpod.io/user/settings, "
                "set it in the server's environment, and restart. See docs/runpod.md."
            )
        return key

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        key = self._key()
        try:
            status, data = client.request(
                method,
                f"{API}{path}",
                headers={"Authorization": f"Bearer {key}"},
                payload=payload,
                redact=(key,),
            )
        except client.RemoteGpuError as exc:
            raise ProvisionerError(f"Runpod is unreachable: {exc}") from None
        if status == 401:
            raise ProvisionerError("Runpod rejected the API key (401). Check LYRE_RUNPOD_API_KEY.")
        if status == 403:
            raise ProvisionerError(
                "The Runpod API key lacks permission for this (403). A read-only key "
                "cannot create or terminate pods."
            )
        if status == 404:
            return None
        if status >= 400:
            detail = data[:600].decode("utf-8", errors="replace").replace(key, "***")
            raise ProvisionerError(f"Runpod error {status}: {detail}")
        if not data:
            return {}
        try:
            return json.loads(data)
        except ValueError:
            raise ProvisionerError("Runpod returned a response that is not JSON") from None

    def _image(self) -> tuple[str, bool]:
        """The image to run, and whether it is the published default."""
        custom = _settings()["image"]
        if custom:
            return custom, False
        return f"{REGISTRY}/{PUBLISHED_REPO}:{build_id()}", True

    def _check_published(self, image: str) -> None:
        """Refuse before creating a pod that could never pull its image. Such a
        pod sits "pulling" and bills until someone stops it. The usual cause is
        a locally edited worker/remote_host or worker/acestep_worker, which
        changes the build id to one no release published."""
        tag = image.rsplit(":", 1)[1]
        try:
            status, data = client.request(
                "GET", f"https://{REGISTRY}/token?scope=repository:{PUBLISHED_REPO}:pull"
            )
            token = json.loads(data).get("token", "") if status == 200 else ""
            status, _ = client.request(
                "HEAD",
                f"https://{REGISTRY}/v2/{PUBLISHED_REPO}/manifests/{tag}",
                headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_TYPES},
            )
        except (client.RemoteGpuError, ValueError) as exc:
            raise ProvisionerError(
                f"Could not confirm the host image {image} is published ({exc}). "
                "Nothing was created; try again."
            ) from None
        if status != 200:
            raise ProvisionerError(
                f"No published host image matches this checkout (build {tag}), which usually "
                "means worker/remote_host or worker/acestep_worker changed locally. Nothing was "
                "created. Build and push your own with `./scripts/lyre remote-image` and set "
                "LYRE_RUNPOD_IMAGE. See docs/runpod.md."
            )

    def _load(self) -> Session | None:
        raw = connection.get_session()
        if raw is None:
            return None
        try:
            session = Session.from_dict(raw)
        except (TypeError, ValueError):
            return None
        return session if session.provisioner == self.id else None

    # ---- protocol ---------------------------------------------------------
    def configured(self) -> bool:
        return bool(_settings()["api_key"])

    def quote(self) -> dict[str, Any]:
        settings = _settings()
        quote: dict[str, Any] = {
            "gpu": settings["gpu_type"],
            "cloud": settings["cloud"],
            "disk_gb": settings["disk_gb"],
            "hourly_usd": None,
            "memory_gb": None,
            "availability": None,
        }
        if not settings["api_key"]:
            return quote
        now = time.monotonic()
        if self._quote is not None and now - self._quote[0] < QUOTE_CACHE_SEC:
            return self._quote[1]
        try:
            catalog = self._request(
                "GET",
                "/catalog/gpus?include=AVAILABILITY&product=POD"
                f"&cloud={settings['cloud']}&cudaVersions={MIN_CUDA_VERSION}",
            )
        except ProvisionerError:
            # The price is advisory; an unreachable catalog must not hide the control.
            return quote
        for gpu in (catalog or {}).get("gpus") or []:
            if gpu.get("id") == settings["gpu_type"]:
                price = (gpu.get("price") or {}).get(settings["cloud"].lower())
                quote["hourly_usd"] = float(price) if price else None
                quote["memory_gb"] = gpu.get("memory")
                quote["availability"] = gpu.get("availability")
                break
        self._quote = (now, quote)
        return quote

    def start(self) -> Session:
        self._key()
        settings = _settings()
        existing = self.adopt()
        if existing is not None:
            # Two pods is the expensive mistake: refuse rather than quietly
            # double the bill because a click was repeated.
            raise AlreadyRunning(f"A Runpod session is already running ({existing.id}).")
        image, published = self._image()
        if published:
            self._check_published(image)
        secret = secrets.token_hex(32)
        env = {
            # Authenticates every request; the only thing between a public
            # proxy URL and this GPU. Unique per session.
            "LYRE_REMOTE_SECRET": secret,
            "LYRE_REMOTE_PORT": str(HOST_PORT),
            MARKER_ENV: "1",
        }
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            env["HF_TOKEN"] = hf_token
        body: dict[str, Any] = {
            "name": f"{NAME_PREFIX}-{uuid.uuid4().hex[:8]}",
            "image": image,
            "cloud": settings["cloud"],
            "gpu": {"id": settings["gpu_type"], "count": 1, "minCudaVersion": MIN_CUDA_VERSION},
            "disk": settings["disk_gb"],
            "ports": [f"{HOST_PORT}/http"],
            "env": env,
        }
        if settings["volume_id"]:
            body["mounts"] = {
                "network": [{"volumeId": settings["volume_id"], "path": "/workspace"}]
            }
            env["LYRE_REMOTE_ROOT"] = "/workspace/wizards-lyre"
        if settings["data_centers"]:
            body["dataCenterIds"] = settings["data_centers"]

        created = self._request("POST", "/pods", body) or {}
        pod_id = str(created.get("id") or "")
        if not pod_id:
            raise ProvisionerError("Runpod accepted the request but returned no pod id")
        base_url = base_url_for(pod_id)
        session = Session(
            id=pod_id,
            provisioner=self.id,
            state=STARTING,
            base_url=base_url,
            detail="pod created; pulling the host image",
            started_at=time.time(),
            hourly_usd=(float(created["cost"]) if created.get("cost") is not None else None),
            gpu=str((created.get("gpu") or {}).get("id") or settings["gpu_type"]),
        )
        # Record before returning: a crash in the next instant must still leave
        # a pod this app can find and stop.
        connection.save_session(session.as_dict())
        connection.set_connection(base_url, secret)
        return session

    def _refresh(self, session: Session) -> Session | None:
        pod = self._request("GET", f"/pods/{session.id}")
        if pod is None:
            # Gone on Runpod's side (terminated in the console, or reclaimed):
            # nothing is billing any more, so forget it rather than block the
            # next start behind a pod that no longer exists.
            self._forget(session)
            return None
        status = str(pod.get("status") or "").upper()
        if status in {"EXITED", "TERMINATED", "ERROR"}:
            state, detail = ERROR, f"the pod is {status.lower()}; stop it to clear this"
        elif status != "RUNNING":
            state, detail = STARTING, f"pod {status.lower() or 'provisioning'}"
        elif not pod.get("runtime"):
            # RUNNING is reported as soon as the container object exists; a
            # first pull of the host image is the long wait.
            state, detail = STARTING, "pulling the host image (the slow part)"
        else:
            state, detail = READY, "pod is up"
        updated = Session(
            id=session.id,
            provisioner=self.id,
            state=state,
            base_url=session.base_url,
            detail=detail,
            started_at=session.started_at,
            hourly_usd=(float(pod["cost"]) if pod.get("cost") is not None else session.hourly_usd),
            gpu=str((pod.get("gpu") or {}).get("id") or session.gpu),
        )
        connection.save_session(updated.as_dict())
        return updated

    def _forget(self, session: Session) -> None:
        connection.clear_session()
        current = connection.get_connection()
        if current is not None and current.base_url == session.base_url:
            connection.clear_connection()

    def status(self) -> Session | None:
        session = self._load()
        return self._refresh(session) if session is not None else None

    def adopt(self) -> Session | None:
        """What a previous process left running, if anything. Only a session
        this app recorded counts; a pod that merely looks like ours is not
        ours to terminate."""
        session = self._load()
        if session is None:
            return None
        try:
            return self._refresh(session)
        except ProvisionerError:
            return session  # unreachable right now; still ours, still billing

    def stop(self) -> None:
        session = self._load()
        if session is None:
            return
        connection.save_session(
            Session(
                id=session.id,
                provisioner=self.id,
                state=STOPPING,
                base_url=session.base_url,
                detail="terminating",
                started_at=session.started_at,
                hourly_usd=session.hourly_usd,
                gpu=session.gpu,
            ).as_dict()
        )
        # Terminate, never stop: a stopped pod gives up its GPU with no promise
        # of getting it back, and keeps billing its disk. If this raises, the
        # record stays so the user can retry instead of losing track of a pod
        # that may still be billing.
        self._request("DELETE", f"/pods/{session.id}")
        self._forget(session)
