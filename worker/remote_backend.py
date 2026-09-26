"""Remote GPU worker backend: `LYRE_WORKER=remote`.

Same contract as `worker.acestep_worker`, so everything around it is unchanged:
`worker/run_worker.py` still claims jobs from the SQLite queue under the GPU
lease, heartbeats them, and `server.jobs` still writes the take. The difference
is where ACE-Step runs. `run_job` sends the job to a `worker.remote_host` over
HTTPS, polls until it is rendered, downloads the audio into the take directory,
and acknowledges it.

The host is whatever `server.remote_gpu.connection` currently names, read
fresh on every call, so starting or connecting a Remote GPU from the UI takes
effect without restarting this process. With nothing connected the worker
reports itself unavailable and the server refuses new jobs with that reason.

Style packs are not offered remotely yet: this backend has no `train_lora`, so
`run_worker` publishes that capability as unsupported, and a job that asks for
a trained LoRA fails with a message saying so.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from server.remote_gpu import client, connection
from worker.remote_host import build_id

POLL_INTERVAL_SEC = 2.0
# Polls may fail while a proxy hiccups or a pod restarts its container. Give
# up only after this long without a single good answer.
MAX_UNREACHABLE_SEC = 180.0

NOT_CONNECTED = (
    "no Remote GPU connected: start one or enter a host address in the Remote GPU "
    "panel (see docs/remote-gpu.md)"
)


class RemoteJobError(RuntimeError):
    pass


def _connection() -> connection.Connection:
    current = connection.get_connection()
    if current is None:
        raise RemoteJobError(NOT_CONNECTED)
    return current


def _health() -> dict[str, Any] | None:
    current = connection.get_connection()
    if current is None:
        return None
    return client.probe(current)


def current_readiness() -> tuple[bool, str]:
    """Live readiness, checked on every `run_worker` heartbeat instead of
    trusting what was true at startup: a Remote GPU comes and goes while this
    process keeps running."""
    health = _health()
    if health is None:
        return False, NOT_CONNECTED
    if not health.get("connected"):
        return False, f"Remote GPU unreachable: {health.get('error')}"
    message = str(health.get("message") or "")
    if not health.get("ready"):
        return False, f"Remote GPU starting: {message}"
    stale = health.get("build") != build_id()
    gpu = health.get("gpu") or "remote"
    note = " (host runs an older build than this checkout)" if stale else ""
    return True, f"Remote GPU ready: {gpu}{note}"


def initialize_worker() -> tuple[bool, str]:
    return current_readiness()


def get_loaded_dit_profile() -> str | None:
    health = _health()
    if not health or not health.get("connected") or not health.get("ready"):
        return None
    loaded = health.get("loaded_dit_profile")
    return loaded if isinstance(loaded, str) else None


def supports_dit_profile(dit_profile: str) -> tuple[bool, str | None]:
    health = _health()
    if not health or not health.get("connected"):
        return False, NOT_CONNECTED
    capability = (health.get("capabilities") or {}).get(dit_profile)
    if not isinstance(capability, dict):
        # A host that has not reported yet is given the benefit of the doubt;
        # the job itself fails cleanly if it really cannot load the profile.
        return True, None
    return bool(capability.get("supported")), capability.get("reason")


def _submission(job: dict[str, Any], plan: dict[str, Any], take_id: str) -> dict[str, Any]:
    if job.get("lora_id") or job.get("lora_adapter_path"):
        raise RemoteJobError(
            "style packs are not available on a Remote GPU yet; generate without one, "
            "or switch back to the local worker"
        )
    body: dict[str, Any] = {
        # The take id is unique per attempt, which makes it a safe idempotency
        # key: a retried submit whose first response was lost never renders twice.
        "client_id": take_id,
        "take_id": take_id,
        "job": {k: v for k, v in job.items() if k != "src_audio"},
        "plan": plan,
    }
    src_audio = job.get("src_audio")
    if src_audio:
        path = Path(src_audio)
        body["src_audio_b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        body["src_audio_suffix"] = path.suffix.lower()
    return body


def _submit(remote: client.RemoteClient, body: dict[str, Any]) -> str:
    deadline = time.monotonic() + MAX_UNREACHABLE_SEC
    while True:
        try:
            return remote.submit(body)
        except client.RemoteGpuError as exc:
            # A 4xx is an answer, not an outage: the host understood and refused.
            if exc.status is not None and 400 <= exc.status < 500 and exc.status != 429:
                raise RemoteJobError(str(exc)) from None
            if time.monotonic() > deadline:
                raise RemoteJobError(f"could not submit to the Remote GPU: {exc}") from None
        time.sleep(POLL_INTERVAL_SEC)


def _wait(
    remote: client.RemoteClient,
    token: str,
    on_dit_loaded: Callable[[str], None] | None,
) -> dict[str, Any]:
    announced: str | None = None
    last_answer = time.monotonic()
    while True:
        try:
            status = remote.job(token)
            last_answer = time.monotonic()
        except client.RemoteGpuError as exc:
            if exc.status == 404:
                raise RemoteJobError(
                    "the Remote GPU no longer knows this job (it restarted or was replaced)"
                ) from None
            if time.monotonic() - last_answer > MAX_UNREACHABLE_SEC:
                raise RemoteJobError(f"lost contact with the Remote GPU: {exc}") from None
            time.sleep(POLL_INTERVAL_SEC)
            continue
        loaded = status.get("dit_loaded")
        if on_dit_loaded is not None and isinstance(loaded, str) and loaded != announced:
            on_dit_loaded(loaded)
            announced = loaded
        if status.get("status") == "done":
            result = status.get("result")
            if not isinstance(result, dict):
                raise RemoteJobError("the Remote GPU reported done without a result")
            return result
        if status.get("status") == "error":
            raise RemoteJobError(f"Remote GPU: {status.get('error') or 'job failed'}")
        time.sleep(POLL_INTERVAL_SEC)


def run_job(
    job: dict[str, Any],
    plan: dict[str, Any],
    take_id: str,
    take_dir: Path,
    on_dit_loaded: Callable[[str], None] | None = None,
) -> tuple[dict, dict | None, str | None]:
    """Render one job on the connected Remote GPU. Returns `(take_meta,
    plan_patch, lrc_text)` exactly as the local backends do."""
    remote = client.RemoteClient(_connection())
    token = _submit(remote, _submission(job, plan, take_id))
    result = _wait(remote, token, on_dit_loaded)

    audio_name = result.get("audio_name")
    if audio_name not in ("mix.wav", "mix.mp3"):
        raise RemoteJobError(f"the Remote GPU returned an unexpected audio file: {audio_name!r}")
    try:
        audio = remote.audio(token)
    except client.RemoteGpuError as exc:
        raise RemoteJobError(f"could not download the take: {exc}") from None
    expected = result.get("audio_bytes")
    if isinstance(expected, int) and expected != len(audio):
        raise RemoteJobError(f"the take download was cut short ({len(audio)} of {expected} bytes)")
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / audio_name).write_bytes(audio)
    try:
        remote.ack(token)
    except client.RemoteGpuError:
        pass  # the host frees unacknowledged results on its own TTL

    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise RemoteJobError("the Remote GPU returned no take metadata")
    # The host never names takes; this process allocated the id.
    meta = {**meta, "id": take_id}
    plan_patch = result.get("plan_patch")
    lrc_text = result.get("lrc_text")
    return (
        meta,
        plan_patch if isinstance(plan_patch, dict) else None,
        lrc_text if isinstance(lrc_text, str) else None,
    )
