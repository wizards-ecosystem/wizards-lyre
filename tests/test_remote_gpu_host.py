"""The Remote GPU host's protocol (`worker.remote_host`), GPU-free.

The host runs against `worker.mock_worker`, so these exercise the real HTTP
surface, auth, and job lifecycle without ACE-Step, torch, or a network.
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worker import mock_worker
from worker.remote_host import build_id, runtime
from worker.remote_host.app import SECRET_HEADER, Host, create_app

SECRET = "s" * 40
AUTH = {SECRET_HEADER: SECRET}


@pytest.fixture()
def host_client(tmp_path: Path) -> Iterator[TestClient]:
    host = Host(lambda: mock_worker, jobs_dir=tmp_path / "jobs")
    host.start()
    with TestClient(create_app(host, SECRET)) as test_client:
        yield test_client


def _job(action: str = "generate", **extra) -> dict:
    return {"action": action, "dit_profile": "iterate", "seed": 7, **extra}


def _submit(test_client: TestClient, client_id: str = "take1", **body) -> str:
    payload = {
        "client_id": client_id,
        "take_id": client_id,
        "job": _job(),
        "plan": {"caption": "calm piano", "lyrics": "[Instrumental]"},
        **body,
    }
    response = test_client.post("/jobs", json=payload, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _wait_done(test_client: TestClient, token: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = test_client.get(f"/jobs/{token}", headers=AUTH).json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.01)
    raise TimeoutError(token)


def test_every_route_requires_the_secret(host_client: TestClient) -> None:
    assert host_client.get("/health").status_code == 401
    assert host_client.get("/health", headers={SECRET_HEADER: "wrong"}).status_code == 401
    assert host_client.post("/jobs", json={}).status_code == 401
    assert host_client.get("/health", headers=AUTH).status_code == 200


def test_health_reports_build_and_readiness(host_client: TestClient) -> None:
    deadline = time.time() + 5
    while time.time() < deadline:
        health = host_client.get("/health", headers=AUTH).json()
        if health["ready"]:
            break
        time.sleep(0.01)
    assert health["ready"] is True
    assert health["build"] == build_id()
    assert health["capabilities"]["iterate"]["supported"] is True
    assert "generate" in health["actions"]


def test_a_job_renders_downloads_and_is_freed_on_ack(host_client: TestClient) -> None:
    token = _submit(host_client)
    status = _wait_done(host_client, token)
    assert status["status"] == "done", status
    assert status["dit_loaded"] == "iterate"
    result = status["result"]
    assert result["audio_name"] == "mix.wav"
    assert result["meta"]["id"] == "take1"

    audio = host_client.get(f"/jobs/{token}/audio", headers=AUTH)
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"
    assert len(audio.content) == result["audio_bytes"]

    # A lost response is recovered by polling again: reads never drop a result.
    assert host_client.get(f"/jobs/{token}", headers=AUTH).json()["status"] == "done"

    assert host_client.post(f"/jobs/{token}/ack", headers=AUTH).json() == {"ok": True}
    assert host_client.get(f"/jobs/{token}", headers=AUTH).status_code == 404


def test_a_retried_submit_never_renders_twice(host_client: TestClient) -> None:
    first = _submit(host_client, client_id="same")
    second = _submit(host_client, client_id="same")
    assert first == second


def test_source_audio_travels_with_the_job(host_client: TestClient) -> None:
    source = base64.b64encode(b"RIFF" + b"\x00" * 64).decode()
    token = _submit(
        host_client,
        job=_job("cover", source_take_id="src"),
        src_audio_b64=source,
        src_audio_suffix=".wav",
    )
    status = _wait_done(host_client, token)
    assert status["status"] == "done", status
    assert status["result"]["meta"]["task_type"] == "cover"


@pytest.mark.parametrize(
    ("job", "extra", "detail"),
    [
        (_job(lora_id="abc"), {}, "style packs"),
        (_job("train_lora"), {}, "not available"),
        (_job(dit_profile="huge"), {}, "dit_profile"),
        (_job(), {"src_audio_b64": "AAAA", "src_audio_suffix": ".exe"}, "format"),
        (_job(), {"src_audio_b64": "not base64!", "src_audio_suffix": ".wav"}, "base64"),
    ],
)
def test_the_host_refuses_what_it_cannot_render(
    host_client: TestClient, job: dict, extra: dict, detail: str
) -> None:
    response = host_client.post(
        "/jobs",
        json={"client_id": "x", "take_id": "x", "job": job, "plan": {}, **extra},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert detail in response.text


def test_ids_cannot_carry_paths(host_client: TestClient) -> None:
    response = host_client.post(
        "/jobs",
        json={"client_id": "../x", "take_id": "x", "job": _job(), "plan": {}},
        headers=AUTH,
    )
    assert response.status_code == 422


def test_an_oversized_body_is_refused_before_it_is_read(host_client: TestClient) -> None:
    response = host_client.post(
        "/jobs",
        content=b"{}",
        headers={**AUTH, "Content-Length": str(10**12), "Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_a_failing_job_reports_its_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(mock_worker, "run_job", broken)
    host = Host(lambda: mock_worker, jobs_dir=tmp_path / "jobs")
    host.start()
    with TestClient(create_app(host, SECRET)) as test_client:
        status = _wait_done(test_client, _submit(test_client))
    assert status["status"] == "error"
    assert "CUDA out of memory" in status["error"]


@pytest.mark.parametrize("secret", ["", "changeme", "short-but-random"])
def test_the_host_refuses_a_weak_secret(monkeypatch: pytest.MonkeyPatch, secret: str) -> None:
    monkeypatch.setenv("LYRE_REMOTE_SECRET", secret)
    with pytest.raises(ValueError, match="LYRE_REMOTE_SECRET"):
        runtime.shared_secret()


def test_build_id_is_a_stable_fingerprint() -> None:
    assert re.fullmatch(r"[0-9a-f]{12}", build_id())
    assert build_id() == build_id()


@pytest.mark.parametrize(
    "relative",
    ["ACE_STEP_REVISION", "requirements/ace-step-security.txt", "docker/remote-gpu/Dockerfile"],
)
def test_build_id_changes_with_what_goes_into_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    # Published tags are immutable, so a changed image must get a new id even
    # when no Python source changed.
    import worker.remote_host as remote_host

    for name in remote_host._BUILD_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original\n")
    monkeypatch.setattr(remote_host, "_ROOT", tmp_path)
    before = build_id()
    (tmp_path / relative).write_text("changed\n")
    assert build_id() != before
