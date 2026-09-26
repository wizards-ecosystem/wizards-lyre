"""The Remote GPU lane end to end, GPU- and network-free.

The local stack runs with `LYRE_WORKER=remote`: the real server, the real
worker loop claiming from SQLite, and `worker.remote_backend` speaking HTTP.
`server.remote_gpu.client.TRANSPORT` is replaced so that HTTP lands in an
in-process `worker.remote_host` running the mock backend.
"""

from __future__ import annotations

import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.remote_gpu import client as remote_client
from server.remote_gpu import connection
from tests.helpers import wait_for_job
from worker import mock_worker, remote_backend
from worker.remote_host.app import Host, create_app
from worker.run_worker import run_loop

SECRET = "r" * 48
BASE_URL = "https://gpu.example.test"


@pytest.fixture(autouse=True)
def _fresh_probe_cache() -> Iterator[None]:
    remote_client.forget_probe()
    yield
    remote_client.forget_probe()


@pytest.fixture()
def remote_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An in-process host, reachable at BASE_URL through the patched transport."""
    host = Host(lambda: mock_worker, jobs_dir=tmp_path / "remote-jobs")
    host.start()
    with TestClient(create_app(host, SECRET)) as host_client:

        def transport(method, url, headers, body, timeout, max_bytes):
            if not url.startswith(BASE_URL):
                raise OSError(f"no route to {url}")
            response = host_client.request(
                method, url[len(BASE_URL) :], headers=headers, content=body
            )
            return response.status_code, response.content

        monkeypatch.setattr(remote_client, "TRANSPORT", transport)
        monkeypatch.setattr(remote_backend, "POLL_INTERVAL_SEC", 0.01)
        yield host_client


@pytest.fixture()
def remote_stack(
    lyre_env: Path, monkeypatch: pytest.MonkeyPatch, remote_host: TestClient
) -> Iterator[TestClient]:
    monkeypatch.setenv("LYRE_WORKER", "remote")
    connection.set_connection(BASE_URL, SECRET)
    stop_event = threading.Event()
    worker = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker.start()
    try:
        with TestClient(app) as test_client:
            _wait_until_ready(test_client)
            yield test_client
    finally:
        stop_event.set()
        worker.join(timeout=5)


def _wait_until_ready(test_client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        gpu = test_client.get("/api/health").json()["gpu"]
        if gpu.startswith("Remote GPU ready"):
            return
        remote_client.forget_probe()
        time.sleep(0.02)
    raise TimeoutError(f"worker never reported the Remote GPU ready: {gpu}")


def _project(test_client: TestClient) -> str:
    project = test_client.post("/api/projects", json={"title": "Remote"}).json()
    test_client.put(
        f"/api/projects/{project['id']}/plan",
        json={"caption": "warm synth pads", "lyrics": "[Instrumental]", "duration": 30},
    )
    return project["id"]


def test_a_generate_job_renders_on_the_remote_gpu(remote_stack: TestClient) -> None:
    project_id = _project(remote_stack)
    job = remote_stack.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    done = wait_for_job(remote_stack, job["id"], timeout=10)
    assert done["status"] == "done", done

    takes = remote_stack.get(f"/api/projects/{project_id}/takes").json()
    assert [t["id"] for t in takes] == [done["take_id"]]
    assert takes[0]["error"] is None
    audio = remote_stack.get(f"/api/projects/{project_id}/takes/{done['take_id']}/audio")
    assert audio.status_code == 200
    assert audio.content[:4] == b"RIFF"


def test_a_cover_sends_its_source_take(remote_stack: TestClient) -> None:
    project_id = _project(remote_stack)
    first = remote_stack.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    source = wait_for_job(remote_stack, first.json()["id"], timeout=10)
    cover = remote_stack.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "cover", "source_take_id": source["take_id"]},
    )
    done = wait_for_job(remote_stack, cover.json()["id"], timeout=10)
    assert done["status"] == "done", done
    takes = {t["id"]: t for t in remote_stack.get(f"/api/projects/{project_id}/takes").json()}
    assert takes[done["take_id"]]["parent_take_id"] == source["take_id"]
    assert takes[done["take_id"]]["task_type"] == "cover"


def test_style_pack_training_is_not_offered_remotely(remote_stack: TestClient) -> None:
    project_id = _project(remote_stack)
    response = remote_stack.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "train_lora", "source_take_ids": ["a"] * 8, "name": "pack"},
    )
    assert response.status_code == 400


def test_without_a_connection_the_worker_reports_why(
    lyre_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_WORKER", "remote")
    stop_event = threading.Event()
    worker = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker.start()
    try:
        with TestClient(app) as test_client:
            deadline = time.time() + 5
            while time.time() < deadline:
                gpu = test_client.get("/api/health").json()["gpu"]
                if "no Remote GPU connected" in gpu:
                    break
                time.sleep(0.02)
            assert "no Remote GPU connected" in gpu
            # Same contract as a local worker that failed to start: the job
            # still queues, then fails with the reason on the job and the take.
            project_id = _project(test_client)
            job = test_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
            failed = wait_for_job(test_client, job.json()["id"])
            assert failed["status"] == "error"
            assert "no Remote GPU connected" in failed["error"]
    finally:
        stop_event.set()
        worker.join(timeout=5)


def test_a_host_that_forgot_the_job_fails_it_cleanly(
    lyre_env: Path, remote_host: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    connection.set_connection(BASE_URL, SECRET)

    def forgetful(self, token):
        raise remote_client.RemoteGpuError("unknown job token", 404)

    monkeypatch.setattr(remote_client.RemoteClient, "job", forgetful)
    with pytest.raises(remote_backend.RemoteJobError, match="no longer knows"):
        remote_backend.run_job(
            job={"action": "generate", "dit_profile": "iterate"},
            plan={"caption": "x"},
            take_id="t1",
            take_dir=tmp_path / "take",
        )


def test_a_lora_job_fails_before_any_network_call(lyre_env: Path, tmp_path: Path) -> None:
    connection.set_connection(BASE_URL, SECRET)
    with pytest.raises(remote_backend.RemoteJobError, match="style packs"):
        remote_backend.run_job(
            job={"action": "generate", "dit_profile": "studio_ops", "lora_id": "l1"},
            plan={},
            take_id="t1",
            take_dir=tmp_path / "take",
        )


def test_the_status_route_joins_connection_and_host(
    lyre_env: Path, remote_host: TestClient
) -> None:
    with TestClient(app) as test_client:
        empty = test_client.get("/api/remote-gpu").json()
        assert empty["provisioner"] == "manual"
        assert empty["can_provision"] is False
        assert empty["connection"] is None and empty["host"] is None

        saved = test_client.put(
            "/api/remote-gpu/connection", json={"base_url": BASE_URL + "/", "secret": SECRET}
        ).json()
        assert saved["connection"] == {"base_url": BASE_URL}
        assert saved["host"]["connected"] is True
        assert saved["host"]["stale_build"] is False
        assert SECRET not in str(saved)

        cleared = test_client.delete("/api/remote-gpu/connection").json()
        assert cleared["connection"] is None


def test_manual_start_explains_itself(api_client: TestClient) -> None:
    response = api_client.post("/api/remote-gpu/session")
    assert response.status_code == 400
    assert "LYRE_REMOTE_GPU_PROVISIONER" in response.json()["detail"]


@pytest.mark.parametrize(
    "url",
    [
        "http://gpu.example.test",
        "ftp://gpu.example.test",
        "https://user:pw@gpu.example.test",
        "https://gpu.example.test/api",
        "https://gpu.example.test?x=1",
        "gpu.example.test",
    ],
)
def test_the_secret_is_only_sent_somewhere_safe(api_client: TestClient, url: str) -> None:
    response = api_client.put(
        "/api/remote-gpu/connection", json={"base_url": url, "secret": SECRET}
    )
    assert response.status_code == 400


def test_loopback_http_is_allowed_for_a_tunnelled_host(lyre_env: Path) -> None:
    saved = connection.set_connection("http://127.0.0.1:8000", SECRET)
    assert saved.base_url == "http://127.0.0.1:8000"


def test_a_short_secret_is_refused(lyre_env: Path) -> None:
    with pytest.raises(ValueError, match="24 characters"):
        connection.set_connection(BASE_URL, "short")


def test_the_connection_file_is_private(lyre_env: Path) -> None:
    connection.set_connection(BASE_URL, SECRET)
    mode = stat.S_IMODE(connection.path().stat().st_mode)
    assert mode == 0o600


def test_errors_never_carry_the_secret(lyre_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky(method, url, headers, body, timeout, max_bytes):
        raise OSError(f"connection reset while sending {headers}")

    monkeypatch.setattr(remote_client, "TRANSPORT", leaky)
    health = remote_client.probe(connection.set_connection(BASE_URL, SECRET))
    assert health["connected"] is False
    assert SECRET not in health["error"]
