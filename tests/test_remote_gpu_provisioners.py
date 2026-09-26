"""Provisioners, against a fake Runpod API and registry. No network, no spend.

These pin the behaviour that protects the user's money and secrets: nothing
starts without a request, a second start is refused, a session is recorded
before it is reported, stop terminates (never merely stops), a failed stop
keeps the record, and the API key never leaves this machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.remote_gpu import client as remote_client
from server.remote_gpu import connection, provisioners
from server.remote_gpu.provisioners import runpod
from server.remote_gpu.provisioners.base import ProvisionerError
from worker.remote_host import build_id

API_KEY = "rpa_TESTKEY_do_not_leak_0123456789"


class FakeRunpod:
    """Answers Runpod REST v2 and GHCR the way the real services do, and
    records every request so tests can assert on bodies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.published = True
        self.pod: dict[str, Any] | None = None
        self.delete_status = 204

    def __call__(self, method, url, headers, body, timeout, max_bytes):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, headers, payload))
        if url.startswith("https://ghcr.io/token"):
            return 200, b'{"token": "anon"}'
        if url.startswith("https://ghcr.io/v2/"):
            return (200 if self.published else 404), b""
        if url.startswith(f"{runpod.API}/catalog/gpus"):
            catalog = {
                "gpus": [
                    {
                        "id": runpod.DEFAULT_GPU_TYPE,
                        "memory": 24,
                        "availability": "HIGH",
                        "price": {"secure": 0.27, "community": 0.16},
                    }
                ]
            }
            return 200, json.dumps(catalog).encode()
        if url == f"{runpod.API}/pods" and method == "POST":
            self.pod = {"id": "pod123", "status": "PROVISIONING", "cost": 0.27, "runtime": None}
            return 200, json.dumps({**self.pod, "gpu": {"id": payload["gpu"]["id"]}}).encode()
        if url == f"{runpod.API}/pods/pod123":
            if method == "DELETE":
                if self.delete_status < 300:
                    self.pod = None
                return self.delete_status, b"" if self.delete_status < 300 else b"boom"
            if self.pod is None:
                return 404, b'{"detail": "not found"}'
            return 200, json.dumps(self.pod).encode()
        # Anything else (the pod's own proxy URL) is unreachable in tests.
        raise OSError(f"no route to {url}")

    def requests(self, method: str, prefix: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == method and c[1].startswith(prefix)]


@pytest.fixture()
def fake_runpod(lyre_env: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRunpod]:
    fake = FakeRunpod()
    monkeypatch.setattr(remote_client, "TRANSPORT", fake)
    monkeypatch.setenv("LYRE_REMOTE_GPU_PROVISIONER", "runpod")
    monkeypatch.setenv("LYRE_RUNPOD_API_KEY", API_KEY)
    provisioners._CACHE.clear()
    remote_client.forget_probe()
    yield fake
    provisioners._CACHE.clear()
    remote_client.forget_probe()


def test_every_request_names_itself(fake_runpod: FakeRunpod) -> None:
    # Cloudflare in front of Runpod rejects Python's default agent (error 1010).
    runpod.RunpodProvisioner().quote()
    assert fake_runpod.calls
    assert all(
        headers.get("User-Agent", "").startswith("wizards-lyre/")
        for _, _, headers, _ in fake_runpod.calls
    )


def test_status_shows_the_price_before_anything_starts(fake_runpod: FakeRunpod) -> None:
    with TestClient(app) as test_client:
        status = test_client.get("/api/remote-gpu").json()
    assert status["provisioner"] == "runpod"
    assert status["can_provision"] is True
    assert status["session"] is None
    assert status["quote"]["gpu"] == runpod.DEFAULT_GPU_TYPE
    assert status["quote"]["hourly_usd"] == 0.27
    assert fake_runpod.requests("POST", f"{runpod.API}/pods") == []


def test_start_creates_one_pod_and_records_it_first(fake_runpod: FakeRunpod) -> None:
    with TestClient(app) as test_client:
        started = test_client.post("/api/remote-gpu/session")
        assert started.status_code == 200, started.text
        body = started.json()

        [(_, _, headers, pod)] = fake_runpod.requests("POST", f"{runpod.API}/pods")
        assert headers["Authorization"] == f"Bearer {API_KEY}"
        assert pod["image"] == f"ghcr.io/{runpod.PUBLISHED_REPO}:{build_id()}"
        assert pod["gpu"] == {
            "id": runpod.DEFAULT_GPU_TYPE,
            "count": 1,
            "minCudaVersion": "12.8",
        }
        assert pod["ports"] == ["8000/http"]
        assert pod["env"][runpod.MARKER_ENV] == "1"
        # The key stays here; the pod gets a secret made for this session only.
        assert API_KEY not in json.dumps(pod)
        saved = connection.get_connection()
        assert saved is not None
        assert saved.base_url == "https://pod123-8000.proxy.runpod.net"
        assert pod["env"]["LYRE_REMOTE_SECRET"] == saved.secret
        assert connection.get_session()["id"] == "pod123"

        assert body["session"]["state"] == "starting"
        assert body["session"]["hourly_usd"] == 0.27
        assert saved.secret not in json.dumps(body)

        again = test_client.post("/api/remote-gpu/session")
        assert again.status_code == 409
        assert len(fake_runpod.requests("POST", f"{runpod.API}/pods")) == 1

        # Connecting another host while paying for this one is refused too.
        other = test_client.put(
            "/api/remote-gpu/connection",
            json={"base_url": "https://elsewhere.test", "secret": "x" * 40},
        )
        assert other.status_code == 409


def test_a_running_pod_is_not_ready_until_its_host_answers(fake_runpod: FakeRunpod) -> None:
    with TestClient(app) as test_client:
        test_client.post("/api/remote-gpu/session")
        fake_runpod.pod = {"id": "pod123", "status": "RUNNING", "runtime": None, "cost": 0.27}
        pulling = test_client.get("/api/remote-gpu").json()["session"]
        assert pulling["state"] == "starting"
        assert "pulling" in pulling["detail"]

        fake_runpod.pod["runtime"] = {"uptime": 5}
        up = test_client.get("/api/remote-gpu").json()
        assert up["session"]["state"] == "starting"  # host is still unreachable
        assert up["host"]["connected"] is False


def test_a_pod_that_vanished_does_not_block_the_next_start(fake_runpod: FakeRunpod) -> None:
    provisioner = runpod.RunpodProvisioner()
    provisioner.start()
    fake_runpod.pod = None  # terminated in the Runpod console
    assert provisioner.status() is None
    assert connection.get_session() is None and connection.get_connection() is None
    provisioner.start()
    assert len(fake_runpod.requests("POST", f"{runpod.API}/pods")) == 2


def test_an_exited_pod_still_blocks_a_second_start(fake_runpod: FakeRunpod) -> None:
    provisioner = runpod.RunpodProvisioner()
    provisioner.start()
    fake_runpod.pod = {"id": "pod123", "status": "EXITED", "runtime": None}
    assert provisioner.status().state == "error"
    # An exited pod can still bill its disk: it must be stopped, not replaced.
    with pytest.raises(ProvisionerError, match="already running"):
        provisioner.start()


def test_stop_terminates_and_forgets(fake_runpod: FakeRunpod) -> None:
    with TestClient(app) as test_client:
        test_client.post("/api/remote-gpu/session")
        stopped = test_client.delete("/api/remote-gpu/session").json()
    assert stopped["session"] is None and stopped["connection"] is None
    assert len(fake_runpod.requests("DELETE", f"{runpod.API}/pods/pod123")) == 1
    # Terminate is DELETE; the stop action would keep billing the disk.
    assert fake_runpod.requests("POST", f"{runpod.API}/pods/pod123/action") == []
    assert connection.get_session() is None


def test_a_failed_stop_keeps_the_record_so_it_can_be_retried(
    fake_runpod: FakeRunpod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_REMOTE_GPU_STOP_ON_EXIT", "false")
    fake_runpod.delete_status = 500
    with TestClient(app) as test_client:
        test_client.post("/api/remote-gpu/session")
        failed = test_client.delete("/api/remote-gpu/session")
        assert failed.status_code == 400
        assert connection.get_session() is not None

        fake_runpod.delete_status = 204
        assert test_client.delete("/api/remote-gpu/session").status_code == 200
    assert connection.get_session() is None


def test_a_clean_exit_terminates_the_pod(fake_runpod: FakeRunpod) -> None:
    with TestClient(app) as test_client:
        test_client.post("/api/remote-gpu/session")
    assert len(fake_runpod.requests("DELETE", f"{runpod.API}/pods/pod123")) == 1
    assert connection.get_session() is None


def test_stop_on_exit_can_be_turned_off(
    fake_runpod: FakeRunpod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_REMOTE_GPU_STOP_ON_EXIT", "false")
    with TestClient(app) as test_client:
        test_client.post("/api/remote-gpu/session")
    assert fake_runpod.requests("DELETE", f"{runpod.API}/pods") == []
    assert connection.get_session()["id"] == "pod123"


def test_startup_adopts_but_never_starts(fake_runpod: FakeRunpod) -> None:
    with TestClient(app):
        pass
    assert fake_runpod.requests("POST", f"{runpod.API}/pods") == []


def test_an_unpublished_image_is_refused_before_any_spend(fake_runpod: FakeRunpod) -> None:
    fake_runpod.published = False
    with TestClient(app) as test_client:
        refused = test_client.post("/api/remote-gpu/session")
    assert refused.status_code == 400
    assert "remote-image" in refused.json()["detail"]
    assert fake_runpod.requests("POST", f"{runpod.API}/pods") == []


def test_a_custom_image_skips_the_registry_check(
    fake_runpod: FakeRunpod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_RUNPOD_IMAGE", "registry.example/me/lyre-host:abc")
    monkeypatch.setenv("LYRE_RUNPOD_NETWORK_VOLUME_ID", "vol1")
    runpod.RunpodProvisioner().start()
    [(_, _, _, pod)] = fake_runpod.requests("POST", f"{runpod.API}/pods")
    assert pod["image"] == "registry.example/me/lyre-host:abc"
    assert pod["mounts"] == {"network": [{"volumeId": "vol1", "path": "/workspace"}]}
    assert pod["env"]["LYRE_REMOTE_ROOT"] == "/workspace/wizards-lyre"
    assert fake_runpod.requests("GET", "https://ghcr.io") == []


def test_provider_errors_never_echo_the_key(
    fake_runpod: FakeRunpod, monkeypatch: pytest.MonkeyPatch
) -> None:
    def echo(method, url, headers, body, timeout, max_bytes):
        return 500, f"bad header {headers['Authorization']}".encode()

    monkeypatch.setattr(remote_client, "TRANSPORT", echo)
    with pytest.raises(ProvisionerError) as caught:
        runpod.RunpodProvisioner()._request("GET", "/pods")
    assert API_KEY not in str(caught.value)


def test_a_missing_key_is_explained(fake_runpod: FakeRunpod, monkeypatch) -> None:
    monkeypatch.delenv("LYRE_RUNPOD_API_KEY")
    provisioners._CACHE.clear()
    with TestClient(app) as test_client:
        status = test_client.get("/api/remote-gpu").json()
        assert status["can_provision"] is False
        refused = test_client.post("/api/remote-gpu/session")
    assert refused.status_code == 400
    assert "LYRE_RUNPOD_API_KEY" in refused.json()["detail"]


def test_an_unknown_provisioner_falls_back_to_manual(
    lyre_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_REMOTE_GPU_PROVISIONER", "somecloud")
    provisioners._CACHE.clear()
    assert provisioners.get_provisioner().id == "manual"
