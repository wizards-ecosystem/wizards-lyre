"""The Remote GPU lane's smaller parts: the real HTTP transport (over loopback
only), the host's model fetching and startup, and its runtime layout."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from server.remote_gpu import client as remote_client
from server.remote_gpu.connection import Connection
from worker.remote_host import __main__ as host_main
from worker.remote_host import models, runtime

SECRET = "t" * 40


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _reply(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.headers.get("X-Lyre-Secret") != SECRET:
            self._reply(401, b'{"detail": "bad secret"}')
        elif self.path == "/health":
            self._reply(200, json.dumps({"ready": True, "build": "x"}).encode())
        elif self.path == "/redirect":
            self._reply(302, b"", {"Location": "http://127.0.0.1:1/steal"})
        elif self.path == "/big":
            self._reply(200, b"x" * 2048)
        else:
            self._reply(404, b'{"detail": "unknown job token"}')


@pytest.fixture()
def loopback_host() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_the_real_transport_talks_http(loopback_host: str) -> None:
    remote = remote_client.RemoteClient(Connection(loopback_host, SECRET))
    assert remote.health()["ready"] is True

    with pytest.raises(remote_client.RemoteGpuError) as missing:
        remote.job("nope")
    assert missing.value.status == 404
    assert "unknown job token" in str(missing.value)

    wrong = remote_client.RemoteClient(Connection(loopback_host, "w" * 40))
    with pytest.raises(remote_client.RemoteGpuError, match="rejected the shared secret"):
        wrong.health()


def test_the_transport_never_follows_a_redirect(loopback_host: str) -> None:
    status, _ = remote_client.urllib_transport(
        "GET", f"{loopback_host}/redirect", {"X-Lyre-Secret": SECRET}, None, 5.0, 1024
    )
    assert status == 302


def test_the_transport_caps_response_size(loopback_host: str) -> None:
    with pytest.raises(remote_client.RemoteGpuError, match="more than 1024 bytes"):
        remote_client.urllib_transport(
            "GET", f"{loopback_host}/big", {"X-Lyre-Secret": SECRET}, None, 5.0, 1024
        )


def test_an_unreachable_host_is_a_status_not_a_crash() -> None:
    health = remote_client.probe(Connection("http://127.0.0.1:9", SECRET), max_age=0)
    assert health["connected"] is False
    assert "unreachable" in health["error"]


def test_profiles_download_only_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[list[str]] = []

    def fake_run(args, check, env):
        runs.append(args)
        root = Path(args[args.index("--dir") + 1])
        if "--model" in args:
            target = root / args[args.index("--model") + 1]
        else:
            (root / models.CORE_LM).mkdir(parents=True, exist_ok=True)
            target = root / models.PROFILE_CHECKPOINTS["iterate"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(models.shutil, "which", lambda _name: "/bin/acestep-download")
    monkeypatch.setattr(models.subprocess, "run", fake_run)

    models.ensure_profile("iterate", tmp_path)
    models.ensure_profile("studio_ops", tmp_path)
    models.ensure_profile("studio_ops", tmp_path)
    assert runs == [
        ["/bin/acestep-download", "--dir", str(tmp_path)],
        [
            "/bin/acestep-download",
            "--dir",
            str(tmp_path),
            "--model",
            "acestep-v15-base",
            "--skip-main",
        ],
    ]


def test_a_failed_download_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda _name: "/bin/acestep-download")
    monkeypatch.setattr(
        models.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 3)
    )
    with pytest.raises(RuntimeError, match="exit 3"):
        models.ensure_profile("iterate", tmp_path)


def test_downloads_need_the_upstream_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="acestep-download"):
        models.ensure_profile("polish", tmp_path)
    with pytest.raises(ValueError, match="unknown dit_profile"):
        models.ensure_profile("huge", tmp_path)


def test_prepare_keeps_every_cache_under_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("CHECKPOINTS", "JOBS", "CACHE", "TMP"):
        monkeypatch.setattr(runtime, name, tmp_path / name.lower())
    for name in ("LYRE_CHECKPOINTS_DIR", "HF_HOME", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    runtime.prepare()
    import os

    assert os.environ["LYRE_CHECKPOINTS_DIR"] == str(tmp_path / "checkpoints")
    assert os.environ["HF_HOME"].startswith(str(tmp_path))
    assert (tmp_path / "jobs").is_dir()
    assert runtime.free_disk_gb(tmp_path) > 0


def test_host_settings_default_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LYRE_REMOTE_HOST", raising=False)
    monkeypatch.delenv("LYRE_REMOTE_PORT", raising=False)
    monkeypatch.delenv("LYRE_REMOTE_BACKEND", raising=False)
    assert runtime.host() == "127.0.0.1"
    assert runtime.port() == 8000
    assert runtime.backend_name() == "acestep"
    monkeypatch.setenv("LYRE_REMOTE_SECRET", "a-real-random-secret-value-0123456789")
    assert runtime.shared_secret().startswith("a-real")


def test_the_host_will_not_start_without_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(runtime, "prepare", lambda: None)
    monkeypatch.setenv("LYRE_REMOTE_SECRET", "changeme")
    with pytest.raises(SystemExit) as exited:
        host_main.main()
    assert exited.value.code == 2
    assert "LYRE_REMOTE_SECRET" in capsys.readouterr().err


def test_the_host_starts_its_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    served: dict = {}
    monkeypatch.setattr(runtime, "prepare", lambda: None)
    monkeypatch.setattr(runtime, "JOBS", tmp_path / "jobs")
    monkeypatch.setenv("LYRE_REMOTE_SECRET", "a-real-random-secret-value-0123456789")
    monkeypatch.setenv("LYRE_REMOTE_BACKEND", "mock")
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: served.update(host=host, port=port))
    host_main.main()
    assert served == {"host": "127.0.0.1", "port": 8000}


def test_only_known_backends_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRE_REMOTE_BACKEND", "mock")
    assert host_main._load_backend().__name__ == "worker.mock_worker"
    monkeypatch.setenv("LYRE_REMOTE_BACKEND", "somethingelse")
    with pytest.raises(ValueError, match="LYRE_REMOTE_BACKEND"):
        host_main._load_backend()
