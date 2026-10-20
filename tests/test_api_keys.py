"""SPEC.md sec 15.2: opt-in API keys. Default is loopback with auth off.

These tests configure the app in-process. They do not open a socket.
"""

from __future__ import annotations

import stat

import pytest
from fastapi.testclient import TestClient

from server import config
from server.api_keys import authenticate, create_key, keys_path, main, revoke_key


def test_default_is_loopback_without_auth(api_client: TestClient, lyre_env) -> None:
    assert config.validate_startup() == "127.0.0.1"
    assert config.auth_required() is False
    assert api_client.get("/v1/health").status_code == 200
    assert api_client.get("/api/health").status_code == 200
    root = api_client.get("/")
    assert root.status_code == 200


def test_network_bind_refuses_to_start_without_a_key(
    lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_BIND", "192.0.2.10")
    with pytest.raises(config.BindConfigError, match="API key"):
        config.validate_startup()


def test_require_auth_zero_on_a_network_bind_is_a_startup_error(
    lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_BIND", "192.0.2.10")
    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "0")
    with pytest.raises(config.BindConfigError, match="LYRE_REQUIRE_AUTH"):
        config.validate_startup()


def test_loopback_require_auth_without_a_key_refuses_to_start(
    lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    with pytest.raises(config.BindConfigError, match="API key"):
        config.validate_startup()


def test_a_stored_key_lets_a_network_bind_start(lyre_env, monkeypatch: pytest.MonkeyPatch) -> None:
    create_key("cursor", ["read"])
    monkeypatch.setenv("LYRE_BIND", "192.0.2.10")
    assert config.validate_startup() == "192.0.2.10"
    mode = keys_path().stat().st_mode & 0o777
    assert mode == stat.S_IRUSR | stat.S_IWUSR
    text = keys_path().read_text(encoding="utf-8")
    assert "sha256" in text
    assert "lyre_" not in text


def test_auth_on_hides_the_studio_and_checks_scopes(
    api_client: TestClient, lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    read_only, read_secret = create_key("reader", ["read"])
    generate, generate_secret = create_key("generator", ["generate"])
    writer, write_secret = create_key("writer", ["write"])

    assert config.validate_startup() == "127.0.0.1"

    assert api_client.get("/").status_code == 404
    assert api_client.get("/api/health").status_code == 404
    assert api_client.get("/docs").status_code == 404

    missing = api_client.get("/v1/health")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["error"]["code"] == "unauthorized"

    unknown = api_client.get("/v1/health", headers={"Authorization": "Bearer lyre_nope"})
    assert unknown.status_code == 401

    read_headers = {"Authorization": f"Bearer {read_secret}"}
    assert api_client.get("/v1/health", headers=read_headers).status_code == 200
    project_id = None
    # Project creation is still an /api route, hidden while auth is on.
    # Seed a project through the service, then read it with the key.
    from server import storage

    project_id = storage.create_project(title="Scoped").get("id")
    assert api_client.get(f"/v1/projects/{project_id}", headers=read_headers).status_code == 200

    from server import jobs as jobs_module

    job = jobs_module.enqueue_job(project_id, {"action": "generate"})
    forbidden = api_client.post(f"/v1/jobs/{job['id']}/cancel", headers=read_headers)
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"
    assert jobs_module.get_job(job["id"])["status"] == "queued"

    generate_headers = {"Authorization": f"Bearer {generate_secret}"}
    assert api_client.get("/v1/projects", headers=generate_headers).status_code == 403
    assert api_client.get(f"/v1/jobs/{job['id']}", headers=generate_headers).status_code == 200
    cancelled = api_client.post(f"/v1/jobs/{job['id']}/cancel", headers=generate_headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    write_headers = {"Authorization": f"Bearer {write_secret}"}
    assert api_client.get("/v1/health", headers=write_headers).status_code == 403

    revoke_key(read_only["id"])
    assert authenticate(read_secret) is None
    assert api_client.get("/v1/health", headers=read_headers).status_code == 401
    assert generate["id"] != writer["id"]


def test_cli_prints_the_secret_once(lyre_env, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["create", "--name", "cursor"]) == 0
    out = capsys.readouterr().out
    secret = next(
        line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("secret: ")
    )
    key_id = next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("id: "))
    assert secret.startswith("lyre_")
    assert "scopes: read, generate, write" in out
    assert secret not in keys_path().read_text(encoding="utf-8")
    assert authenticate(secret) is not None
    assert main(["revoke", key_id]) == 0
    assert authenticate(secret) is None
    assert main(["revoke", key_id]) == 1


def test_corrupt_key_file_fails_closed(
    api_client: TestClient, lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys_path().write_text("{", encoding="utf-8")
    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    with pytest.raises(config.BindConfigError, match="cannot be read"):
        config.validate_startup()

    keys_path().unlink()
    create_key("ok")
    keys_path().write_text("not-json", encoding="utf-8")
    response = api_client.get("/v1/health", headers={"Authorization": "Bearer lyre_x"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "misconfigured"


def test_cli_rejects_a_bad_name_or_scope(lyre_env, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["create", "--name", "  "]) == 1
    assert "name" in capsys.readouterr().err
    assert main(["create", "--name", "ok", "--scopes", "read,nope"]) == 1
    assert "unknown scope" in capsys.readouterr().err
    assert main(["create", "--name", "ok", "--scopes", "read,read"]) == 0
    secret = next(
        line.split(": ", 1)[1]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("secret: ")
    )
    record = authenticate(secret)
    assert record is not None
    assert record["scopes"] == ["read"]


def test_a_non_string_hash_is_skipped(lyre_env) -> None:
    import json

    _public, secret = create_key("cursor")
    path = keys_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["keys"][0]["sha256"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert authenticate(secret) is None


def test_localhost_and_ipv6_loopback_are_not_network_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYRE_BIND", "localhost")
    assert config.is_loopback() is True
    assert config.auth_required() is False
    monkeypatch.setenv("LYRE_BIND", "::1")
    assert config.is_loopback() is True
    monkeypatch.setenv("LYRE_BIND", "[::1]")
    assert config.bind_host() == "::1"
    assert config.is_loopback() is True
