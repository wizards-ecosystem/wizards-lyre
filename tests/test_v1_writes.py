"""SPEC.md sec 15 phase 8: `/v1` writes and Idempotency-Key.

No GPU. The worker is the mock. `/api` stays on its own routes and its
own error shape.
"""

from __future__ import annotations

import asyncio
import io
import wave
from contextlib import closing

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import jobs as jobs_module
from server import storage
from server.api_keys import create_key
from server.app import app
from server.jobs.db import _connect
from server.v1.idempotency import (
    IdempotencyConflict,
    IdempotencyMiddleware,
    _http_error,
    owner_for,
    release,
    reserve,
)
from tests.helpers import wait_for_job

_ACTIONS = ("generate", "cover", "repaint", "extract", "lego", "complete", "train_lora")
_SIGNATURE = (
    "task_type",
    "dit_profile",
    "seed",
    "caption",
    "lyrics",
    "error",
    "favorite",
    "notes",
    "track_name",
)


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 400)
    return buf.getvalue()


def _plan(client: TestClient, project_id: str, *, v1: bool) -> None:
    body = {"query": "rain", "caption": "wet streets", "lyrics": "neon", "duration_sec": 30}
    path = f"/v1/projects/{project_id}/plan" if v1 else f"/api/projects/{project_id}/plan"
    response = client.put(path, json=body)
    assert response.status_code == 200, response.text


def _project(client: TestClient, title: str, *, v1: bool) -> str:
    if v1:
        created = client.post("/v1/projects", json={"title": title, "query": "rain"})
        assert created.status_code == 200, created.text
    else:
        created = client.post("/api/projects", json={"title": title, "query": "rain"})
        assert created.status_code == 200, created.text
    project_id = created.json()["id"]
    _plan(client, project_id, v1=v1)
    return project_id


def _enqueue(client: TestClient, project_id: str, body: dict, *, v1: bool) -> str:
    path = f"/v1/projects/{project_id}/jobs" if v1 else f"/api/projects/{project_id}/jobs"
    response = client.post(path, json=body)
    assert response.status_code in (200, 202), response.text
    if v1:
        assert response.status_code == 202
        assert response.headers["location"] == f"/v1/jobs/{response.json()['id']}"
    return response.json()["id"]


def _signature(take: dict) -> dict:
    return {key: take.get(key) for key in _SIGNATURE}


def test_plan_round_trip_and_project_writes(client: TestClient) -> None:
    project_id = _project(client, "Agent", v1=True)
    detail = client.get(f"/v1/projects/{project_id}").json()
    assert detail["project"]["title"] == "Agent"
    assert detail["plan"]["caption"] == "wet streets"
    assert detail["plan"]["lyrics"] == "neon"
    assert detail["plan"]["query"] == "rain"
    assert detail["takes"] == []

    renamed = client.patch(
        f"/v1/projects/{project_id}",
        json={"title": "Renamed", "favorite": True, "dit_profile": "polish"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == "Renamed"
    assert renamed.json()["favorite"] is True
    assert renamed.json()["dit_profile"] == "polish"

    bad_profile = client.patch(f"/v1/projects/{project_id}", json={"dit_profile": "nope"})
    assert bad_profile.status_code == 400
    assert bad_profile.json()["error"]["code"] == "invalid_request"

    bad_plan = client.put(f"/v1/projects/{project_id}/plan", json={"duration_sec": "long"})
    assert bad_plan.status_code == 400
    assert bad_plan.json()["error"]["code"] == "invalid_plan"
    assert bad_plan.json()["error"]["param"] == "duration_sec"
    assert client.get(f"/v1/projects/{project_id}").json()["plan"]["caption"] == "wet streets"

    missing = client.put("/v1/projects/missing/plan", json={"caption": "x"})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert "detail" not in missing.json()


def test_each_action_matches_the_api_take(client: TestClient) -> None:
    """Enqueue every live action through `/v1` and through `/api`. The takes
    (and, for train_lora, the style pack) come from the same enqueue function,
    so the recorded result matches."""
    v1_id = _project(client, "Via v1", v1=True)
    api_id = _project(client, "Via api", v1=False)
    source_body = {"action": "generate", "seed": 11}
    v1_source = wait_for_job(client, _enqueue(client, v1_id, source_body, v1=True))
    api_source = wait_for_job(client, _enqueue(client, api_id, source_body, v1=False))
    assert v1_source["status"] == "done"
    assert api_source["status"] == "done"

    v1_take = client.get(f"/v1/projects/{v1_id}/takes").json()["data"]
    api_take = client.get(f"/api/projects/{api_id}/takes").json()
    assert _signature(v1_take[0]) == _signature(api_take[0])
    assert v1_take[0]["parent_take_id"] is None

    shaped = {
        "cover": {"action": "cover", "seed": 12},
        "repaint": {"action": "repaint", "seed": 13, "repainting_start": 0, "repainting_end": 1},
        "extract": {"action": "extract", "seed": 14, "track_name": "vocals"},
        "lego": {"action": "lego", "seed": 15, "track_name": "drums"},
        "complete": {"action": "complete", "seed": 16, "track_name": "bass"},
    }
    for action, body in shaped.items():
        v1_body = {**body, "source_take_id": v1_source["take_id"]}
        api_body = {**body, "source_take_id": api_source["take_id"]}
        v1_done = wait_for_job(client, _enqueue(client, v1_id, v1_body, v1=True))
        api_done = wait_for_job(client, _enqueue(client, api_id, api_body, v1=False))
        assert v1_done["status"] == "done", v1_done
        assert api_done["status"] == "done", api_done
        v1_meta = storage.get_take(v1_id, v1_done["take_id"])
        api_meta = storage.get_take(api_id, api_done["take_id"])
        assert _signature(v1_meta) == _signature(api_meta)
        assert v1_meta["parent_take_id"] == v1_source["take_id"]
        assert v1_meta["task_type"]
        job = client.get(f"/v1/jobs/{v1_done['id']}").json()
        assert job["action"] == action
        assert job["status"] == "done"
        assert job["take_id"] == v1_done["take_id"]

    v1_sources = [v1_source["take_id"]]
    api_sources = [api_source["take_id"]]
    for _ in range(7):
        v1_sources.append(
            wait_for_job(
                client, _enqueue(client, v1_id, {"action": "generate", "seed": 3}, v1=True)
            )["take_id"]
        )
        api_sources.append(
            wait_for_job(
                client, _enqueue(client, api_id, {"action": "generate", "seed": 3}, v1=False)
            )["take_id"]
        )
    train = {"action": "train_lora", "name": "night pack", "source_take_ids": v1_sources}
    v1_train = wait_for_job(client, _enqueue(client, v1_id, train, v1=True), timeout=10)
    api_train = wait_for_job(
        client,
        _enqueue(
            client,
            api_id,
            {**train, "source_take_ids": api_sources},
            v1=False,
        ),
        timeout=10,
    )
    assert v1_train["status"] == "done", v1_train
    assert api_train["status"] == "done", api_train
    v1_loras = client.get(f"/v1/projects/{v1_id}/loras")
    api_loras = client.get(f"/api/projects/{api_id}/loras")
    assert v1_loras.status_code == 200
    assert v1_loras.json()[0]["name"] == api_loras.json()[0]["name"] == "night pack"
    assert v1_loras.json()[0]["source_take_count"] == 8
    assert v1_loras.json()[0]["id"] == v1_train["lora_id"]
    assert set(_ACTIONS) >= {"generate", "train_lora"}


def test_upload_annotation_active_take_export_and_delete(client: TestClient) -> None:
    project_id = _project(client, "Studio", v1=True)
    uploaded = client.post(
        f"/v1/projects/{project_id}/uploads",
        files={"file": ("clip.wav", _wav(), "audio/wav")},
    )
    assert uploaded.status_code == 200, uploaded.text
    upload_path = uploaded.json()["upload_path"]
    assert upload_path.startswith("uploads/") and upload_path.endswith(".wav")
    resolved = storage.resolve_upload_path(project_id, upload_path)
    assert resolved.is_relative_to(storage.project_dir(project_id))

    escaped = client.post(
        f"/v1/projects/{project_id}/uploads",
        files={"file": ("../../etc/passwd.wav", _wav(), "audio/wav")},
    )
    assert escaped.status_code == 200, escaped.text
    jailed = storage.resolve_upload_path(project_id, escaped.json()["upload_path"])
    assert jailed.is_relative_to(storage.uploads_dir(project_id))

    rejected = client.post(
        f"/v1/projects/{project_id}/uploads",
        files={"file": ("notes.txt", b"nope", "text/plain")},
    )
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_request"

    covered = wait_for_job(
        client,
        _enqueue(
            client,
            project_id,
            {"action": "cover", "upload_path": upload_path, "seed": 4},
            v1=True,
        ),
    )
    assert covered["status"] == "done"
    take_id = covered["take_id"]
    noted = client.patch(
        f"/v1/projects/{project_id}/takes/{take_id}",
        json={"favorite": True, "notes": "keep"},
    )
    assert noted.status_code == 200, noted.text
    assert noted.json()["favorite"] is True
    assert noted.json()["notes"] == "keep"
    assert noted.json()["seed"] == storage.get_take(project_id, take_id)["seed"]

    active = client.post(f"/v1/projects/{project_id}/active_take", json={"take_id": take_id})
    assert active.status_code == 200, active.text
    assert active.json()["active_take_id"] == take_id

    failed_dir = storage.take_dir(project_id, "failed-take")
    failed_dir.mkdir(parents=True, exist_ok=True)
    storage.write_take_meta(
        project_id,
        "failed-take",
        {"id": "failed-take", "error": "nope", "created_at": "2000-01-01T00:00:00+00:00"},
    )
    refused = client.post(f"/v1/projects/{project_id}/active_take", json={"take_id": "failed-take"})
    assert refused.status_code == 400
    assert "cannot set active" in refused.json()["error"]["message"]

    exported = client.get(f"/v1/projects/{project_id}/export")
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/zip")
    assert exported.content[:2] == b"PK"
    quiet = client.get(f"/v1/projects/{project_id}/export", params={"include_stems": False})
    assert quiet.status_code == 200
    assert quiet.content[:2] == b"PK"


def test_delete_cancels_a_queued_job(api_client: TestClient) -> None:
    project_id = api_client.post("/v1/projects", json={"title": "Gone"}).json()["id"]
    queued = api_client.post(f"/v1/projects/{project_id}/jobs", json={"action": "generate"}).json()
    assert queued["status"] == "queued"
    removed = api_client.delete(f"/v1/projects/{project_id}")
    assert removed.status_code == 204
    row = jobs_module.get_job(queued["id"])
    assert row["status"] == "error"
    assert row["error"] == "project deleted"
    assert api_client.get(f"/v1/projects/{project_id}").status_code == 404
    assert api_client.delete("/v1/projects/missing").status_code == 404


def test_idempotent_replay_and_conflict(api_client: TestClient) -> None:
    headers = {"Idempotency-Key": "project-1"}
    first = api_client.post("/v1/projects", json={"title": "Once"}, headers=headers)
    assert first.status_code == 200, first.text
    again = api_client.post("/v1/projects", json={"title": "Once"}, headers=headers)
    assert again.status_code == 200
    assert again.content == first.content
    assert again.json()["id"] == first.json()["id"]
    titles = [row["title"] for row in api_client.get("/v1/projects").json()["data"]]
    assert titles.count("Once") == 1

    conflict = api_client.post("/v1/projects", json={"title": "Other"}, headers=headers)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert conflict.json()["error"]["param"] == "Idempotency-Key"
    assert "detail" not in conflict.json()
    titles = [row["title"] for row in api_client.get("/v1/projects").json()["data"]]
    assert "Other" not in titles

    project_id = first.json()["id"]
    job_headers = {"Idempotency-Key": "job-1", "Content-Type": "application/json"}
    raw = b'{"action":"generate","seed":9}'
    queued = api_client.post(f"/v1/projects/{project_id}/jobs", content=raw, headers=job_headers)
    assert queued.status_code == 202, queued.text
    assert queued.headers["location"] == f"/v1/jobs/{queued.json()['id']}"
    replay = api_client.post(f"/v1/projects/{project_id}/jobs", content=raw, headers=job_headers)
    assert replay.status_code == 200
    assert replay.content == queued.content
    assert replay.headers["location"] == queued.headers["location"]
    listed = api_client.get("/v1/jobs", params={"project_id": project_id}).json()["data"]
    assert [row["id"] for row in listed] == [queued.json()["id"]]

    spaced = b'{"action": "generate", "seed": 9}'
    different = api_client.post(
        f"/v1/projects/{project_id}/jobs", content=spaced, headers=job_headers
    )
    assert different.status_code == 409
    assert len(api_client.get("/v1/jobs", params={"project_id": project_id}).json()["data"]) == 1

    queried = api_client.post(
        f"/v1/projects/{project_id}/jobs?trace=1", content=raw, headers=job_headers
    )
    assert queried.status_code == 409

    bare = {"Idempotency-Key": "cancel-1"}
    cancelled = api_client.post(f"/v1/jobs/{queued.json()['id']}/cancel", headers=bare)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    calls = {"n": 0}
    real = jobs_module.cancel_job

    def _spy(job_id: str) -> dict:
        calls["n"] += 1
        return real(job_id)

    jobs_module.cancel_job = _spy  # type: ignore[method-assign]
    try:
        second = api_client.post(f"/v1/jobs/{queued.json()['id']}/cancel", headers=bare)
    finally:
        jobs_module.cancel_job = real  # type: ignore[method-assign]
    assert second.status_code == 200
    assert second.content == cancelled.content
    assert calls["n"] == 0

    # A key is optional. Two posts without one create two projects.
    one = api_client.post("/v1/projects", json={"title": "A"}).json()["id"]
    two = api_client.post("/v1/projects", json={"title": "A"}).json()["id"]
    assert one != two

    # `/api` does not honor the header.
    api_headers = {"Idempotency-Key": "api-same"}
    api_one = api_client.post("/api/projects", json={"title": "Api"}, headers=api_headers).json()
    api_two = api_client.post("/api/projects", json={"title": "Api"}, headers=api_headers).json()
    assert api_one["id"] != api_two["id"]


def test_idempotency_stores_errors_and_expires(api_client: TestClient) -> None:
    project_id = api_client.post("/v1/projects", json={"title": "Err"}).json()["id"]
    headers = {"Idempotency-Key": "bad-action"}
    calls = {"n": 0}
    real = jobs_module.enqueue_job

    def _count(project: str, body: dict) -> dict:
        calls["n"] += 1
        return real(project, body)

    jobs_module.enqueue_job = _count  # type: ignore[method-assign]
    try:
        first = api_client.post(
            f"/v1/projects/{project_id}/jobs", json={"action": "nope"}, headers=headers
        )
        second = api_client.post(
            f"/v1/projects/{project_id}/jobs", json={"action": "nope"}, headers=headers
        )
    finally:
        jobs_module.enqueue_job = real  # type: ignore[method-assign]
    assert first.status_code == 400
    assert second.status_code == 400
    assert second.content == first.content
    assert calls["n"] == 1
    clash = api_client.post(
        f"/v1/projects/{project_id}/jobs", json={"action": "generate"}, headers=headers
    )
    assert clash.status_code == 409

    fresh = {"Idempotency-Key": "expires"}
    created = api_client.post("/v1/projects", json={"title": "Temp"}, headers=fresh)
    assert created.status_code == 200
    with closing(_connect()) as conn:
        conn.execute("UPDATE idempotency_keys SET expires_at = 0 WHERE idempotency_key = 'expires'")
        conn.commit()
    replaced = api_client.post("/v1/projects", json={"title": "After"}, headers=fresh)
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["id"] != created.json()["id"]


def test_a_failed_request_can_retry_the_same_key(api_client: TestClient, monkeypatch) -> None:
    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("disk")

    monkeypatch.setattr(storage, "create_project", _boom)
    headers = {"Idempotency-Key": "retry"}
    with pytest.raises(RuntimeError):
        api_client.post("/v1/projects", json={"title": "X"}, headers=headers)
    with pytest.raises(RuntimeError):
        api_client.post("/v1/projects", json={"title": "X"}, headers=headers)
    assert calls["n"] == 2


def test_invalid_key_and_in_progress(api_client: TestClient, lyre_env) -> None:
    project_id = api_client.post("/v1/projects", json={"title": "Key"}).json()["id"]
    bad = api_client.post(
        f"/v1/projects/{project_id}/jobs",
        json={"action": "generate"},
        headers={"Idempotency-Key": "has space"},
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["param"] == "Idempotency-Key"
    empty = api_client.post(
        "/v1/projects",
        json={"title": "Z"},
        headers={"Idempotency-Key": "   "},
    )
    assert empty.status_code == 400
    huge = api_client.post(
        "/v1/projects",
        json={"title": "Z"},
        headers={"Idempotency-Key": "k" * 256},
    )
    assert huge.status_code == 400

    assert reserve("", "held", "hash-a") is None
    with pytest.raises(IdempotencyConflict, match="in progress"):
        reserve("", "held", "hash-a")
    with pytest.raises(IdempotencyConflict, match="different body"):
        reserve("", "held", "hash-b")
    release("", "held")
    assert reserve("", "held", "hash-b") is None


def test_oversized_upload_with_a_key_is_not_stored(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 10)
    project_id = api_client.post("/v1/projects", json={"title": "Big"}).json()["id"]
    headers = {"Idempotency-Key": "upload-1"}
    oversized = api_client.post(
        f"/v1/projects/{project_id}/uploads",
        files={"file": ("big.wav", b"0" * 200_000, "audio/wav")},
        headers=headers,
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "payload_too_large"
    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
    followed = api_client.post(
        f"/v1/projects/{project_id}/uploads",
        files={"file": ("clip.wav", _wav(), "audio/wav")},
        headers=headers,
    )
    # The multipart body differs from the rejected upload, and the 413 was
    # not stored, so this is a new request rather than a replay of the error.
    assert followed.status_code == 200, followed.text


def test_v1_validation_errors_use_the_error_schema(api_client: TestClient) -> None:
    project_id = api_client.post("/v1/projects", json={"title": "Typed"}).json()["id"]
    missing = api_client.post(f"/v1/projects/{project_id}/jobs", json={})
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "invalid_request"
    assert missing.json()["error"]["param"] == "action"
    assert "detail" not in missing.json()

    strength = api_client.post(
        f"/v1/projects/{project_id}/jobs",
        json={"action": "generate", "audio_cover_strength": 4},
    )
    assert strength.status_code == 400
    assert strength.json()["error"]["param"] == "audio_cover_strength"

    api_strength = api_client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "audio_cover_strength": 4},
    )
    assert api_strength.status_code == 422
    assert "detail" in api_strength.json()

    bad_limit = api_client.get("/v1/projects", params={"limit": "abc"})
    assert bad_limit.status_code == 400
    assert bad_limit.json()["error"]["code"] == "invalid_request"
    api_limit = api_client.get("/api/jobs", params={"limit": "abc"})
    assert api_limit.status_code == 422
    assert "detail" in api_limit.json()


def test_auth_on_and_off_for_writes(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Auth off: a write needs no header.
    opened = api_client.post("/v1/projects", json={"title": "Open"})
    assert opened.status_code == 200

    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    reader, read_secret = create_key("reader", ["read"])
    _generator, generate_secret = create_key("generator", ["generate"])
    _writer, write_secret = create_key("writer", ["write"])
    read_headers = {"Authorization": f"Bearer {read_secret}"}
    generate_headers = {"Authorization": f"Bearer {generate_secret}"}
    write_headers = {"Authorization": f"Bearer {write_secret}"}

    missing = api_client.post("/v1/projects", json={"title": "Nope"})
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"

    # An unauthenticated attempt does not consume the idempotency key.
    key_headers = {"Idempotency-Key": "shared", "Authorization": f"Bearer {write_secret}"}
    denied = api_client.post(
        "/v1/projects",
        json={"title": "Later"},
        headers={"Idempotency-Key": "shared"},
    )
    assert denied.status_code == 401
    created = api_client.post("/v1/projects", json={"title": "Later"}, headers=key_headers)
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]

    assert (
        api_client.post("/v1/projects", json={"title": "R"}, headers=read_headers).status_code
        == 403
    )
    assert (
        api_client.post("/v1/projects", json={"title": "G"}, headers=generate_headers).status_code
        == 403
    )
    assert (
        api_client.put(
            f"/v1/projects/{project_id}/plan", json={"caption": "x"}, headers=generate_headers
        ).status_code
        == 403
    )
    plan = api_client.put(
        f"/v1/projects/{project_id}/plan", json={"caption": "kept"}, headers=write_headers
    )
    assert plan.status_code == 200, plan.text
    assert (
        api_client.get(f"/v1/projects/{project_id}", headers=read_headers).json()["plan"]["caption"]
        == "kept"
    )

    forbidden_job = api_client.post(
        f"/v1/projects/{project_id}/jobs",
        json={"action": "generate"},
        headers=write_headers,
    )
    assert forbidden_job.status_code == 403
    queued = api_client.post(
        f"/v1/projects/{project_id}/jobs",
        json={"action": "generate"},
        headers=generate_headers,
    )
    assert queued.status_code == 202, queued.text
    assert (
        api_client.get(f"/v1/jobs/{queued.json()['id']}", headers=generate_headers).status_code
        == 200
    )
    assert (
        api_client.get(f"/v1/projects/{project_id}/loras", headers=read_headers).status_code == 200
    )
    assert (
        api_client.get(f"/v1/projects/{project_id}/export", headers=read_headers).status_code == 200
    )
    assert (
        api_client.get(f"/v1/projects/{project_id}/export", headers=generate_headers).status_code
        == 403
    )
    assert api_client.delete(f"/v1/projects/{project_id}", headers=read_headers).status_code == 403
    assert api_client.delete(f"/v1/projects/{project_id}", headers=write_headers).status_code == 204

    # Distinct API keys do not share an idempotency slot.
    other = {"Idempotency-Key": "per-key", **write_headers}
    other_reader = {"Idempotency-Key": "per-key", **read_headers}
    assert api_client.get("/v1/health", headers=other_reader).status_code == 200
    first_key = api_client.post("/v1/projects", json={"title": "K1"}, headers=other)
    # The reader cannot create, so give the same key string to a second writer.
    _other_writer, other_secret = create_key("writer-2", ["write"])
    second_key = api_client.post(
        "/v1/projects",
        json={"title": "K2"},
        headers={
            "Idempotency-Key": "per-key",
            "Authorization": f"Bearer {other_secret}",
        },
    )
    assert first_key.status_code == 200
    assert second_key.status_code == 200
    assert first_key.json()["id"] != second_key.json()["id"]
    assert reader["id"]
    assert api_client.get("/api/health").status_code == 404


def test_owner_for_fails_closed(lyre_env, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Headers

    monkeypatch.delenv("LYRE_REQUIRE_AUTH", raising=False)
    assert owner_for(Headers({"idempotency-key": "a"})) == ""

    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "nope")
    assert owner_for(Headers()) is None

    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    assert owner_for(Headers()) is None
    assert owner_for(Headers({"authorization": "Basic abc"})) is None
    assert owner_for(Headers({"authorization": "Bearer lyre_missing"})) is None
    _public, secret = create_key("owner", ["read"])
    assert owner_for(Headers({"authorization": f"Bearer {secret}"}))
    from server.api_keys import keys_path

    keys_path().write_text("{", encoding="utf-8")
    assert owner_for(Headers({"authorization": f"Bearer {secret}"})) is None


def test_http_500_does_not_stick(lyre_env) -> None:
    jobs_module.init_db()
    calls = {"n": 0}

    async def _app(scope, receive, send):
        del scope, receive
        calls["n"] += 1
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"{}", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = IdempotencyMiddleware(_app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/projects",
        "query_string": b"",
        "headers": [(b"idempotency-key", b"five-hundred")],
    }

    async def _receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def _run() -> None:
        for _ in range(2):

            async def _send(message):
                del message

            await middleware(scope, _receive, _send)

    asyncio.run(_run())
    assert calls["n"] == 2

    response = _http_error(HTTPException(status_code=400, detail="bad"))
    assert response.status_code == 400
    not_found = _http_error(HTTPException(status_code=404, detail="missing"))
    assert not_found.status_code == 404


def test_openapi_schemas_cover_the_new_operations() -> None:
    spec = app.openapi()
    paths = spec["paths"]
    expected = {
        ("post", "/v1/projects"),
        ("patch", "/v1/projects/{project_id}"),
        ("delete", "/v1/projects/{project_id}"),
        ("put", "/v1/projects/{project_id}/plan"),
        ("post", "/v1/projects/{project_id}/active_take"),
        ("patch", "/v1/projects/{project_id}/takes/{take_id}"),
        ("post", "/v1/projects/{project_id}/uploads"),
        ("get", "/v1/projects/{project_id}/loras"),
        ("get", "/v1/projects/{project_id}/export"),
        ("post", "/v1/projects/{project_id}/jobs"),
    }
    for method, path in expected:
        assert method in paths[path]

    for path, methods in paths.items():
        if not path.startswith("/v1"):
            continue
        for method, operation in methods.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            success = next(code for code in operation["responses"] if code.startswith("2"))
            body = operation["responses"][success]
            if success == "204":
                assert "content" not in body
            else:
                assert any("schema" in media for media in body["content"].values())
            if method == "post":
                names = [param["name"].lower() for param in operation.get("parameters", [])]
                assert "idempotency-key" in names

    job_post = paths["/v1/projects/{project_id}/jobs"]["post"]
    assert "202" in job_post["responses"]
    assert "200" in job_post["responses"]
    assert "Location" in job_post["responses"]["202"]["headers"]
    assert "JobCreate" in spec["components"]["schemas"]
    assert "PlanWrite" in spec["components"]["schemas"]
    assert "Error" in spec["components"]["schemas"]
