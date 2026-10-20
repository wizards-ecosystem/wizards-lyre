"""SPEC.md sec 15, first slice: typed /v1 reads, the job object, cancel, OpenAPI.

No GPU. The worker is the mock. Auth-off is the default; key checks live in
test_api_keys.py.
"""

from __future__ import annotations

import threading
import time
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.v1.present import job_object
from tests.helpers import wait_for_job
from worker.run_worker import run_loop

_READS = (
    "/v1/health",
    "/v1/projects",
    "/v1/projects/{project_id}",
    "/v1/projects/{project_id}/takes",
    "/v1/jobs",
    "/v1/jobs/{job_id}",
)
_STATUSES = ("queued", "running", "done", "error", "cancelled")


def _wait_v1(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.01)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_v1_reads_round_trip_the_studio_records(client: TestClient) -> None:
    created = client.post("/api/projects", json={"title": "Agent", "query": "rain"}).json()
    project_id = created["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={"caption": "wet streets", "lyrics": "[Verse]\nneon", "duration_sec": 30},
    )
    detail = client.get(f"/v1/projects/{project_id}").json()
    assert detail["project"]["id"] == project_id
    assert detail["plan"]["caption"] == "wet streets"
    assert detail["takes"] == []

    queued = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()

    health = client.get("/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["backend"] == "mock"
    assert "queue" in body and set(body["queue"]) == {"queued", "running"}

    page = client.get("/v1/projects", params={"limit": 1}).json()
    assert page["data"][0]["id"] == project_id
    assert page["data"][0]["title"] == "Agent"
    assert set(page["data"][0]) == {"id", "title", "updated_at", "active_take_id", "favorite"}
    assert page["next_cursor"] in (None, "1")

    missing = client.get("/v1/projects/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert "detail" not in missing.json()

    # /api keeps its own error shape.
    api_missing = client.get("/api/projects/does-not-exist")
    assert api_missing.status_code == 404
    assert "detail" in api_missing.json()

    done = wait_for_job(client, queued["id"])
    assert done["status"] == "done"
    job = client.get(f"/v1/jobs/{queued['id']}").json()
    assert job["status"] == "done"
    assert job["stage"] == "writing_take"
    assert job["take_id"]
    assert job["error"] is None
    assert job["seed"] not in (None, -1)
    assert any(url.endswith("/mix.wav") for url in job["artifacts"])
    audio = client.get(job["artifacts"][0])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/")

    takes = client.get(f"/v1/projects/{project_id}/takes").json()
    assert takes["data"][0]["id"] == job["take_id"]
    assert takes["next_cursor"] is None

    listed = client.get("/v1/jobs", params={"project_id": project_id, "limit": 10}).json()
    assert listed["data"][0]["id"] == job["id"]

    bad_limit = client.get("/v1/projects", params={"limit": 1000})
    assert bad_limit.status_code == 400
    assert bad_limit.json()["error"]["param"] == "limit"
    bad_cursor = client.get("/v1/jobs", params={"cursor": "nope"})
    assert bad_cursor.status_code == 400
    assert bad_cursor.json()["error"]["code"] == "invalid_request"


def test_openapi_pins_a_schema_on_every_v1_operation() -> None:
    spec = app.openapi()
    error = spec["components"]["schemas"]["Error"]
    assert "error" in error["properties"]
    nested = error["properties"]["error"]
    # A $ref or an inline object; either way the component exists and is the
    # shared error body, not /api's detail string.
    assert "detail" not in error["properties"]
    assert nested

    for path in _READS:
        operation = spec["paths"][path]["get"]
        schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema
    cancel = spec["paths"]["/v1/jobs/{job_id}/cancel"]["post"]
    assert cancel["responses"]["200"]["content"]["application/json"]["schema"]
    job_schema = spec["components"]["schemas"]["Job"]
    assert set(job_schema["properties"]) >= {
        "id",
        "project_id",
        "action",
        "dit_profile",
        "status",
        "stage",
        "take_id",
        "error",
        "seed",
        "created_at",
        "updated_at",
        "artifacts",
    }
    assert set(job_schema["properties"]["status"]["enum"]) == set(_STATUSES)

    for name in ("mix.wav", "mix.mp3", "lyrics.lrc"):
        operation = spec["paths"][f"/v1/artifacts/{{project_id}}/{{take_id}}/{name}"]["get"]
        content = operation["responses"]["200"]["content"]
        assert any("schema" in media for media in content.values())


def test_job_object_covers_every_state(api_client: TestClient) -> None:
    project_id = api_client.post("/api/projects", json={"title": "States"}).json()["id"]
    queued = api_client.post(
        f"/api/projects/{project_id}/jobs", json={"action": "generate", "seed": 7}
    ).json()
    view = api_client.get(f"/v1/jobs/{queued['id']}").json()
    assert view["status"] == "queued"
    assert view["stage"] is None
    assert view["take_id"] is None
    assert view["error"] is None
    assert view["artifacts"] == []
    assert view["seed"] == 7

    # Running: claim, then observe the row through /v1 before it finishes.
    from server import jobs as jobs_module

    claimed = jobs_module.claim_next_queued_job()
    assert claimed is not None
    jobs_module.set_stage(claimed["id"], "rendering")
    running = api_client.get(f"/v1/jobs/{claimed['id']}").json()
    assert running["status"] == "running"
    assert running["stage"] == "rendering"
    assert running["take_id"] is None

    # Error: mark the running row the way a failed worker would.
    jobs_module._set_status(claimed["id"], "error", error="synthetic")
    failed = api_client.get(f"/v1/jobs/{claimed['id']}").json()
    assert failed["status"] == "error"
    assert failed["error"] == "synthetic"
    assert failed["artifacts"] == []

    # Cancelled, built directly so the schema accepts it with nulls.
    cancelled = job_object(
        {
            **jobs_module.read_job(claimed["id"]),
            "status": "cancelled",
            "error": "should be cleared",
            "take_id": "ignored",
            "stage": None,
        }
    )
    assert cancelled["status"] == "cancelled"
    assert cancelled["error"] is None
    assert cancelled["take_id"] is None
    assert set(cancelled) == set(view)


def test_queued_cancel_never_calls_the_worker(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worker.mock_worker as mock_worker_module

    called: list[int] = []
    real = mock_worker_module.run_job

    def spy(*args, **kwargs):
        called.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(mock_worker_module, "run_job", spy)
    project_id = api_client.post("/api/projects", json={"title": "Cancel queued"}).json()["id"]
    job = api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    cancelled = api_client.post(f"/v1/jobs/{job['id']}/cancel")
    assert cancelled.status_code == 200
    body = cancelled.json()
    assert body["status"] == "cancelled"
    assert body["error"] is None
    assert body["take_id"] is None

    from server import jobs as jobs_module

    assert jobs_module.process_one_queued_job() is False
    assert called == []
    assert api_client.get(f"/v1/jobs/{job['id']}").json()["status"] == "cancelled"
    # A second cancel is a no-op on the terminal row.
    again = api_client.post(f"/v1/jobs/{job['id']}/cancel")
    assert again.json()["status"] == "cancelled"


def test_cancel_during_render_discards_the_take(lyre_env, monkeypatch: pytest.MonkeyPatch) -> None:
    import worker.mock_worker as mock_worker_module

    entered = threading.Event()
    release = threading.Event()
    real = mock_worker_module.run_job

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(mock_worker_module, "run_job", blocking)
    stop = threading.Event()
    worker = threading.Thread(target=run_loop, args=(stop, 0.01), daemon=True)
    worker.start()
    try:
        with TestClient(app) as client:
            project_id = client.post("/api/projects", json={"title": "Drop"}).json()["id"]
            job_id = client.post(
                f"/api/projects/{project_id}/jobs", json={"action": "generate"}
            ).json()["id"]
            assert entered.wait(5)
            response = client.post(f"/v1/jobs/{job_id}/cancel")
            assert response.status_code == 200
            assert response.json()["status"] == "running"
            release.set()
            final = _wait_v1(client, job_id)
            assert final["status"] == "cancelled"
            assert final["take_id"] is None
            assert final["error"] is None
            assert final["artifacts"] == []
            assert client.get(f"/v1/projects/{project_id}/takes").json()["data"] == []
    finally:
        release.set()
        stop.set()
        worker.join(timeout=5)


def test_finished_job_stays_done_when_cancelled(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Keep"}).json()["id"]
    posted = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    job_id = posted.json()["id"]
    done = wait_for_job(client, job_id)
    assert done["status"] == "done"
    response = client.post(f"/v1/jobs/{job_id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["take_id"] == done["take_id"]
    assert body["artifacts"]


def test_error_job_is_a_v1_object(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import worker.mock_worker as mock_worker_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(mock_worker_module, "run_job", boom)
    project_id = client.post("/api/projects", json={"title": "Fail"}).json()["id"]
    posted = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    job_id = posted.json()["id"]
    failed = _wait_v1(client, job_id)
    assert failed["status"] == "error"
    assert "synthetic" in failed["error"]
    assert failed["artifacts"] == []


def test_stale_reclaim_does_not_requeue_a_cancel(lyre_env) -> None:
    from server import jobs as jobs_module
    from server import storage

    jobs_module.init_db()
    project = storage.create_project(title="Stale cancel")
    job = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    assert claimed is not None
    jobs_module.cancel_job(job["id"])
    assert jobs_module.get_job(job["id"])["status"] == "running"
    with closing(jobs_module._connect()) as conn:
        conn.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", job["id"]),
        )
        conn.commit()
    reclaimed = jobs_module.reclaim_stale_jobs(stale_after=0)
    assert job["id"] in reclaimed
    final = jobs_module.get_job(job["id"])
    assert final["status"] == "cancelled"
    assert final["error"] is None
    assert final["take_id"] is None
    assert jobs_module.claim_next_queued_job() is None
    # A second reclaim must leave the terminal row alone.
    assert jobs_module.reclaim_stale_jobs(stale_after=0) == []
    assert jobs_module.get_job(job["id"])["status"] == "cancelled"


def test_unknown_job_cancel_is_404(api_client: TestClient) -> None:
    response = api_client.post("/v1/jobs/missing/cancel")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_api_health_shape_is_unchanged(api_client: TestClient) -> None:
    body = api_client.get("/api/health").json()
    assert body["ok"] is True
    assert "queue" not in body
    assert "backend" not in body


def test_pages_follow_a_numeric_cursor_and_job_filters(client: TestClient) -> None:
    first = client.post("/api/projects", json={"title": "One"}).json()["id"]
    second = client.post("/api/projects", json={"title": "Two"}).json()["id"]
    page = client.get("/v1/projects", params={"limit": 1, "cursor": ""}).json()
    assert page["next_cursor"] == "1"
    nxt = client.get("/v1/projects", params={"limit": 1, "cursor": "1"}).json()
    assert {page["data"][0]["id"], nxt["data"][0]["id"]} == {first, second}
    assert nxt["next_cursor"] is None

    posted = client.post(f"/api/projects/{first}/jobs", json={"action": "generate"})
    done = wait_for_job(client, posted.json()["id"])
    active = client.get("/v1/jobs", params={"action": "generate", "active": "true"}).json()
    assert active["data"] == []
    listed = client.get("/v1/jobs", params={"action": "generate", "project_id": first}).json()
    assert listed["data"][0]["id"] == done["id"]


def test_stable_artifact_gets_match_the_files_on_the_take(client: TestClient) -> None:
    from server import storage

    project_id = client.post("/api/projects", json={"title": "Files"}).json()["id"]
    posted = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    done = wait_for_job(client, posted.json()["id"])
    take_id = done["take_id"]
    directory = storage.take_dir(project_id, take_id)
    (directory / "mix.mp3").write_bytes(b"mp3")
    (directory / "lyrics.lrc").write_text("[00:00.00] hi\n", encoding="utf-8")

    mp3 = client.get(f"/v1/artifacts/{project_id}/{take_id}/mix.mp3")
    assert mp3.status_code == 200
    assert mp3.content == b"mp3"
    lrc = client.get(f"/v1/artifacts/{project_id}/{take_id}/lyrics.lrc")
    assert lrc.status_code == 200
    assert b"hi" in lrc.content
    missing = client.get(f"/v1/artifacts/{project_id}/no-such-take/mix.wav")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_health_reports_an_unready_worker(api_client: TestClient) -> None:
    from server.jobs.worker_registry import publish_worker_status

    idle = api_client.get("/v1/health").json()
    assert idle["ready"] is False
    assert "not reported yet" in idle["gpu"]

    publish_worker_status(False, "no gpu", None)
    body = api_client.get("/v1/health").json()
    assert body["ready"] is False
    assert body["gpu"].startswith("unavailable:")
    assert body["dit_loaded"] is None


def test_done_job_with_a_missing_take_has_no_seed(lyre_env) -> None:
    from server import jobs as jobs_module
    from server import storage

    jobs_module.init_db()
    project = storage.create_project(title="gone")
    job = jobs_module.enqueue_job(project["id"], {"action": "generate", "seed": -1})
    jobs_module._set_status(job["id"], "done", take_id="missing-take")
    body = job_object(jobs_module.read_job(job["id"]))
    assert body["status"] == "done"
    assert body["artifacts"] == []
    assert body["seed"] is None


def test_cancel_after_claim_never_enters_the_worker(
    lyre_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import worker.mock_worker as mock_worker_module
    from server import jobs as jobs_module
    from server import storage
    from server.jobs.runner import _run_generate_shaped_job, _run_train_lora_job

    called: list[str] = []
    monkeypatch.setattr(
        mock_worker_module, "run_job", lambda *_args, **_kwargs: called.append("generate")
    )
    monkeypatch.setattr(mock_worker_module, "train_lora", lambda **_kwargs: called.append("train"))

    jobs_module.init_db()
    project = storage.create_project(title="claimed")
    generate = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    assert claimed is not None
    jobs_module.cancel_job(generate["id"])
    _run_generate_shaped_job(
        generate["id"], project["id"], "generate", claimed["dit_profile"], claimed["payload"]
    )
    assert jobs_module.get_job(generate["id"])["status"] == "cancelled"

    training = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    assert jobs_module.claim_next_queued_job() is not None
    jobs_module.cancel_job(training["id"])
    _run_train_lora_job(training["id"], project["id"], {"action": "train_lora"})
    assert jobs_module.get_job(training["id"])["status"] == "cancelled"
    assert called == []
