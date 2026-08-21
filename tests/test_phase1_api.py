"""Phase 1 API tests: health, project+plan disk round trip, mocked generate
flow, path jail, and studio_ops enforcement. No GPU, no CUDA, no ACE-Step.
See SPEC.md sec 11 and 14.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app
from worker.run_worker import run_loop

FORBIDDEN_IMPORTS = (
    "google.genai",
    "google.generativeai",
    "elevenlabs",
    "stability_sdk",
    "suno",
    "udio",
)


@pytest.fixture(autouse=True)
def _reset_mock_worker_state():
    """worker.mock_worker tracks a simulated "loaded" flag at module scope
    (mirroring worker.acestep_worker's real _STATE) so tests can exercise
    worker/run_worker.py's republish-after-recovery behavior; reset it
    between tests or an earlier test's job run would leak into a later
    test expecting a fresh "nothing loaded yet" state."""
    import worker.mock_worker as mock_worker_module

    mock_worker_module._simulated_loaded_dit_profile = None
    yield
    mock_worker_module._simulated_loaded_dit_profile = None


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    # Tests always use the mocked worker; the real acestep_worker is
    # production's default (see server/jobs.py) and is exercised only by the
    # manual, non-pytest scripts/smoke-gpu.py.
    monkeypatch.setenv("BARD_WORKER", "mock")

    # `enqueue_job` only inserts a `queued` row -- it never runs a job
    # itself (SPEC.md sec 5). This thread stands in for the dedicated
    # `worker/run_worker.py` process that drains the same SQLite queue in
    # production, fast-polled so tests stay quick.
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker_thread.start()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.01)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "dit_loaded" in body


def test_health_reports_published_worker_readiness(client: TestClient) -> None:
    """SPEC.md sec 8: /api/health must reflect real worker startup state
    (published by worker/run_worker.py), not a static guess -- the `client`
    fixture's background worker thread publishes it almost immediately."""
    deadline = time.time() + 5.0
    body: dict[str, Any] = {}
    while time.time() < deadline:
        body = client.get("/api/health").json()
        if body.get("dit_loaded") is not None:
            break
        time.sleep(0.01)
    assert body.get("dit_loaded") == "iterate"
    assert "unavailable" not in body["gpu"].lower()


def test_health_before_any_worker_reports_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No worker process has published status yet -- health must say so
    plainly instead of implying a worker is ready."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    with TestClient(app) as c:
        resp = c.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["dit_loaded"] is None
        assert "not reported yet" in body["gpu"]


def test_health_reports_unavailable_when_worker_startup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that fails to start (missing ACE-Step/CUDA/weights) must
    show up as an unavailable/error state, not a silent null."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    jobs_module.publish_worker_status(False, "boom: no GPU found", None)

    with TestClient(app) as c:
        resp = c.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dit_loaded"] is None
        assert "unavailable" in body["gpu"].lower()
        assert "boom" in body["gpu"]


def test_worker_republishes_status_after_recovering_from_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 8: a worker that initially reports unavailable (e.g. a
    transient CUDA error at startup) but goes on to successfully process a
    queued job must have its published readiness/loaded-profile/
    capabilities updated to reflect that recovery -- not stay stuck
    reporting unavailable (and rejecting new jobs) forever."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    import worker.mock_worker as mock_worker_module

    monkeypatch.setattr(
        mock_worker_module, "initialize_worker", lambda: (False, "boom: no GPU yet")
    )

    # The mock is fast enough that, left unblocked, the worker thread can
    # publish the startup failure *and* finish the job before this test's
    # poll loop ever observes the transient "unavailable" state. Block the
    # job until the test has confirmed it, so the ordering is deterministic
    # instead of a timing race.
    proceed_with_job = threading.Event()
    real_run_job = mock_worker_module.run_job

    def blocking_run_job(job, plan, take_id, take_dir):
        proceed_with_job.wait(timeout=5)
        return real_run_job(job, plan, take_id, take_dir)

    monkeypatch.setattr(mock_worker_module, "run_job", blocking_run_job)

    from server import jobs as jobs_module

    jobs_module.init_db()
    project = storage.create_project(title="Recovery")
    project_id = project["id"]

    # Enqueue before the worker publishes anything, purely so the ordering
    # below is deterministic (the job's claim/run loop never re-checks
    # capability once queued, regardless of when it was enqueued -- ordinary
    # 'generate' requests are never capability-gated in the first place,
    # see test_ordinary_profiles_queue_despite_total_worker_startup_failure).
    job = jobs_module.enqueue_job(project_id, {"action": "generate"})
    assert job["status"] == "queued"

    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker_thread.start()
    try:
        # The simulated startup failure must be published first (the job
        # is still blocked in run_job, so this can't have raced ahead).
        deadline = time.time() + 5.0
        status = None
        while time.time() < deadline:
            status = jobs_module.get_worker_status()
            if status is not None and status["ready"] is False:
                break
            time.sleep(0.01)
        assert status is not None and status["ready"] is False
        assert "boom" in status["message"]

        # Now let the already-queued job run (claiming never re-checks
        # published capability) -- it succeeds, and the worker recovers.
        proceed_with_job.set()
        deadline = time.time() + 5.0
        current_job = None
        while time.time() < deadline:
            current_job = jobs_module.get_job(job["id"])
            if current_job["status"] in ("done", "error"):
                break
            time.sleep(0.01)
        assert current_job is not None and current_job["status"] == "done", (
            current_job and current_job.get("error")
        )

        # Readiness/capabilities must be republished to reflect the
        # recovery, not stay stuck on the startup failure.
        deadline = time.time() + 5.0
        status_after = None
        while time.time() < deadline:
            status_after = jobs_module.get_worker_status()
            if status_after is not None and status_after["ready"] is True:
                break
            time.sleep(0.01)
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)

    assert status_after is not None
    assert status_after["ready"] is True
    assert status_after["loaded_dit_profile"] == "iterate"
    assert jobs_module.get_worker_capability("iterate") == (True, None)


def test_project_create_and_plan_roundtrip(client: TestClient) -> None:
    resp = client.post("/api/projects", json={"title": "My Song", "query": "lofi beat"})
    assert resp.status_code == 200
    project = resp.json()
    project_id = project["id"]
    assert project["title"] == "My Song"
    assert project["dit_profile"] == "iterate"

    # project.json actually landed on disk under projects/
    on_disk = storage.load_project(project_id)
    assert on_disk["id"] == project_id

    new_plan = {
        "query": "lofi beat",
        "caption": "lofi, chill, rhodes, brushed drums",
        "negative": [],
        "lyrics": "[Verse]\nquiet streets tonight",
        "instrumental": False,
        "vocal_language": "en",
        "bpm": 84,
        "keyscale": "A Minor",
        "timesignature": "4/4",
        "duration_sec": 90,
        "sections": [],
    }
    resp = client.put(f"/api/projects/{project_id}/plan", json=new_plan)
    assert resp.status_code == 200

    # plan.json round trips through disk, not just the response body
    plan_path = storage.plan_json_path(project_id)
    on_disk_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert on_disk_plan["caption"] == new_plan["caption"]
    assert on_disk_plan["bpm"] == 84

    resp = client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"]["caption"] == new_plan["caption"]
    assert body["takes"] == []

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    listed = resp.json()
    assert any(p["id"] == project_id for p in listed)


def test_generate_job_creates_playable_take(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Take Test"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "synthwave, driving bass"},
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    assert resp.status_code == 200
    queued = resp.json()
    # enqueue_job only inserts the row; it never runs the job in the request
    # (SPEC.md sec 5), so it may still be queued/running when this returns.
    assert queued["status"] in ("queued", "running", "done")

    job = _wait_for_job(client, queued["id"])
    assert job["status"] == "done"
    assert job["error"] is None
    take_id = job["take_id"]
    assert take_id

    # job is retrievable by id and shows up in recent jobs
    resp = client.get(f"/api/jobs/{job['id']}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"

    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert any(j["id"] == job["id"] for j in resp.json())

    # take metadata recorded an actual seed (seed=-1 means "worker picks")
    resp = client.get(f"/api/projects/{project_id}/takes")
    assert resp.status_code == 200
    takes = resp.json()
    assert len(takes) == 1
    take = takes[0]
    assert take["id"] == take_id
    assert take["task_type"] == "text2music"
    assert isinstance(take["seed"], int)
    assert take["seed"] != -1
    assert take["caption"] == "synthwave, driving bass"

    # audio is a real, playable (tiny) WAV file written under projects/
    resp = client.get(f"/api/projects/{project_id}/takes/{take_id}/audio")
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"
    assert resp.content[8:12] == b"WAVE"
    assert len(resp.content) > 44  # header + at least some frame data

    audio_path = storage.take_audio_path(project_id, take_id)
    assert audio_path.exists()
    assert audio_path.is_relative_to(storage.config.projects_dir())

    # take is immutable metadata + audio living in its own directory
    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    assert meta_path.exists()


def test_simple_query_generation_fills_and_persists_plan(client: TestClient) -> None:
    """SPEC.md sec 7.2: Simple mode ('query' set, caption/lyrics blank) must
    have the LM (mocked here) fill caption/lyrics/metas, and the filled plan
    must be persisted to plan.json, not discarded."""
    project = client.post(
        "/api/projects", json={"title": "Simple Mode", "query": "dreamy synthwave drive"}
    ).json()
    project_id = project["id"]

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["plan"]["query"] == "dreamy synthwave drive"
    assert detail["plan"]["caption"] == ""

    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done", job.get("error")

    detail = client.get(f"/api/projects/{project_id}").json()
    assert "dreamy synthwave drive" in detail["plan"]["caption"]
    assert detail["plan"]["bpm"] is not None
    assert detail["plan"]["keyscale"] is not None

    take = detail["takes"][0]
    assert take["caption"] == detail["plan"]["caption"]


def test_plan_patch_merge_preserves_concurrent_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 7.2: a plan edit saved while a simple-mode generate job is
    running (sections, negative prompts, instrumental, ...) must survive --
    the job's plan_patch is a delta merged onto the *current* on-disk plan
    when the job finishes, not a full-plan overwrite of the stale snapshot
    loaded when the job started."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    project = storage.create_project(title="Race", query="dreamy synthwave drive")
    project_id = project["id"]

    import worker.mock_worker as mock_worker_module

    real_run_job = mock_worker_module.run_job

    def _run_job_with_concurrent_edit(job, plan, take_id, take_dir):
        # Simulate a user's PUT /plan landing while this "long-running" job
        # is executing -- server.jobs loaded `plan` before this call.
        current = storage.load_plan(project_id)
        storage.save_plan(
            project_id, {**current, "negative": ["concurrent edit"], "instrumental": True}
        )
        return real_run_job(job=job, plan=plan, take_id=take_id, take_dir=take_dir)

    monkeypatch.setattr(mock_worker_module, "run_job", _run_job_with_concurrent_edit)

    job = jobs_module.enqueue_job(project_id, {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    jobs_module.run_claimed_job(claimed)

    final_job = jobs_module.get_job(job["id"])
    assert final_job["status"] == "done", final_job.get("error")

    final_plan = storage.load_plan(project_id)
    # the concurrent edit to unrelated fields survived ...
    assert final_plan["negative"] == ["concurrent edit"]
    assert final_plan["instrumental"] is True
    # ... and the LM-filled fields from the job were still applied on top
    assert "dreamy synthwave drive" in final_plan["caption"]


def test_project_json_updates_are_serialized_across_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 5/10: save_plan/patch_project (HTTP server) and
    set_active_take (worker, after a generation completes) all
    read-modify-write project.json. Without serialization, one can read a
    stale copy and overwrite the other's concurrent change -- e.g. a plan
    save racing generation completion clobbering the newly assigned
    active_take_id. This forces a genuine interleaving window (one writer
    is paused mid-critical-section while the other attempts its own
    update) and asserts neither update is lost."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))

    project = storage.create_project(title="Race")
    project_id = project["id"]
    take_id = storage.new_id()

    real_write_json = storage._write_json
    project_path = storage.project_json_path(project_id)
    thread_a_writing = threading.Event()
    thread_b_attempted = threading.Event()

    def slow_write_json(path, data):
        if path == project_path and not thread_a_writing.is_set():
            thread_a_writing.set()
            # Give thread B a real chance to race: it must block on the
            # project lock here, not read a stale project dict.
            thread_b_attempted.wait(timeout=2)
            time.sleep(0.1)
        real_write_json(path, data)

    monkeypatch.setattr(storage, "_write_json", slow_write_json)

    errors: list[Exception] = []

    def run_patch():
        try:
            storage.patch_project(project_id, {"title": "renamed while racing"})
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    def run_set_active():
        thread_a_writing.wait(timeout=2)
        thread_b_attempted.set()
        try:
            storage.set_active_take(project_id, take_id)
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    t_patch = threading.Thread(target=run_patch)
    t_active = threading.Thread(target=run_set_active)
    t_patch.start()
    t_active.start()
    t_patch.join(timeout=5)
    t_active.join(timeout=5)

    assert not errors, errors
    assert not t_patch.is_alive()
    assert not t_active.is_alive()

    final = storage.load_project(project_id)
    assert final["title"] == "renamed while racing"
    assert final["active_take_id"] == take_id


def test_plan_json_updates_are_serialized_across_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 5/10: save_plan (a PUT /plan from the HTTP server) and
    merge_plan_patch (the worker's read-merge-write of a simple-mode
    plan_patch after a job finishes) both mutate plan.json. Without
    serialization, the worker can read a plan.json snapshot from before a
    concurrent PUT lands and then overwrite that PUT when it writes its
    merged result back. This forces a genuine interleaving window (one
    writer is paused mid-critical-section while the other attempts its own
    update) and asserts neither update is lost."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))

    project = storage.create_project(title="PlanRace")
    project_id = project["id"]
    storage.save_plan(project_id, {**storage.default_plan(), "caption": "original"})

    real_write_json = storage._write_json
    plan_path = storage.plan_json_path(project_id)
    thread_a_writing = threading.Event()
    thread_b_attempted = threading.Event()

    def slow_write_json(path, data):
        if path == plan_path and not thread_a_writing.is_set():
            thread_a_writing.set()
            # Give thread B (the worker's patch merge) a real chance to
            # race: it must block on the plan lock here, reading plan.json
            # only after thread A's write below actually lands.
            thread_b_attempted.wait(timeout=2)
            time.sleep(0.1)
        real_write_json(path, data)

    monkeypatch.setattr(storage, "_write_json", slow_write_json)

    errors: list[Exception] = []

    def run_save():
        try:
            storage.save_plan(project_id, {**storage.default_plan(), "caption": "user edit"})
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    def run_merge():
        thread_a_writing.wait(timeout=2)
        thread_b_attempted.set()
        try:
            storage.merge_plan_patch(project_id, {"bpm": 120, "keyscale": "C Major"})
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    t_save = threading.Thread(target=run_save)
    t_merge = threading.Thread(target=run_merge)
    t_save.start()
    t_merge.start()
    t_save.join(timeout=5)
    t_merge.join(timeout=5)

    assert not errors, errors
    assert not t_save.is_alive()
    assert not t_merge.is_alive()

    final_plan = storage.load_plan(project_id)
    # the user's PUT survived ...
    assert final_plan["caption"] == "user edit"
    # ... and so did the worker's merged patch, on top of it
    assert final_plan["bpm"] == 120
    assert final_plan["keyscale"] == "C Major"


def test_active_take_not_promoted_when_plan_patch_merge_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 10 point 5: a failed job must leave the project's
    active_take_id untouched, not pointing at the take that just failed.
    The worker must promote a take to active only after every fallible
    persistence step (write_take_meta, merge_plan_patch) has actually
    succeeded (reviewer-flagged: merge_plan_patch used to run *after*
    set_active_take, so a merge failure -- e.g. a lock timeout or disk
    error -- left active_take_id pointing at a take whose meta.json says
    `error`)."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    # Simple mode (query set, caption/lyrics empty via default_plan) so
    # mock_worker returns a plan_patch and merge_plan_patch actually runs.
    project = storage.create_project(title="Patch Merge Failure", query="dreamy synthwave drive")
    project_id = project["id"]
    assert storage.load_project(project_id)["active_take_id"] is None

    def boom(*args, **kwargs):
        raise RuntimeError("disk error")

    monkeypatch.setattr(storage, "merge_plan_patch", boom)

    job = jobs_module.enqueue_job(project_id, {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    assert claimed["id"] == job["id"]
    jobs_module.run_claimed_job(claimed)

    final_job = jobs_module.get_job(job["id"])
    assert final_job["status"] == "error"
    assert "disk error" in final_job["error"]

    # never promoted -- must not point at the take that just failed
    assert storage.load_project(project_id)["active_take_id"] is None

    takes = storage.list_takes(project_id)
    assert len(takes) == 1
    assert takes[0]["id"] == final_job["take_id"]
    assert takes[0]["error"] is not None


def test_worker_failure_writes_error_take_meta(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 10 point 5: on failure the worker writes meta.json with
    `error` set for the take it already allocated, not just a job error."""
    project = client.post("/api/projects", json={"title": "Boom"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "will explode"},
    )

    import worker.mock_worker as mock_worker

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(mock_worker, "run_job", _boom)

    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    job = _wait_for_job(client, resp.json()["id"])
    assert job["status"] == "error"
    assert "synthetic worker failure" in job["error"]

    take_id = job["take_id"]
    assert take_id  # the take dir was allocated before the worker blew up

    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["error"] and "synthetic worker failure" in meta["error"]

    assert not (storage.take_dir(project_id, take_id) / "mix.wav").exists()

    # a failed take is never promoted to active
    project_after = client.get(f"/api/projects/{project_id}").json()["project"]
    assert project_after["active_take_id"] != take_id


def test_path_jail_rejects_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    with pytest.raises(storage.PathJailError):
        storage.jailed_path("..", "evil.txt")

    with pytest.raises(storage.PathJailError):
        storage.jailed_path("..", "..", "..", "Users", "evil.txt")

    # a well-behaved relative path stays inside the jail
    ok_path = storage.jailed_path("some-project", "takes", "some-take", "mix.wav")
    assert ok_path.is_relative_to(storage.config.projects_dir())


def test_path_jail_rejects_escape_via_api(client: TestClient) -> None:
    resp = client.get("/api/projects/..")
    assert resp.status_code in (400, 404)


def test_jailed_output_path_rejects_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """scripts/smoke-gpu.py writes under output/, not a bare OS temp dir --
    same jail mechanism as projects/, just rooted at output_dir()."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    with pytest.raises(storage.PathJailError):
        storage.jailed_output_path("..", "evil.txt")

    with pytest.raises(storage.PathJailError):
        storage.jailed_output_path("..", "..", "..", "Users", "evil.txt")

    ok_path = storage.jailed_output_path("smoke-gpu", "some-take-id")
    assert ok_path.is_relative_to(storage.config.output_dir())


def test_smoke_gpu_script_writes_under_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 8.1/11: generated audio may only land under projects/ or
    output/. scripts/smoke-gpu.py used to write into tempfile.mkdtemp(),
    which is outside both -- this drives the real script end to end (with a
    faked run_job so it needs no GPU) and checks the file actually lands
    under output_dir()."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke-gpu.py"
    spec = importlib.util.spec_from_file_location("bard_smoke_gpu_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _fake_run_job(job, plan, take_id, take_dir):
        take_dir.mkdir(parents=True, exist_ok=True)
        (take_dir / "mix.wav").write_bytes(b"RIFF____WAVEfake")
        return {"seed": 42, "duration_sec": 10.0}, None

    monkeypatch.setattr(module, "run_job", _fake_run_job)

    assert module.main() == 0

    output_root = storage.config.output_dir()
    written = list(output_root.rglob("mix.wav"))
    assert len(written) == 1
    assert written[0].is_relative_to(output_root)


def test_phase_gated_actions_rejected_until_their_phase(client: TestClient) -> None:
    """SPEC.md sec 12 (phase order): phase 1 is generate; phase 2 adds cover
    and repaint (now that the web UI has a waveform region-select feeding
    repainting_start/repainting_end -- see test_repaint_take_flow).
    extract/lego/complete (phase 3) still need frontend workflow this build
    doesn't have yet -- a base-model-swap confirmation/loading UX -- so the
    API must reject them outright instead of accepting a job the UI can't
    drive, regardless of how well-formed the rest of the request is."""
    project = client.post("/api/projects", json={"title": "Phase Gate Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    )
    gen = _wait_for_job(client, resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    for action in ("extract", "lego", "complete"):
        # Well-formed in every other respect (real source, studio_ops
        # profile where that would otherwise be required) -- still rejected
        # purely because this action isn't available yet.
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "dit_profile": "studio_ops",
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action
        assert "not available yet" in resp.json()["detail"], action

        # No job row is left behind for a rejected action.
        assert all(j["action"] != action for j in client.get("/api/jobs").json())


def test_generate_rejects_studio_ops_dit_profile(client: TestClient) -> None:
    """API-level companion to test_resolve_dit_profile_studio_ops_enforcement
    (reviewer-flagged): a client must not be able to sneak the reserved
    studio_ops profile into an enabled action, whether via an explicit
    override or the project's persisted default."""
    project = client.post("/api/projects", json={"title": "Studio Ops Reject Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "studio_ops"},
    )
    assert resp.status_code == 400
    assert "studio_ops" in resp.json()["detail"]
    assert all(j["action"] != "generate" for j in client.get("/api/jobs").json())

    client.patch(f"/api/projects/{project_id}", json={"dit_profile": "studio_ops"})
    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 400
    assert "studio_ops" in resp.json()["detail"]


def test_cover_rejects_out_of_range_audio_cover_strength(client: TestClient) -> None:
    """SPEC.md sec 8.1: audio_cover_strength is a 0-1 mix ratio (reviewer-
    flagged: an unconstrained float let clients enqueue negative,
    greater-than-one, NaN, or infinite values that would only fail deep
    inside the worker)."""
    project = client.post("/api/projects", json={"title": "Cover Strength Bounds Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    )
    gen = _wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    for bad_strength in (-0.1, 1.1):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": "cover",
                "source_take_id": source_take_id,
                "audio_cover_strength": bad_strength,
            },
        )
        assert resp.status_code == 422, bad_strength

    for bad_strength in (float("nan"), float("inf"), float("-inf")):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": "cover",
                "source_take_id": source_take_id,
                "audio_cover_strength": bad_strength,
            },
        )
        # ge/le still rejects these (any comparison with NaN/inf against 0/1
        # is False/out-of-bounds), but FastAPI's 422 body echoes back the
        # rejected `input`, and Starlette's JSONResponse renders with
        # allow_nan=False -- so a non-finite input surfaces as a 400 while
        # the response is built, not a 422. Either way it never reaches
        # enqueue_job/the worker, which is the property under test.
        assert resp.status_code in (400, 422), bad_strength
        assert all(j["action"] != "cover" for j in client.get("/api/jobs").json())


def test_resolve_dit_profile_studio_ops_enforcement() -> None:
    """Direct unit coverage for _resolve_dit_profile's studio_ops coercion
    (SPEC.md sec 8.1) -- currently unreachable via enqueue_job (see
    test_phase_gated_actions_rejected_until_their_phase) because
    extract/lego/complete are phase-3-gated, but the logic stays correct and
    tested so enabling those actions later is just a PHASE_GATED_ACTIONS
    edit, not new/unverified logic."""
    from server import jobs as jobs_module

    for action in ("extract", "lego", "complete"):
        assert jobs_module._resolve_dit_profile(action, None, "iterate") == "studio_ops"
        assert jobs_module._resolve_dit_profile(action, "studio_ops", "iterate") == "studio_ops"
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_dit_profile(action, "iterate", "iterate")

    # An explicit job-level dit_profile always wins over the project default.
    assert jobs_module._resolve_dit_profile("generate", "polish", "iterate") == "polish"
    # An omitted job-level dit_profile must fall back to the *project's*
    # persisted default, not a hardcoded "iterate" (reviewer-flagged: the
    # included frontend always omits it, so this is the only thing that
    # makes PATCH .../dit_profile have any effect on generation).
    assert jobs_module._resolve_dit_profile("generate", None, "polish") == "polish"
    assert jobs_module._resolve_dit_profile("generate", None, "iterate") == "iterate"

    # studio_ops is reserved for extract/lego/complete (SPEC.md sec 8.1) --
    # reject it for every other action, whether from an explicit override or
    # a project's persisted default (reviewer-flagged: either path could
    # otherwise load the base model for ordinary generation/cover).
    for action in ("generate", "cover"):
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_dit_profile(action, "studio_ops", "iterate")
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_dit_profile(action, None, "studio_ops")


def test_resolve_source_audio_requires_a_real_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct unit coverage for _resolve_source_audio's source validation
    (SPEC.md sec 8.1/11) -- see test_resolve_dit_profile_studio_ops_enforcement
    for why this is tested below enqueue_job rather than through it."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    from server import jobs as jobs_module

    project = storage.create_project(title="Source Unit Test")
    project_id = project["id"]

    for action in ("cover", "repaint", "extract", "lego", "complete"):
        # no source_take_id and no upload_path at all
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_source_audio(project_id, action, {"track_name": "vocals"})

        # a source_take_id that doesn't exist
        with pytest.raises(jobs_module.JobError):
            jobs_module._resolve_source_audio(
                project_id, action, {"source_take_id": "does-not-exist"}
            )

        # an upload_path that escapes the project's jail
        with pytest.raises(storage.PathJailError):
            jobs_module._resolve_source_audio(
                project_id, action, {"upload_path": "../../../../evil.wav"}
            )

    # generate never requires a source
    assert jobs_module._resolve_source_audio(project_id, "generate", {}) is None


def test_enqueue_uses_project_dit_profile_when_job_omits_it(client: TestClient) -> None:
    """SPEC.md: a project's dit_profile (set via PATCH /api/projects/{id})
    must actually affect generation. The included frontend never sends a
    job-level dit_profile, so enqueue_job's fallback for an omitted one must
    be the project's own persisted profile, not a hardcoded 'iterate'
    (reviewer-flagged: otherwise PATCH .../dit_profile is silently
    ineffective for every real generate request)."""
    project = client.post("/api/projects", json={"title": "Polish Project"}).json()
    project_id = project["id"]
    assert project["dit_profile"] == "iterate"  # default, before the PATCH below
    client.put(f"/api/projects/{project_id}/plan", json={**storage.default_plan(), "caption": "x"})

    patch_resp = client.patch(f"/api/projects/{project_id}", json={"dit_profile": "polish"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["dit_profile"] == "polish"

    resp = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 200
    queued = resp.json()
    assert queued["dit_profile"] == "polish"

    job = _wait_for_job(client, queued["id"])
    assert job["status"] == "done", job.get("error")
    take = next(t for t in client.get(f"/api/projects/{project_id}").json()["takes"] if t["id"] == job["take_id"])
    assert take["dit_profile"] == "polish"

    # An explicit job-level dit_profile still overrides the project default.
    resp = client.post(
        f"/api/projects/{project_id}/jobs", json={"action": "generate", "dit_profile": "iterate"}
    )
    assert resp.json()["dit_profile"] == "iterate"


def test_batch_size_forced_to_one(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC.md sec 8.1: 'batch_size > 1 is optional later; v1 may force 1 on
    16 GB.' An unvalidated client value (zero, negative, huge) must never
    reach the GPU backend."""
    project = client.post("/api/projects", json={"title": "Batch"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json={**storage.default_plan(), "caption": "x"})

    import worker.mock_worker as mock_worker_module

    seen_batch_sizes: list[Any] = []
    real_run_job = mock_worker_module.run_job

    def _spy_run_job(job, plan, take_id, take_dir):
        seen_batch_sizes.append(job.get("batch_size"))
        return real_run_job(job=job, plan=plan, take_id=take_id, take_dir=take_dir)

    monkeypatch.setattr(mock_worker_module, "run_job", _spy_run_job)

    for requested in (0, -5, 999):
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "batch_size": requested},
        )
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")

    assert seen_batch_sizes == [1, 1, 1]


def test_invalid_action_rejected(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Bad Action"}).json()
    project_id = project["id"]
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "not-a-real-action"},
    )
    assert resp.status_code == 400


def test_quality_profile_rejected_without_cpu_offload_support(client: TestClient) -> None:
    """SPEC.md sec 4.1/8.1: `quality` (XL) needs CPU offload on a 16 GB card;
    the server must reject it up front, not let a job OOM mid-run. It does
    this by reading capability the worker process already published to
    SQLite (SPEC.md sec 10 point 4: the server never imports acestep itself
    -- this test never touches `worker.acestep_worker` either)."""
    from server import jobs as jobs_module

    project = client.post("/api/projects", json={"title": "Quality"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "orchestral, cinematic"},
    )

    # The `client` fixture's background worker thread also publishes
    # capabilities once at its own startup; wait for that one-time publish
    # to land before overriding it below, or it can race and clobber our
    # override right back to "supported" afterward.
    deadline = time.time() + 5.0
    while jobs_module.get_worker_capability("quality") is None and time.time() < deadline:
        time.sleep(0.01)

    # simulate what worker/run_worker.py publishes at startup
    jobs_module.publish_worker_capability(
        "quality", False, "no cpu-offload-capable handler in this environment"
    )

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "quality"},
    )
    assert resp.status_code == 400
    assert "cpu-offload" in resp.json()["detail"].lower()

    # other profiles are unaffected by the quality-only capability check
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    )
    assert resp.status_code == 200

    # once the worker reports it can load quality after all, it's accepted
    jobs_module.publish_worker_capability("quality", True, None)
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "quality"},
    )
    assert resp.status_code == 200


def test_ordinary_profiles_queue_despite_total_worker_startup_failure(
    client: TestClient,
) -> None:
    """When the *whole worker* fails to start (missing ACE-Step/CUDA/
    weights), worker/run_worker.py's _publish_capabilities marks every DiT
    profile unsupported, not just quality. SPEC.md sec 4.1/8.1 only calls
    for early rejection of quality (CPU-offload support); rejecting an
    ordinary 'iterate' generate request the same way would mean it can never
    reach the queue, so it can never become a recoverable `error` job with
    failure metadata (SPEC.md sec 10 point 5 / README's documented "jobs
    fail cleanly instead of being rejected" contract), and a transient
    startup failure could never self-heal since no job would ever reach
    _ensure_loaded to retry it (reviewer-flagged)."""
    from server import jobs as jobs_module

    project = client.post("/api/projects", json={"title": "Startup Failure"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "x"},
    )

    # Simulate worker/run_worker.py._publish_capabilities' blanket-failure
    # publish: every profile marked unsupported, exactly as it does when
    # initialize_worker() fails.
    for dit_profile in storage.VALID_DIT_PROFILES:
        jobs_module.publish_worker_capability(
            dit_profile, False, "worker startup: failed to preload default 'iterate' DiT + LM: boom"
        )
    jobs_module.publish_worker_status(False, "boom: no GPU found", None)

    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["status"] == "queued"

    # quality remains gated even in this scenario -- it's the one profile
    # SPEC.md actually calls for early rejection of.
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "quality"},
    )
    assert resp.status_code == 400


def test_quality_profile_allowed_when_worker_has_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No worker has published capability yet (e.g. it hasn't started) --
    enqueue must fail open rather than block the user forever; the worker's
    own guard still enforces this when the job actually runs."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    project = storage.create_project(title="No Worker Yet")
    job = jobs_module.enqueue_job(project["id"], {"action": "generate", "dit_profile": "quality"})
    assert job["status"] == "queued"


def test_worker_lease_is_a_cross_process_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 4.3: one GPU occupant. Two worker processes must never
    both hold the lease at once -- a live rival is refused, but a lease
    whose heartbeat has gone stale (crashed/killed owner) can be taken
    over, and the original owner can no longer renew it once that happens."""
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))

    from server import jobs as jobs_module

    jobs_module.init_db()

    assert jobs_module.acquire_worker_lease("worker-a") is True
    # A second, live worker must not be able to acquire the same lease.
    assert jobs_module.acquire_worker_lease("worker-b") is False
    # The holder can keep renewing it; a non-holder never can.
    assert jobs_module.renew_worker_lease("worker-a") is True
    assert jobs_module.renew_worker_lease("worker-b") is False

    # Releasing lets another worker take over immediately.
    jobs_module.release_worker_lease("worker-a")
    assert jobs_module.acquire_worker_lease("worker-b") is True
    assert jobs_module.renew_worker_lease("worker-a") is False

    # A crashed worker's lease must not block forever: stale_after=0 treats
    # worker-b's just-set heartbeat as immediately stale, standing in for a
    # heartbeat that actually went quiet.
    assert jobs_module.acquire_worker_lease("worker-c", stale_after=0) is True
    assert jobs_module.renew_worker_lease("worker-b") is False


def test_worker_status_and_capability_read_as_stale_after_heartbeat_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker_status/worker_capabilities row from a worker that has since
    exited or hung must not be trusted forever: /api/health (via
    get_worker_status) and enqueue validation (via _check_worker_capability)
    must both treat a stale publish as unknown/unavailable rather than
    repeating a long-dead process's last report (SPEC.md sec 4.3 / sec 8)."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    jobs_module.publish_worker_status(True, "worker: 'iterate' DiT + LM currently loaded", "iterate")
    jobs_module.publish_worker_capability("quality", False, "quality requires CPU offload")

    # Freshly published: trusted as-is.
    fresh = jobs_module.get_worker_status()
    assert fresh is not None
    assert fresh["ready"] is True
    assert fresh["loaded_dit_profile"] == "iterate"
    with pytest.raises(jobs_module.JobError):
        jobs_module._check_worker_capability("quality")

    # stale_after=0 treats the publish above as immediately stale, standing
    # in for a worker that has since exited or hung without republishing.
    stale = jobs_module.get_worker_status(stale_after=0)
    assert stale is not None
    assert stale["ready"] is False
    assert stale["loaded_dit_profile"] is None
    assert "stale" in stale["message"].lower()

    # A stale capability publish must no longer block enqueue -- it reads as
    # unknown (same as never-published), not as a trusted rejection.
    jobs_module._check_worker_capability("quality", stale_after=0)  # must not raise


def test_reclaim_stale_running_job_requeues_then_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 10 point 5: a job a dead worker left stuck `running` must
    be recoverable, not stuck forever. A stale heartbeat requeues it for a
    retry; past `MAX_ATTEMPTS` it's marked `error` instead of retried
    forever."""
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")

    from server import jobs as jobs_module

    jobs_module.init_db()
    project = storage.create_project(title="Stale")
    job = jobs_module.enqueue_job(project["id"], {"action": "generate"})

    claimed = jobs_module.claim_next_queued_job()
    assert claimed["id"] == job["id"]
    assert claimed["status"] == "running"

    # heartbeat/updated_at is "now"; stale_after=0 treats it as immediately stale
    reclaimed = jobs_module.reclaim_stale_jobs(stale_after=0)
    assert job["id"] in reclaimed
    after_first_reclaim = jobs_module.get_job(job["id"])
    assert after_first_reclaim["status"] == "queued"  # first miss: requeued for a retry

    # burn through the remaining attempts the same way
    for _ in range(jobs_module.MAX_ATTEMPTS - 1):
        jobs_module.claim_next_queued_job()
        jobs_module.reclaim_stale_jobs(stale_after=0)

    final = jobs_module.get_job(job["id"])
    assert final["status"] == "error"
    assert "heartbeat" in final["error"].lower()

    # a job with a fresh heartbeat is left alone
    other = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    jobs_module.claim_next_queued_job()
    reclaimed_fresh = jobs_module.reclaim_stale_jobs(stale_after=3600)
    assert other["id"] not in reclaimed_fresh
    assert jobs_module.get_job(other["id"])["status"] == "running"


def test_no_forbidden_engine_imports() -> None:
    """SPEC.md sec 11: static check that no Lyria/Gemini/ElevenLabs/Stability
    /Suno/Udio client code has been added to this project's own source (see
    also tests/test_spec_lock.py, which scans the whole repo)."""
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r"^\s*(?:from|import)\s+(" + "|".join(re.escape(n) for n in FORBIDDEN_IMPORTS) + r")\b",
        re.MULTILINE,
    )
    skip_parts = {".venv", "node_modules", ".git", "dist"}
    hits: list[str] = []
    for glob in ("**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.mjs"):
        for path in root.glob(glob):
            if any(part in skip_parts for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if pattern.search(text):
                hits.append(str(path.relative_to(root)))
    assert hits == []
