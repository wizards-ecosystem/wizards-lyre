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


def test_studio_ops_required_for_extract_lego_complete(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Ops Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    # a source take to extract/lego/complete from
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate"},
    )
    gen = _wait_for_job(client, resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    for action in ("extract", "lego", "complete"):
        # explicit wrong profile is rejected outright
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "dit_profile": "iterate",
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action

        # omitting dit_profile is coerced to studio_ops and the job runs
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 200, action
        queued = resp.json()
        assert queued["dit_profile"] == "studio_ops"
        job = _wait_for_job(client, queued["id"])
        assert job["status"] == "done", job.get("error")

        # explicit studio_ops is accepted
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "dit_profile": "studio_ops",
                "source_take_id": source_take_id,
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 200, action
        job = _wait_for_job(client, resp.json()["id"])
        assert job["status"] == "done", job.get("error")


def test_cover_repaint_extract_require_a_real_source(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Source Test"}).json()
    project_id = project["id"]
    client.put(f"/api/projects/{project_id}/plan", json=storage.default_plan())

    for action in ("cover", "repaint", "extract", "lego", "complete"):
        # no source_take_id and no upload_path at all
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": action, "track_name": "vocals"},
        )
        assert resp.status_code == 400, action

        # a source_take_id that doesn't exist
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "source_take_id": "does-not-exist",
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action

        # an upload_path that escapes the project's jail
        resp = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "action": action,
                "upload_path": "../../../../evil.wav",
                "track_name": "vocals",
            },
        )
        assert resp.status_code == 400, action


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
