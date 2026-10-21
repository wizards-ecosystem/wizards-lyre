"""The job heartbeat lease also bounds a hung generate.

Renewing `heartbeat_at` forever kept a stuck VAE decode `running`. Past
`job_timeout_sec` the same loop marks the job `error` and exits the worker
so the OS GPU lock drops. Reclaim must not see a fresh heartbeat and requeue it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from server import jobs as jobs_module
from server import storage
from server.jobs import queue as queue_module


def _running_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("LYRE_DB_PATH", str(tmp_path / "lyre.db"))
    monkeypatch.setenv("LYRE_WORKER", "mock")
    jobs_module.init_db()
    project = storage.create_project(title="Watchdog")
    queued = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    assert claimed is not None
    assert claimed["id"] == queued["id"]
    return queued["id"]


def test_timeout_values_fall_back_when_the_environment_is_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LYRE_JOB_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("LYRE_LORA_TIMEOUT_SEC", raising=False)
    assert queue_module.job_timeout_sec("generate") == queue_module.DEFAULT_JOB_TIMEOUT_SEC
    assert queue_module.job_timeout_sec("train_lora") == queue_module.DEFAULT_LORA_TIMEOUT_SEC

    monkeypatch.setenv("LYRE_JOB_TIMEOUT_SEC", "nope")
    assert queue_module.job_timeout_sec("generate") == queue_module.DEFAULT_JOB_TIMEOUT_SEC
    monkeypatch.setenv("LYRE_JOB_TIMEOUT_SEC", "0")
    assert queue_module.job_timeout_sec("generate") == queue_module.DEFAULT_JOB_TIMEOUT_SEC
    monkeypatch.setenv("LYRE_JOB_TIMEOUT_SEC", "12.5")
    assert queue_module.job_timeout_sec("generate") == 12.5
    # A short generate budget must not cut off style-pack training.
    assert queue_module.job_timeout_sec("train_lora") == queue_module.DEFAULT_LORA_TIMEOUT_SEC
    monkeypatch.setenv("LYRE_LORA_TIMEOUT_SEC", "99")
    assert queue_module.job_timeout_sec("train_lora") == 99.0


def test_exit_process_uses_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(queue_module.os, "_exit", lambda code: called.append(code))
    queue_module._exit_process()
    assert called == [1]


def test_deadline_marks_a_running_job_error_and_asks_the_process_to_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _running_job(tmp_path, monkeypatch)
    jobs_module.set_stage(job_id, "rendering")
    exited: list[int] = []
    monkeypatch.setattr(queue_module, "_exit_process", lambda: exited.append(1))

    stop = threading.Event()
    thread = threading.Thread(
        target=queue_module._heartbeat_loop,
        args=(job_id, stop),
        kwargs={"deadline_sec": 0.05},
        daemon=True,
    )
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    job = jobs_module.get_job(job_id)
    assert job["status"] == "error"
    assert job["error"] is not None
    assert "watchdog" in job["error"]
    assert "rendering" in job["error"]
    assert "LYRE_JOB_TIMEOUT_SEC" in job["error"]
    assert exited == [1]


def test_deadline_does_not_overwrite_a_finished_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _running_job(tmp_path, monkeypatch)
    jobs_module._set_status(job_id, "done", take_id="take-done")
    exited: list[int] = []
    monkeypatch.setattr(queue_module, "_exit_process", lambda: exited.append(1))
    stop = threading.Event()
    thread = threading.Thread(
        target=queue_module._heartbeat_loop,
        args=(job_id, stop),
        kwargs={"deadline_sec": 0.05},
        daemon=True,
    )
    thread.start()
    thread.join(2)
    assert jobs_module.get_job(job_id)["status"] == "done"
    assert exited == []


def test_deadline_while_cancel_is_pending_marks_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _running_job(tmp_path, monkeypatch)
    jobs_module.cancel_job(job_id)
    exited: list[int] = []
    monkeypatch.setattr(queue_module, "_exit_process", lambda: exited.append(1))
    queue_module._expire_job(job_id, 30)
    job = jobs_module.get_job(job_id)
    assert job["status"] == "cancelled"
    assert job["error"] is None
    assert exited == [1]


def test_stopping_the_lease_before_the_deadline_leaves_the_job_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _running_job(tmp_path, monkeypatch)
    exited: list[int] = []
    monkeypatch.setattr(queue_module, "_exit_process", lambda: exited.append(1))
    stop = threading.Event()
    stop.set()
    queue_module._heartbeat_loop(job_id, stop, deadline_sec=30)
    assert jobs_module.get_job(job_id)["status"] == "running"
    assert exited == []


def test_run_claimed_job_errors_a_stuck_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("LYRE_DB_PATH", str(tmp_path / "lyre.db"))
    monkeypatch.setenv("LYRE_WORKER", "mock")
    jobs_module.init_db()
    project = storage.create_project(title="Stuck")
    queued = jobs_module.enqueue_job(project["id"], {"action": "generate"})
    claimed = jobs_module.claim_next_queued_job()
    assert claimed is not None

    import worker.mock_worker as mock_worker_module

    real_run_job = mock_worker_module.run_job

    def _slow(*args: object, **kwargs: object) -> tuple[dict, dict | None, str | None]:
        time.sleep(0.4)
        return real_run_job(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mock_worker_module, "run_job", _slow)
    monkeypatch.setattr(queue_module, "job_timeout_sec", lambda _action: 0.05)
    exited: list[int] = []
    monkeypatch.setattr(queue_module, "_exit_process", lambda: exited.append(1))

    jobs_module.run_claimed_job(claimed)

    job = jobs_module.get_job(queued["id"])
    assert job["status"] == "error"
    assert job["error"] is not None
    assert "watchdog" in job["error"]
    assert exited == [1]
