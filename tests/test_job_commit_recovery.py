"""SPEC.md sec 16.5: staged commits, short locks, and failures that keep
existing audio.

A worker thread runs a job whose mock render blocks on an event, so each
test can act (edit, cancel, delete) while the "GPU" is busy and then let it
finish. Mock worker only.
"""

from __future__ import annotations

import errno
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import worker.mock_worker as mock_worker_module
from server import jobs, storage

_REAL_RUN_JOB = mock_worker_module.run_job


class _Crash(BaseException):
    pass


class _BlockedRender:
    """Runs one claimed job on a thread; its render waits for `release()`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rendering = threading.Event()
        self._release = threading.Event()
        self._armed = False
        self.thread: threading.Thread | None = None

        def blocking(**kwargs: Any):
            result = _REAL_RUN_JOB(**kwargs)
            if not self._armed:
                return result
            self.rendering.set()
            assert self._release.wait(10)
            return result

        monkeypatch.setattr(mock_worker_module, "run_job", blocking)

    def start(self) -> None:
        self._armed = True
        claimed = jobs.claim_next_queued_job()
        assert claimed is not None
        self.thread = threading.Thread(target=jobs.run_claimed_job, args=(claimed,))
        self.thread.start()
        assert self.rendering.wait(10)

    def release(self) -> None:
        self._release.set()
        assert self.thread is not None
        self.thread.join(10)
        assert not self.thread.is_alive()


@pytest.fixture()
def blocked(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[_BlockedRender]:
    render = _BlockedRender(monkeypatch)
    yield render
    render._release.set()
    if render.thread is not None:
        render.thread.join(10)


def _project_with_a_kept_take(api_client: TestClient, caption: str = "A") -> tuple[str, str]:
    project_id = api_client.post("/api/projects", json={"title": "Commit"}).json()["id"]
    api_client.put(
        f"/api/projects/{project_id}/plan", json={"caption": caption, "caption_rewrite": False}
    )
    api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    claimed = jobs.claim_next_queued_job()
    jobs.run_claimed_job(claimed)
    kept = jobs.get_job(claimed["id"])["take_id"]
    assert storage.load_project(project_id)["active_take_id"] == kept
    return project_id, kept


def _enqueue(api_client: TestClient, project_id: str) -> str:
    resp = api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def test_editing_draft_b_while_a_renders(api_client: TestClient, blocked: _BlockedRender) -> None:
    """Gap 5 of the completion review: plan edits no longer wait behind
    inference. Take A keeps A's provenance, draft B stays, and the prior
    active take stays kept."""
    project_id, kept = _project_with_a_kept_take(api_client, "A")
    job_id = _enqueue(api_client, project_id)
    blocked.start()

    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    started = time.monotonic()
    resp = api_client.put(f"/api/projects/{project_id}/plan", json={**plan, "caption": "B"})
    assert resp.status_code == 200
    assert time.monotonic() - started < 2.0  # not blocked by the running render

    blocked.release()
    done = jobs.get_job(job_id)
    assert done["status"] == "done", done.get("error")
    take = storage.get_take(project_id, done["take_id"])
    assert take["caption"] == "A"
    assert take["provenance"]["brief"]["caption"] == "A"
    assert storage.load_plan(project_id)["caption"] == "B"
    assert storage.load_project(project_id)["active_take_id"] == kept


def test_nothing_is_visible_until_the_commit(
    api_client: TestClient, blocked: _BlockedRender
) -> None:
    project_id, kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)
    blocked.start()
    assert [take["id"] for take in storage.list_takes(project_id)] == [kept]
    assert len(storage.list_staged_takes()) == 1
    blocked.release()
    assert len(storage.list_takes(project_id)) == 2
    assert storage.list_staged_takes() == []
    assert jobs.get_job(job_id)["status"] == "done"


def test_a_cancel_during_render_keeps_the_previous_audio(
    api_client: TestClient, blocked: _BlockedRender
) -> None:
    project_id, kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)
    blocked.start()
    # From another client: the cancel is honored when the render returns.
    assert api_client.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "running"
    blocked.release()

    assert jobs.get_job(job_id)["status"] == "cancelled"
    assert [take["id"] for take in storage.list_takes(project_id)] == [kept]
    assert storage.list_staged_takes() == []
    assert storage.load_project(project_id)["active_take_id"] == kept
    assert storage.take_audio_path(project_id, kept).is_file()


def test_a_deleted_project_is_not_resurrected_by_a_late_result(
    api_client: TestClient, blocked: _BlockedRender
) -> None:
    project_id, _kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)
    blocked.start()
    assert api_client.delete(f"/api/projects/{project_id}").status_code == 204
    blocked.release()

    row = jobs.get_job(job_id)
    assert row["status"] == "error"
    assert "project deleted" in row["error"]
    assert not storage.project_dir(project_id).exists()
    assert storage.list_staged_takes() == []


def test_death_before_promotion_leaves_no_take_and_retries_cleanly(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id, kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)

    def render_then_die(**kwargs):
        _REAL_RUN_JOB(**kwargs)
        raise _Crash()

    monkeypatch.setattr(mock_worker_module, "run_job", render_then_die)
    with pytest.raises(_Crash):
        jobs.run_claimed_job(jobs.claim_next_queued_job())
    # The render is quarantined in staging, not listed as a take.
    assert [take["id"] for take in storage.list_takes(project_id)] == [kept]
    assert len(storage.list_staged_takes()) == 1

    monkeypatch.setattr(mock_worker_module, "run_job", _REAL_RUN_JOB)
    jobs.reclaim_stale_jobs(stale_after=-1)
    assert len(jobs.cleanup_staging()) == 1
    claimed = jobs.claim_next_queued_job()
    jobs.run_claimed_job(claimed)
    done = jobs.get_job(job_id)
    assert done["status"] == "done"
    assert len(storage.list_takes(project_id)) == 2
    assert storage.list_staged_takes() == []


def test_a_live_attempts_staging_survives_cleanup(
    api_client: TestClient, blocked: _BlockedRender
) -> None:
    project_id, _kept = _project_with_a_kept_take(api_client)
    _enqueue(api_client, project_id)
    blocked.start()
    assert jobs.cleanup_staging() == []
    assert len(storage.list_staged_takes()) == 1
    blocked.release()


def test_a_full_disk_at_commit_fails_cleanly(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id, kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)

    def disk_full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(storage, "write_staged_meta", disk_full)
    jobs.run_claimed_job(jobs.claim_next_queued_job())

    row = jobs.get_job(job_id)
    assert row["status"] == "error"
    assert "disk is full" in row["error"]
    assert f"[job {job_id}" in row["error"]
    assert storage.list_staged_takes() == []
    # The kept take and its audio are untouched; the failure is a take with
    # `error` and no audio.
    assert storage.load_project(project_id)["active_take_id"] == kept
    assert storage.take_audio_path(project_id, kept).is_file()
    failed = storage.get_take(project_id, row["take_id"])
    assert failed["error"] is not None
    assert not (storage.take_dir(project_id, row["take_id"]) / "mix.wav").exists()


def test_a_reclaimed_attempts_late_commit_is_rejected(
    api_client: TestClient, blocked: _BlockedRender
) -> None:
    project_id, kept = _project_with_a_kept_take(api_client)
    job_id = _enqueue(api_client, project_id)
    blocked.start()
    # The heartbeat looked stale: the job is requeued and claimed again.
    jobs.reclaim_stale_jobs(stale_after=-1)
    blocked.release()
    assert [take["id"] for take in storage.list_takes(project_id)] == [kept]
    assert jobs.get_job(job_id)["status"] == "queued"
    assert storage.list_staged_takes() == []
