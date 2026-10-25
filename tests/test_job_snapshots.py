"""SPEC.md sec 16.4/16.5: frozen submissions, fenced attempts, and retries.

The queue is driven by hand (`claim_next_queued_job` / `run_claimed_job`)
so each test controls exactly when an attempt starts, crashes, or is
reclaimed. Mock worker only.
"""

from __future__ import annotations

import json
from contextlib import closing
from typing import Any

import pytest
from fastapi.testclient import TestClient

import worker.mock_worker as mock_worker_module
from server import jobs, storage

_REAL_RUN_JOB = mock_worker_module.run_job
_REAL_PLANNER = mock_worker_module._mock_planner


def _restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the real mock backend back without `monkeypatch.undo()`, which
    would also undo the fixture's database and projects paths."""
    monkeypatch.setattr(mock_worker_module, "run_job", _REAL_RUN_JOB)
    monkeypatch.setattr(mock_worker_module, "_mock_planner", _REAL_PLANNER)


class _Crash(BaseException):
    """Stands in for the worker process dying: not an `Exception`, so the
    runner's failure handling never sees it and the row stays `running`."""


def _project(api_client: TestClient, **plan: Any) -> str:
    project_id = api_client.post("/api/projects", json={"title": "Snapshots"}).json()["id"]
    if plan:
        api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    return project_id


def _enqueue(api_client: TestClient, project_id: str, **body: Any) -> dict:
    resp = api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate", **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _run_next() -> dict:
    claimed = jobs.claim_next_queued_job()
    assert claimed is not None
    jobs.run_claimed_job(claimed)
    return jobs.get_job(claimed["id"])


def _crash_once(monkeypatch: pytest.MonkeyPatch, *, after_resolved: bool) -> None:
    real = _REAL_RUN_JOB

    def crashing(job, plan, take_id, take_dir, **kwargs):
        if not after_resolved:
            raise _Crash()

        def resolved_then_crash(brief):
            kwargs["on_resolved"](brief)
            raise _Crash()

        return real(
            job=job,
            plan=plan,
            take_id=take_id,
            take_dir=take_dir,
            **{**kwargs, "on_resolved": resolved_then_crash},
        )

    monkeypatch.setattr(mock_worker_module, "run_job", crashing)


def _count_planner_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    real = _REAL_PLANNER

    def counting(request):
        calls.append(request)
        return real(request)

    monkeypatch.setattr(mock_worker_module, "_mock_planner", counting)
    return calls


def test_a_queued_request_ignores_later_edits(api_client: TestClient) -> None:
    """Gap 3 of the completion review: queue A, edit the plan to B, run --
    A renders A."""
    project_id = _project(api_client, caption="A: brass fanfare", caption_rewrite=False)
    job = _enqueue(api_client, project_id)
    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    api_client.put(f"/api/projects/{project_id}/plan", json={**plan, "caption": "B: harp"})

    done = _run_next()
    assert done["id"] == job["id"]
    take = storage.get_take(project_id, done["take_id"])
    assert take["caption"] == "A: brass fanfare"
    assert take["provenance"]["brief"]["caption"] == "A: brass fanfare"


def test_distinct_queued_requests_keep_their_own_brief(api_client: TestClient) -> None:
    project_id = _project(api_client, caption="A", caption_rewrite=False)
    first = _enqueue(api_client, project_id)
    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    api_client.put(f"/api/projects/{project_id}/plan", json={**plan, "caption": "B"})
    second = _enqueue(api_client, project_id)

    captions = {}
    for _ in range(2):
        done = _run_next()
        captions[done["id"]] = storage.get_take(project_id, done["take_id"])["caption"]
    assert captions == {first["id"]: "A", second["id"]: "B"}


def test_retry_after_the_brief_was_frozen_never_replans_or_reseeds(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = _project(api_client, query="glass harmonica lullaby")
    job = _enqueue(api_client, project_id, seed=-1)
    planner_calls = _count_planner_calls(monkeypatch)
    _crash_once(monkeypatch, after_resolved=True)
    claimed = jobs.claim_next_queued_job()
    with pytest.raises(_Crash):
        jobs.run_claimed_job(claimed)

    frozen = jobs.read_job(job["id"])["resolved"]
    assert frozen is not None
    assert frozen["schema"] == "lyre.resolved/1"
    assert isinstance(frozen["seed"], int) and frozen["seed"] != -1
    assert len(planner_calls) == 1

    # The worker died; reclaim requeues it for another attempt.
    _restore(monkeypatch)
    planner_calls = _count_planner_calls(monkeypatch)
    assert jobs.reclaim_stale_jobs(stale_after=-1) == [job["id"]]
    # Meanwhile the user edits the draft: the retry must not see it.
    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    api_client.put(f"/api/projects/{project_id}/plan", json={**plan, "query": "something else"})

    done = _run_next()
    assert done["status"] == "done", done.get("error")
    assert planner_calls == []
    take = storage.get_take(project_id, done["take_id"])
    assert take["seed"] == frozen["seed"]
    assert take["provenance"]["brief"]["content_hash"] == frozen["brief"]["content_hash"]
    assert take["provenance"]["job"]["attempt"] == 2


def test_retry_before_the_brief_was_frozen_may_replan(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = _project(api_client, query="cave drips")
    job = _enqueue(api_client, project_id)
    _crash_once(monkeypatch, after_resolved=False)
    with pytest.raises(_Crash):
        jobs.run_claimed_job(jobs.claim_next_queued_job())
    assert jobs.read_job(job["id"])["resolved"] is None

    _restore(monkeypatch)
    planner_calls = _count_planner_calls(monkeypatch)
    jobs.reclaim_stale_jobs(stale_after=-1)
    done = _run_next()
    assert done["status"] == "done"
    assert len(planner_calls) == 1  # replanned from the frozen intent, not the draft
    assert planner_calls[0]["query"] == "cave drips"


def test_a_fixed_seed_is_frozen_as_requested(api_client: TestClient) -> None:
    project_id = _project(api_client, caption="x")
    job = _enqueue(api_client, project_id, seed=4242)
    done = _run_next()
    assert jobs.read_job(job["id"])["resolved"]["seed"] == 4242
    take = storage.get_take(project_id, done["take_id"])
    assert take["seed"] == 4242
    assert take["provenance"]["requested"]["seed_policy"] == "fixed"


def test_a_stale_attempt_cannot_publish(api_client: TestClient) -> None:
    project_id = _project(api_client, caption="x")
    job = _enqueue(api_client, project_id)
    first = jobs.claim_next_queued_job()
    jobs.reclaim_stale_jobs(stale_after=-1)
    second = jobs.claim_next_queued_job()
    assert second["id"] == job["id"]
    assert first["attempt_token"] != second["attempt_token"]

    with pytest.raises(jobs.JobFenced):
        jobs.freeze_resolved(job["id"], first["attempt_token"], {"brief": {}, "seed": 1})
    assert (
        jobs.finish_done(job["id"], take_id="zombie", attempt_token=first["attempt_token"]) is False
    )
    jobs._set_status(job["id"], "error", error="zombie", attempt_token=first["attempt_token"])
    assert jobs.get_job(job["id"])["status"] == "running"

    # The lost attempt rendering to completion publishes nothing.
    jobs.run_claimed_job(first)
    assert jobs.get_job(job["id"])["status"] == "running"
    assert storage.list_takes(project_id) == []

    jobs.run_claimed_job(second)
    assert jobs.get_job(job["id"])["status"] == "done"
    assert len(storage.list_takes(project_id)) == 1


def test_a_crash_after_the_commit_finishes_with_that_take(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At most one committed take per logical job: the process dies after
    the staged take was renamed into takes/ but before the job row was
    marked done. The retry finds that take and finishes with it."""
    project_id = _project(api_client, caption="x")
    job = _enqueue(api_client, project_id)
    real_promote = storage.promote_staged_take

    def promote_then_crash(*args, **kwargs):
        real_promote(*args, **kwargs)
        raise _Crash()

    monkeypatch.setattr(storage, "promote_staged_take", promote_then_crash)
    with pytest.raises(_Crash):
        jobs.run_claimed_job(jobs.claim_next_queued_job())
    assert jobs.get_job(job["id"])["status"] == "running"
    committed = storage.list_takes(project_id)
    assert len(committed) == 1
    assert committed[0]["job_id"] == job["id"]

    monkeypatch.setattr(storage, "promote_staged_take", real_promote)
    rendered: list[str] = []
    monkeypatch.setattr(
        mock_worker_module,
        "run_job",
        lambda **kwargs: rendered.append("again") or _REAL_RUN_JOB(**kwargs),
    )
    jobs.reclaim_stale_jobs(stale_after=-1)
    done = _run_next()
    assert done["status"] == "done"
    assert done["take_id"] == committed[0]["id"]
    assert rendered == []
    assert len(storage.list_takes(project_id)) == 1


def test_submission_id_replays_the_same_request_and_rejects_another(
    api_client: TestClient,
) -> None:
    project_id = _project(api_client, caption="x")
    first = _enqueue(api_client, project_id, submission_id="browser-1")
    again = _enqueue(api_client, project_id, submission_id="browser-1")
    assert again["id"] == first["id"]
    assert first["submission_id"] == "browser-1"

    conflict = api_client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "submission_id": "browser-1", "seed": 5},
    )
    assert conflict.status_code == 409
    v1_conflict = api_client.post(
        f"/v1/projects/{project_id}/jobs",
        json={"action": "generate", "submission_id": "browser-1", "seed": 5},
    )
    assert v1_conflict.status_code == 409
    assert v1_conflict.json()["error"]["param"] == "submission_id"
    assert len(api_client.get(f"/api/jobs?project_id={project_id}").json()) == 1

    bad = api_client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "submission_id": "has space"},
    )
    assert bad.status_code == 400


def test_v1_idempotency_still_compares_raw_bodies(api_client: TestClient) -> None:
    """The internal canonical request hash does not change the `/v1`
    contract: a reordered but equivalent JSON body under the same
    Idempotency-Key is a different raw body, so it is 409."""
    project_id = _project(api_client, caption="x")
    headers = {"Idempotency-Key": "k1", "Content-Type": "application/json"}
    url = f"/v1/projects/{project_id}/jobs"
    first = api_client.post(url, content='{"action":"generate","seed":3}', headers=headers)
    assert first.status_code == 202
    replay = api_client.post(url, content='{"action":"generate","seed":3}', headers=headers)
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    reordered = api_client.post(url, content='{"seed":3,"action":"generate"}', headers=headers)
    assert reordered.status_code == 409


def test_explicit_retry_repeats_the_frozen_request(api_client: TestClient, monkeypatch) -> None:
    project_id = _project(api_client, caption="original", caption_rewrite=False)
    job = _enqueue(api_client, project_id, seed=11)

    def boom(**_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(mock_worker_module, "run_job", boom)
    failed = _run_next()
    assert failed["status"] == "error"
    assert f"[job {job['id']}" in failed["error"]
    _restore(monkeypatch)

    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    api_client.put(f"/api/projects/{project_id}/plan", json={**plan, "caption": "edited"})
    retry = api_client.post(f"/v1/jobs/{job['id']}/retry")
    assert retry.status_code == 202, retry.text
    assert retry.json()["retry_of"] == job["id"]
    done = _run_next()
    take = storage.get_take(project_id, done["take_id"])
    assert take["caption"] == "original"
    assert take["seed"] == 11
    assert take["provenance"]["job"]["retry_of"] == job["id"]

    # The failed job is kept, not deleted to tidy the list.
    assert jobs.get_job(job["id"])["status"] == "error"
    assert api_client.post(f"/api/jobs/{done['id']}/retry").status_code == 400


def _insert_legacy_job(project_id: str) -> str:
    """A generate row as a pre-section-16 server wrote it: no snapshot."""
    job_id = "legacy" + "0" * 26
    now = jobs._now()
    with closing(jobs._connect()) as conn:
        conn.execute(
            "INSERT INTO jobs (id, project_id, action, dit_profile, status, payload_json, "
            "take_id, error, created_at, updated_at, heartbeat_at, attempts) "
            "VALUES (?, ?, 'generate', 'iterate', 'queued', ?, NULL, NULL, ?, ?, NULL, 0)",
            (job_id, project_id, json.dumps({"action": "generate", "seed": -1}), now, now),
        )
        conn.commit()
    return job_id


def test_an_unsnapshotted_queued_job_waits_for_confirmation(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = _project(api_client, caption="current brief", caption_rewrite=False)
    legacy_id = _insert_legacy_job(project_id)
    called: list[str] = []
    monkeypatch.setattr(mock_worker_module, "run_job", lambda **_k: called.append("ran"))

    assert jobs.claim_next_queued_job() is None
    assert called == []
    held = api_client.get(f"/v1/jobs/{legacy_id}").json()
    assert held["status"] == "queued"
    assert held["stage"] == "needs_confirmation"

    _restore(monkeypatch)
    retry = api_client.post(f"/api/jobs/{legacy_id}/retry").json()
    assert retry["retry_of"] == legacy_id
    assert jobs.get_job(legacy_id)["status"] == "cancelled"
    done = _run_next()
    assert done["id"] == retry["id"]
    take = storage.get_take(project_id, done["take_id"])
    assert take["caption"] == "current brief"
    # A new submission, not a reconstruction labelled as the original.
    assert take["provenance"]["job"]["id"] == retry["id"]


def test_a_reclaimed_remote_render_is_not_resubmitted(api_client: TestClient) -> None:
    project_id = _project(api_client, caption="x")
    job = _enqueue(api_client, project_id)
    claimed = jobs.claim_next_queued_job()
    jobs.record_remote_handle(job["id"], claimed["attempt_token"], "remote-token-1")
    jobs.reclaim_stale_jobs(stale_after=-1)
    row = jobs.get_job(job["id"])
    assert row["status"] == "error"
    assert "remote-token-1" in row["error"]
    assert "repeat paid compute" in row["error"]
    assert jobs.claim_next_queued_job() is None
