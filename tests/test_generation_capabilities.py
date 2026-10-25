"""SPEC.md sec 16.6: truthful readiness and curated presets.

Readiness keeps installed, files present, validated, running, loading, busy,
and unavailable apart. Presets are deterministic, preserve the user's own
fields, and never touch lyrics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import capabilities, jobs, storage
from server.generation import presets
from tests.helpers import wait_for_job
from worker.acestep_worker import settings as worker_settings


def _profile(report: dict, profile: str) -> dict:
    return next(entry for entry in report["profiles"] if entry["id"] == profile)


def test_a_worker_that_never_reported_is_stopped_not_ready(api_client: TestClient) -> None:
    report = api_client.get("/v1/capabilities").json()
    assert report["worker"]["state"] == "stopped"
    assert "./scripts/lyre worker" in report["worker"]["reason"]
    assert report["lm"] == {"model": "acestep-5Hz-lm-1.7B", "selectable": False}
    assert [profile["id"] for profile in report["profiles"]] == ["iterate", "polish", "quality"]


def test_ready_busy_loading_and_unavailable_are_distinct(api_client: TestClient) -> None:
    jobs.publish_worker_status(True, "worker: 'iterate' DiT + LM currently loaded", "iterate")
    assert capabilities.worker_state()["state"] == "ready"

    project_id = storage.create_project(title="Busy")["id"]
    jobs.enqueue_job(project_id, {"action": "generate"})
    claimed = jobs.claim_next_queued_job()
    assert capabilities.worker_state()["state"] == "busy"
    jobs.set_stage(claimed["id"], "loading_model")
    assert capabilities.worker_state()["state"] == "loading"

    jobs.publish_worker_status(False, "CUDA is not available on this machine", None)
    state = capabilities.worker_state()
    assert state["state"] == "unavailable"
    assert state["reason"] == "CUDA is not available on this machine"


def test_a_stale_worker_reads_as_stopped(api_client: TestClient) -> None:
    jobs.publish_worker_status(True, "ready", "iterate")
    with jobs._connect() as conn:
        conn.execute("UPDATE worker_status SET updated_at = '2000-01-01T00:00:00+00:00'")
    assert capabilities.worker_state()["state"] == "stopped"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({}, "missing"),
        ({"config.json": b"{}"}, "incomplete"),
        ({"model.safetensors": b"x"}, "incomplete"),
        ({"config.json": b"{}", "model.safetensors": b"x"}, "present"),
        ({"config.json": b"{}", "model.safetensors.index.json": b"{}"}, "present"),
    ],
)
def test_checkpoint_files_are_reported_from_the_filesystem(
    tmp_path: Path, files: dict[str, bytes], expected: str
) -> None:
    root = tmp_path / "checkpoints"
    if files:
        directory = root / "acestep-v15-turbo"
        directory.mkdir(parents=True)
        for name, data in files.items():
            (directory / name).write_bytes(data)
    assert worker_settings.checkpoint_status("acestep-v15-turbo", root)["state"] == expected


def test_missing_weights_make_a_profile_unsupported_with_a_reason(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LYRE_WORKER", "acestep")
    monkeypatch.setattr(worker_settings, "CHECKPOINTS_ROOT", tmp_path / "empty")
    report = api_client.get("/api/capabilities").json()
    iterate = _profile(report, "iterate")
    assert iterate["files"] == "missing"
    assert iterate["supported"] is False
    assert "acestep-v15-turbo" in iterate["reason"]
    assert iterate["validated"] is False


def test_validated_means_a_job_on_that_profile_completed(client: TestClient) -> None:
    before = _profile(client.get("/v1/capabilities").json(), "iterate")
    assert before["validated"] is False
    project_id = client.post("/api/projects", json={"title": "Validate"}).json()["id"]
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    assert wait_for_job(client, job["id"])["status"] == "done"
    after = _profile(client.get("/v1/capabilities").json(), "iterate")
    assert after["validated"] is True
    assert _profile(client.get("/v1/capabilities").json(), "polish")["validated"] is False


def test_remote_profiles_do_not_claim_local_files(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LYRE_WORKER", "remote")
    report = api_client.get("/v1/capabilities").json()
    assert report["worker"]["backend"] == "remote"
    assert {profile["files"] for profile in report["profiles"]} == {"remote"}


def test_profile_labels_describe_what_each_profile_is(api_client: TestClient) -> None:
    report = api_client.get("/v1/capabilities").json()
    polish = _profile(report, "polish")
    assert polish["checkpoint"] == "acestep-v15-sft"
    assert (polish["inference_steps"], polish["guidance_scale"]) == (50, 7.5)
    assert "not always better" in polish["summary"]
    quality = _profile(report, "quality")
    assert quality["cpu_offload"] is True
    assert "offload" in quality["summary"].lower()


def test_a_nearly_full_disk_refuses_a_render_before_gpu_work(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.jobs import queue

    monkeypatch.setattr(queue, "free_disk_bytes", lambda: 10 * 1024 * 1024)
    project_id = api_client.post("/api/projects", json={"title": "Full"}).json()["id"]
    resp = api_client.post(f"/v1/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "disk_full"
    assert api_client.get(f"/v1/jobs?project_id={project_id}").json()["data"] == []


# --- presets ---------------------------------------------------------------


def test_the_registry_is_small_versioned_and_fully_described() -> None:
    listed = presets.list_presets()
    assert [preset["id"] for preset in listed] == [
        "lyric_song",
        "intimate_acoustic_song",
        "energetic_electronic_song",
        "exploration_ambience",
        "combat_cue",
        "menu_theme",
    ]
    for preset in listed:
        assert isinstance(preset["version"], int)
        for key in ("name", "purpose", "caption", "example"):
            assert preset[key].strip()
        assert set(preset["fields"]) <= set(presets.PRESET_FIELDS)
        assert "lyrics" not in preset["fields"]


def test_expansion_is_deterministic() -> None:
    plan = storage.default_plan()
    assert presets.apply_preset(plan, "combat_cue") == presets.apply_preset(plan, "combat_cue")


def test_a_preset_keeps_user_fields_and_never_touches_lyrics() -> None:
    plan = {
        **storage.default_plan(),
        "caption": "my caption",
        "lyrics": "[Verse]\nmy words",
        "bpm": 100,
        "field_origins": {
            **storage.default_plan()["field_origins"],
            "caption": "user",
            "lyrics": "user",
            "bpm": "user",
        },
    }
    result = presets.apply_preset(plan, "combat_cue")
    proposed = result["plan"]
    assert proposed["caption"] == "my caption"
    assert proposed["lyrics"] == "[Verse]\nmy words"
    assert proposed["bpm"] == 100
    # Lyrics are words someone wrote, so the instrumental advice is not applied.
    assert proposed["instrumental"] is False
    assert any("stays a song" in warning for warning in result["warnings"])
    # Unset fields were filled and labelled.
    assert proposed["duration_sec"] == 90
    assert proposed["field_origins"]["duration_sec"] == "preset"
    assert proposed["preset"] == {"id": "combat_cue", "version": 1}


def test_preview_writes_nothing_and_apply_is_revision_checked(api_client: TestClient) -> None:
    project_id = api_client.post("/api/projects", json={"title": "Preset"}).json()["id"]
    before = api_client.get(f"/v1/projects/{project_id}").json()["plan"]
    preview = api_client.post(
        f"/v1/projects/{project_id}/preset", json={"preset_id": "menu_theme"}
    ).json()
    assert preview["applied"] is False
    assert {change["field"] for change in preview["changes"]} >= {"caption", "bpm", "instrumental"}
    assert api_client.get(f"/v1/projects/{project_id}").json()["plan"] == before

    stale = api_client.post(
        f"/v1/projects/{project_id}/preset",
        json={"preset_id": "menu_theme", "apply": True, "expected_revision": 99},
    )
    assert stale.status_code == 409
    applied = api_client.post(
        f"/api/projects/{project_id}/preset",
        json={"preset_id": "menu_theme", "apply": True, "expected_revision": before["revision"]},
    ).json()
    assert applied["applied"] is True
    stored = api_client.get(f"/v1/projects/{project_id}").json()["plan"]
    assert stored["preset"] == {"id": "menu_theme", "version": 1}
    assert stored["instrumental"] is True
    assert stored["field_origins"]["caption"] == "preset"

    unknown = api_client.post(f"/v1/projects/{project_id}/preset", json={"preset_id": "nope"})
    assert unknown.status_code == 404


def test_the_applied_preset_is_recorded_on_the_take(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Preset take"}).json()["id"]
    client.post(
        f"/api/projects/{project_id}/preset",
        json={"preset_id": "exploration_ambience", "apply": True},
    )
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    done = wait_for_job(client, job["id"])
    take = storage.get_take(project_id, done["take_id"])
    assert take["provenance"]["requested"]["preset"] == {"id": "exploration_ambience", "version": 1}
    assert take["provenance"]["brief"]["origins"]["bpm"] == "preset"
    assert take["provenance"]["actual"]["bpm"] == 70
