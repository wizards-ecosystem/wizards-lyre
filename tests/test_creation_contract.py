"""SPEC.md sec 16.3: the creation contract, end to end.

Precedence is asserted on the arguments the fake ACE-Step actually received
(`GenerationParams`, `create_sample`, `format_sample`), not only on what was
stored, and the mock worker is held to the same creative fields as the real
adapter. No GPU: the real adapter runs against the fakes from
tests/test_acestep_worker_adapter.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.generation import brief as creation
from tests.helpers import wait_for_job
from tests.test_acestep_worker_adapter import _RESET_STATE, _install_fake_acestep
from worker import acestep_worker, mock_worker


@pytest.fixture(autouse=True)
def _reset_adapter_state():
    acestep_worker._STATE.update(_RESET_STATE)
    yield
    acestep_worker._STATE.update(_RESET_STATE)


def _plan(**overrides: Any) -> dict[str, Any]:
    plan = storage.default_plan()
    plan.update(overrides)
    return plan


def _intent(plan: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return creation.validate_creation(creation.creation_input(plan, **kwargs))


def _run_real(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
    job_extra: dict[str, Any] | None = None,
    **fake_kwargs: Any,
) -> tuple[list[tuple], dict, dict | None]:
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log, **fake_kwargs)
    job = {
        "action": "generate",
        "dit_profile": "iterate",
        "seed": 7,
        "src_audio": None,
        "creation": _intent(plan),
        **(job_extra or {}),
    }
    meta, plan_patch, _ = acestep_worker.run_job(
        job=job, plan=plan, take_id="t", take_dir=tmp_path / "take"
    )
    return log, meta, plan_patch


def _params(log: list[tuple]) -> Any:
    return next(entry for entry in log if entry[0] == "generate_music")[3]


def test_explicit_fields_survive_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 1/3: the planner suggests different metas; the user's win."""
    plan = _plan(
        query="stormy sea shanty",
        bpm=88,
        keyscale="d minor",
        timesignature="6/8",
        duration_sec=45,
        vocal_language="fr",
        field_origins={
            **creation.default_origins(),
            "bpm": "user",
            "keyscale": "user",
            "timesignature": "user",
            "duration_sec": "user",
            "vocal_language": "user",
        },
    )
    log, meta, plan_patch = _run_real(tmp_path, monkeypatch, plan)

    create_call = next(entry for entry in log if entry[0] == "create_sample")
    assert create_call[4] == "fr"  # the explicit language constrains the lyrics
    params = _params(log)
    assert params.bpm == 88
    assert params.keyscale == "D minor"  # documented transport normalization
    assert params.timesignature == "6/8"
    assert params.duration == 45
    assert params.vocal_language == "fr"
    assert params.use_cot_metas is False
    assert params.use_cot_language is False
    # The planner's caption and lyrics filled the unset text fields.
    assert params.caption == "auto: stormy sea shanty"
    assert plan_patch is not None
    assert set(plan_patch["field_origins"]) == {"caption", "lyrics"}
    assert meta["provenance"]["brief"]["origins"]["bpm"] == "user"


def test_unspecified_fields_are_filled_and_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4: defaults are not locks. The planner fills them (origin
    `planner`) and what it leaves unset CoT fills (origin `model`)."""
    plan = _plan(query="lofi study beat", bpm=None, keyscale=None, timesignature="")
    log, meta, plan_patch = _run_real(
        tmp_path,
        monkeypatch,
        plan,
        create_sample_result={"caption": "lofi", "lyrics": "[Verse]\nla", "bpm": 72},
    )
    params = _params(log)
    assert params.bpm == 72
    assert params.keyscale == ""
    assert params.use_cot_metas is True
    origins = meta["provenance"]["actual_origins"]
    assert origins["bpm"] == "planner"
    assert origins["keyscale"] == "model"
    assert meta["provenance"]["actual"]["keyscale"] == "E minor"
    assert origins["duration_sec"] == "default"  # the application default rendered
    assert plan_patch is not None
    assert plan_patch["bpm"] == 72
    assert "keyscale" not in plan_patch


def test_supplied_lyrics_are_kept_verbatim_while_the_idea_is_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 2: with user lyrics the planner is format_sample, constrained by
    the explicit metas, and its reformatted lyrics are discarded."""
    lyrics = "[Verse]\r\nsalt on the rope\r\n[Chorus]\r\nhaul away"
    plan = _plan(
        query="sea shanty",
        lyrics=lyrics,
        bpm=100,
        field_origins={**creation.default_origins(), "lyrics": "user", "bpm": "user"},
    )
    log, meta, _ = _run_real(tmp_path, monkeypatch, plan)

    format_call = next(entry for entry in log if entry[0] == "format_sample")
    assert format_call[2] == "sea shanty"  # no user caption: the idea stands in
    assert format_call[3] == lyrics.replace("\r\n", "\n")
    assert format_call[4] == {"bpm": 100}
    assert not any(entry[0] == "create_sample" for entry in log)
    params = _params(log)
    assert params.lyrics == "[Verse]\nsalt on the rope\n[Chorus]\nhaul away"
    assert params.bpm == 100
    assert params.caption == "formatted: sea shanty"
    brief = meta["provenance"]["brief"]
    assert brief["lyrics_original"] == lyrics
    assert brief["origins"]["lyrics"] == "user"


def test_user_caption_is_kept_when_rewrite_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 5: a user caption survives planning and rendering."""
    plan = _plan(
        query="desert drive",
        caption="dusty slide guitar, slow",
        caption_rewrite=False,
        field_origins={**creation.default_origins(), "caption": "user"},
    )
    log, meta, _ = _run_real(tmp_path, monkeypatch, plan)
    params = _params(log)
    assert params.caption == "dusty slide guitar, slow"
    assert params.use_cot_caption is False
    assert meta["caption"] == "dusty slide guitar, slow"


def test_a_resolved_brief_renders_without_replanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(query="first idea")
    resolved = creation.apply_planner(
        _intent(plan),
        {"caption": "frozen caption", "lyrics": "[Verse]\nfrozen", "bpm": 111},
        {"name": "create_sample", "lm_model": "x", "deterministic": False},
    )
    log, meta, _ = _run_real(
        tmp_path, monkeypatch, _plan(query="a later edit"), {"resolved": {"brief": resolved}}
    )
    assert not any(entry[0] in ("create_sample", "format_sample") for entry in log)
    params = _params(log)
    assert params.caption == "frozen caption"
    assert params.bpm == 111
    assert meta["provenance"]["brief"]["content_hash"] == resolved["content_hash"]


def test_mock_and_real_adapter_build_the_same_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mock is held to the real adapter's creative GenerationParams
    fields for the same frozen input."""
    cases = [
        _plan(query="neon city", caption_rewrite=True),
        _plan(caption="string quartet", lyrics="", instrumental=True, caption_rewrite=False),
        _plan(
            query="fado",
            lyrics="[Verse]\nsaudade",
            vocal_language="pt",
            field_origins={
                **creation.default_origins(),
                "lyrics": "user",
                "vocal_language": "user",
            },
        ),
    ]
    keys = (
        "caption",
        "lyrics",
        "bpm",
        "keyscale",
        "duration",
        "instrumental",
        "vocal_language",
        "timesignature",
        "thinking",
        "use_cot_caption",
        "use_cot_language",
        "use_cot_metas",
    )
    for index, plan in enumerate(cases):
        intent = _intent(plan)
        # Both planners are replaced by the same fixed result so only the
        # resolution and request construction are compared.
        fixed = {"caption": "planned", "lyrics": "[Verse]\nplanned", "bpm": 90}
        monkeypatch.setattr(mock_worker, "_mock_planner", lambda _request, fixed=fixed: fixed)
        log, real_meta, _ = _run_real(
            tmp_path / f"real{index}",
            monkeypatch,
            plan,
            create_sample_result=fixed,
            format_sample_result=fixed,
        )
        mock_meta, _, _ = mock_worker.run_job(
            job={
                "action": "generate",
                "dit_profile": "iterate",
                "seed": 7,
                "creation": intent,
            },
            plan=plan,
            take_id="m",
            take_dir=tmp_path / f"mock{index}",
        )
        params = _params(log)
        real_fields = {key: getattr(params, key) for key in keys}
        mock_fields = {key: mock_meta["provenance"]["request"][key] for key in keys}
        assert real_fields == mock_fields, index
        assert real_meta["provenance"]["request"] == mock_meta["provenance"]["request"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bpm", 500),
        ("bpm", 10),
        ("duration_sec", 5),
        ("duration_sec", 900),
        ("keyscale", "H major"),
        ("keyscale", "C lydian"),
        ("timesignature", "5/4"),
        ("vocal_language", "klingon"),
    ],
)
def test_unsupported_values_fail_at_enqueue(api_client: TestClient, field: str, value: Any) -> None:
    project_id = api_client.post("/api/projects", json={"title": "Bad"}).json()["id"]
    api_client.put(f"/api/projects/{project_id}/plan", json={"query": "rain", field: value})
    resp = api_client.post(f"/v1/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["param"] == field
    assert api_client.get(f"/v1/jobs?project_id={project_id}").json()["data"] == []


def test_instrumental_conflict_keeps_the_lyrics(api_client: TestClient) -> None:
    project_id = api_client.post("/api/projects", json={"title": "Conflict"}).json()["id"]
    api_client.put(
        f"/api/projects/{project_id}/plan",
        json={"query": "rain", "instrumental": True, "lyrics": "[Verse]\nmy words"},
    )
    resp = api_client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"})
    assert resp.status_code == 400
    assert "Instrumental" in resp.json()["detail"]
    plan = api_client.get(f"/api/projects/{project_id}").json()["plan"]
    assert plan["lyrics"] == "[Verse]\nmy words"


def test_a_changed_idea_after_the_first_render_is_resolved(client: TestClient) -> None:
    """Gap 1 of the completion review: the old rule ignored a new idea once
    caption and lyrics had been filled. Now an edited idea re-plans, and an
    unedited Generate reuses the brief ("another take")."""
    project_id = client.post("/api/projects", json={"title": "Ideas", "query": "rain"}).json()["id"]
    first = wait_for_job(
        client,
        client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()["id"],
    )
    assert first["status"] == "done", first.get("error")
    plan = client.get(f"/api/projects/{project_id}").json()["plan"]
    assert plan["caption"] == "auto: rain"
    assert plan["creation_mode"] == "from_brief"

    another = wait_for_job(
        client,
        client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()["id"],
    )
    take = storage.get_take(project_id, another["take_id"])
    assert take["provenance"]["brief"]["planner"] is None
    assert take["caption"] == "auto: rain"

    client.put(f"/api/projects/{project_id}/plan", json={**plan, "query": "snow"})
    third = wait_for_job(
        client,
        client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()["id"],
    )
    take = storage.get_take(project_id, third["take_id"])
    assert take["caption"] == "auto: snow"
    assert client.get(f"/api/projects/{project_id}").json()["plan"]["caption"] == "auto: snow"


def test_job_body_can_request_a_replan(client: TestClient) -> None:
    project_id = client.post("/api/projects", json={"title": "Replan", "query": "rain"}).json()[
        "id"
    ]
    wait_for_job(
        client,
        client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()["id"],
    )
    replanned = wait_for_job(
        client,
        client.post(
            f"/api/projects/{project_id}/jobs",
            json={"action": "generate", "creation_mode": "from_idea"},
        ).json()["id"],
    )
    take = storage.get_take(project_id, replanned["take_id"])
    assert take["provenance"]["brief"]["planner"]["name"] == "create_sample"

    rejected = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "cover",
            "creation_mode": "from_idea",
            "source_take_id": replanned["take_id"],
        },
    )
    assert rejected.status_code == 400


def test_language_list_matches_upstream() -> None:
    constants = pytest.importorskip("acestep.constants")
    upstream = set(constants.VALID_LANGUAGES)
    ours = set(creation.VALID_LANGUAGES)
    assert ours <= upstream
    # Exactly one upstream code is withheld (see VALID_LANGUAGES).
    assert len(upstream - ours) == 1
    assert (constants.BPM_MIN, constants.BPM_MAX) == (creation.BPM_MIN, creation.BPM_MAX)
    assert (constants.DURATION_MIN, constants.DURATION_MAX) == (
        creation.DURATION_MIN,
        creation.DURATION_MAX,
    )
    for key in constants.VALID_KEYSCALES:
        assert creation.parse_keyscale(key) == key
