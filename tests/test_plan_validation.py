"""SPEC.md sec 7.2 plan hardening: PUT /api/projects/{id}/plan validates and
normalizes the body (server/storage.validate_plan) instead of storing whatever
dict the api_client sent verbatim. Covers: a valid round-trip, defaults filled for
missing keys, unknown keys dropped, each invalid type rejected with HTTP 400,
legacy plans (GET via _normalize_plan) unchanged, and the worker's simple-mode
merge_plan_patch still working after a validated save.

No GPU, no CUDA, no ACE-Step -- mocked worker only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import storage


def _create_project(api_client: TestClient, title: str = "Plan validation") -> str:
    project = api_client.post("/api/projects", json={"title": title}).json()
    return project["id"]


def _valid_plan() -> dict:
    """The exact shape the SPA's Plan type sends (web/src/api.ts): every key
    present with a spec-valid value."""
    return {
        "query": "",
        "caption": "ambient, pads, slow build",
        "negative": ["distortion", "clipping"],
        "lyrics": "[Verse]\nquiet streets tonight",
        "instrumental": False,
        "vocal_language": "en",
        "bpm": 90,
        "keyscale": "D Minor",
        "timesignature": "4/4",
        "duration_sec": 150,
        "sections": [
            {"name": "intro", "start_sec": 0, "end_sec": 10, "lyrics": ""},
            {"name": "verse", "start_sec": 10, "end_sec": 42},
        ],
        "caption_rewrite": True,
    }


def test_valid_plan_roundtrip(api_client: TestClient) -> None:
    """A well-formed plan is accepted unchanged and round-trips through disk
    (no behavior change for well-formed plans)."""
    project_id = _create_project(api_client)
    plan = _valid_plan()

    resp = api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 200
    assert resp.json() == plan

    # What landed on disk matches, and GET returns the same plan.
    on_disk = json.loads(storage.plan_json_path(project_id).read_text(encoding="utf-8"))
    assert on_disk == plan
    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["plan"] == plan


def test_missing_keys_filled_from_defaults(api_client: TestClient) -> None:
    """A partial body gets every missing key filled from default_plan() -- the
    worker (.get() reads) and SPA (exact Plan shapes) never see a hole."""
    project_id = _create_project(api_client)

    resp = api_client.put(
        f"/api/projects/{project_id}/plan", json={"query": "lofi beat", "bpm": 84}
    )
    assert resp.status_code == 200
    body = resp.json()

    expected = storage.default_plan()
    expected["query"] = "lofi beat"
    expected["bpm"] = 84
    assert body == expected

    # Specifically: a body omitting caption_rewrite normalizes to the
    # default_plan() value (True for new plans), not the legacy False.
    assert body["caption_rewrite"] is True
    assert body["sections"] == []
    assert body["vocal_language"] == "en"


def test_unknown_top_level_keys_dropped(api_client: TestClient) -> None:
    """Unknown top-level keys are dropped rather than persisted."""
    project_id = _create_project(api_client)
    plan = _valid_plan()
    plan["not_a_real_field"] = "junk"
    plan["another"] = {"nested": True}

    resp = api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 200
    saved = resp.json()
    assert "not_a_real_field" not in saved
    assert "another" not in saved

    on_disk = json.loads(storage.plan_json_path(project_id).read_text(encoding="utf-8"))
    assert "not_a_real_field" not in on_disk
    assert "another" not in on_disk


def test_unknown_section_keys_dropped(api_client: TestClient) -> None:
    project_id = _create_project(api_client)
    plan = _valid_plan()
    plan["sections"] = [{"name": "intro", "start_sec": 0, "end_sec": 8, "color": "#fff"}]

    resp = api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 200
    assert resp.json()["sections"] == [{"name": "intro", "start_sec": 0, "end_sec": 8}]


# Each case: (field, bad value, substring expected in the 400 detail). The
# message must name the offending field.
_INVALID_PLANS: list[tuple[str, Any, str]] = [
    ("query", 5, "query"),
    ("query", None, "query"),
    ("caption", ["tags"], "caption"),
    ("lyrics", 42, "lyrics"),
    ("vocal_language", None, "vocal_language"),
    ("timesignature", 4, "timesignature"),
    # negative as a bare string (the canonical malformed plan), a dict, and a
    # list containing a non-string.
    ("negative", "distortion", "negative"),
    ("negative", {"a": 1}, "negative"),
    ("negative", ["fine", 3], "negative[1]"),
    ("instrumental", "false", "instrumental"),
    ("instrumental", 1, "instrumental"),
    ("caption_rewrite", "true", "caption_rewrite"),
    ("caption_rewrite", 0, "caption_rewrite"),
    ("bpm", "120", "bpm"),
    ("bpm", 120.5, "bpm"),
    ("bpm", True, "bpm"),
    ("keyscale", 5, "keyscale"),
    ("keyscale", ["C Major"], "keyscale"),
    ("duration_sec", "120", "duration_sec"),
    ("duration_sec", None, "duration_sec"),
    ("duration_sec", True, "duration_sec"),
    ("sections", "intro", "sections"),
    ("sections", {"name": "intro"}, "sections"),
    ("sections", ["intro"], "sections[0]"),
    ("sections", [{"start_sec": 0, "end_sec": 8}], "sections[0].name"),
    ("sections", [{"name": 5, "start_sec": 0, "end_sec": 8}], "sections[0].name"),
    ("sections", [{"name": "intro", "start_sec": "0", "end_sec": 8}], "sections[0].start_sec"),
    ("sections", [{"name": "intro", "start_sec": 0}], "sections[0].end_sec"),
    ("sections", [{"name": "intro", "start_sec": 0, "end_sec": 8, "lyrics": ["x"]}], "sections[0].lyrics"),
]


@pytest.mark.parametrize("field,bad_value,expected_detail", _INVALID_PLANS)
def test_invalid_plan_rejected_with_400(
    api_client: TestClient, field: str, bad_value: Any, expected_detail: str
) -> None:
    project_id = _create_project(api_client)
    plan = _valid_plan()
    plan[field] = bad_value

    resp = api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 400, resp.text
    assert expected_detail in resp.json()["detail"]


def test_invalid_plan_leaves_stored_plan_untouched(api_client: TestClient) -> None:
    project_id = _create_project(api_client)
    original = _valid_plan()
    assert api_client.put(f"/api/projects/{project_id}/plan", json=original).status_code == 200

    bad = _valid_plan()
    bad["bpm"] = "fast"
    resp = api_client.put(f"/api/projects/{project_id}/plan", json=bad)
    assert resp.status_code == 400

    assert api_client.get(f"/api/projects/{project_id}").json()["plan"] == original


def test_non_finite_numbers_rejected() -> None:
    # Storage-level check (HTTP JSON parsers may reject NaN tokens earlier):
    # NaN/+-inf would serialize back out as non-standard JSON, so validate_plan
    # refuses them like any other bad duration_sec.
    with pytest.raises(ValueError, match="duration_sec"):
        storage.validate_plan({**storage.default_plan(), "duration_sec": float("nan")})
    with pytest.raises(ValueError, match="sections"):
        storage.validate_plan(
            {
                **storage.default_plan(),
                "sections": [{"name": "intro", "start_sec": float("inf"), "end_sec": 8}],
            }
        )


def test_non_dict_plan_rejected() -> None:
    with pytest.raises(ValueError, match="plan"):
        storage.validate_plan(["not", "a", "dict"])
    with pytest.raises(ValueError, match="plan"):
        storage.validate_plan(None)


def test_legacy_plan_without_caption_rewrite_still_loads_false(
    api_client: TestClient, tmp_path: Path
) -> None:
    """GET must not change for legacy plans: a plan.json written before
    caption_rewrite existed has no key at all and still loads as False via
    _normalize_plan -- even though a *new* PUT omitting the key fills it from
    default_plan() as True."""
    project_id = _create_project(api_client)

    plan_path = storage.plan_json_path(project_id)
    legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    del legacy_plan["caption_rewrite"]
    plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")

    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["plan"]["caption_rewrite"] is False


def test_get_stays_lenient_for_legacy_malformed_plans(api_client: TestClient) -> None:
    """Validation guards only the PUT write path. _normalize_plan (the GET
    path) must stay lenient, so plan.json files already on disk with shapes
    this change now rejects keep loading unchanged instead of erroring."""
    project_id = _create_project(api_client)

    plan_path = storage.plan_json_path(project_id)
    legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    del legacy_plan["caption_rewrite"]
    legacy_plan["bpm"] = "120"  # rejected on PUT today, but already on disk
    legacy_plan["some_old_key"] = "kept"
    plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")

    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    plan = resp.json()["plan"]
    assert plan["bpm"] == "120"
    assert plan["some_old_key"] == "kept"
    assert plan["caption_rewrite"] is False


def test_simple_mode_merge_plan_patch_after_validated_save(api_client: TestClient) -> None:
    """SPEC.md sec 7.2 simple mode: after a validated query-only plan is
    saved, the worker's plan_patch (the LM-filled delta server.jobs merges via
    storage.merge_plan_patch when the job finishes) still applies cleanly onto
    it -- both the filled fields and the untouched validated fields survive."""
    project_id = _create_project(api_client)
    simple_plan = storage.default_plan()
    simple_plan["query"] = "dreamy synthwave drive"
    assert api_client.put(f"/api/projects/{project_id}/plan", json=simple_plan).status_code == 200

    # Same shape as worker.acestep_worker's simple-mode plan_patch.
    plan_patch = {
        "caption": "auto: dreamy synthwave drive",
        "lyrics": "[Verse]\nneon lights",
        "bpm": 104,
        "keyscale": "A Minor",
        "duration_sec": 118.4,
        "vocal_language": "en",
        "timesignature": "4/4",
    }
    merged = storage.merge_plan_patch(project_id, plan_patch)

    assert merged["query"] == "dreamy synthwave drive"  # untouched by the patch
    for key, value in plan_patch.items():
        assert merged[key] == value
    assert merged["caption_rewrite"] is True  # validated-save field survives

    # The merge result is what GET now serves, and it is still a valid plan.
    assert api_client.get(f"/api/projects/{project_id}").json()["plan"] == merged
    assert storage.validate_plan(merged) == merged
