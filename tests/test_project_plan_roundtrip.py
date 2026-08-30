"""SPEC.md sec 11: create project -> plan round-trip on disk.

Thin, focused coverage of the specific SPEC sec 11 bullet; the broader
project/plan/take surface (list, patch, path jail, etc.) is covered by
tests/test_phase1_api.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from server import storage


def test_project_plan_roundtrip(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Roundtrip"}).json()
    project_id = project["id"]

    plan = {
        "query": "",
        "caption": "ambient, pads, slow build",
        "negative": ["distortion"],
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "vocal_language": "en",
        "bpm": 90,
        "keyscale": "D Minor",
        "timesignature": "4/4",
        "duration_sec": 150,
        "sections": [{"name": "intro", "start_sec": 0, "end_sec": 10, "lyrics": ""}],
        # SPEC.md sec 9.2: Custom-mode caption-rewrite checkbox -- round-trips
        # like every other plan field.
        "caption_rewrite": True,
    }
    resp = api_client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 200
    assert resp.json() == plan

    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] == plan
    assert body["project"]["id"] == project_id


def test_default_plan_has_caption_rewrite_true(api_client: TestClient) -> None:
    """SPEC.md sec 9.2/7.2: a brand-new project's plan defaults `caption_rewrite`
    to True -- Custom-mode generation may rewrite the user's caption unless
    they explicitly disable it."""
    project = api_client.post("/api/projects", json={"title": "Defaults"}).json()
    project_id = project["id"]

    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["plan"]["caption_rewrite"] is True


def test_plan_missing_caption_rewrite_loads_as_false(
    api_client: TestClient, tmp_path: Path
) -> None:
    """Backward compatibility: a plan.json written to disk before this field
    existed has no `caption_rewrite` key at all -- loading it must default to
    False (SPEC.md sec 7.2: Custom mode previously never rewrote captions),
    not raise or silently omit the key from the response."""
    project = api_client.post("/api/projects", json={"title": "Legacy plan"}).json()
    project_id = project["id"]

    plan_path = storage.plan_json_path(project_id)
    legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    del legacy_plan["caption_rewrite"]
    plan_path.write_text(json.dumps(legacy_plan), encoding="utf-8")

    resp = api_client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["plan"]["caption_rewrite"] is False
