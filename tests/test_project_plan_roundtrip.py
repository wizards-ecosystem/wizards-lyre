"""SPEC.md sec 11: create project -> plan round-trip on disk.

Thin, focused coverage of the specific SPEC sec 11 bullet; the broader
project/plan/take surface (list, patch, path jail, etc.) is covered by
tests/test_phase1_api.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")
    with TestClient(app) as c:
        yield c


def test_project_plan_roundtrip(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Roundtrip"}).json()
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
    }
    resp = client.put(f"/api/projects/{project_id}/plan", json=plan)
    assert resp.status_code == 200
    assert resp.json() == plan

    resp = client.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plan"] == plan
    assert body["project"]["id"] == project_id
