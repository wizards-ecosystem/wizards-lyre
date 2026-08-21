"""Phase 6 (SPEC.md sec 12): project-level favorites. Library search is
client-side only (no backend endpoint), so it isn't covered here.
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


def test_new_project_is_not_favorited(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Fave Test"}).json()
    project_id = project["id"]

    listed = client.get("/api/projects").json()
    entry = next(p for p in listed if p["id"] == project_id)
    assert entry["favorite"] is False

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is False


def test_patch_favorite_flips_it_in_list_and_detail(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Fave Test 2"}).json()
    project_id = project["id"]

    resp = client.patch(f"/api/projects/{project_id}", json={"favorite": True})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is True

    listed = client.get("/api/projects").json()
    entry = next(p for p in listed if p["id"] == project_id)
    assert entry["favorite"] is True

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is True

    # flip back off
    resp = client.patch(f"/api/projects/{project_id}", json={"favorite": False})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is False


def test_patch_title_alone_leaves_favorite_unchanged(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Fave Test 3"}).json()
    project_id = project["id"]

    resp = client.patch(f"/api/projects/{project_id}", json={"favorite": True})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is True

    resp = client.patch(f"/api/projects/{project_id}", json={"title": "renamed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "renamed"
    assert body["favorite"] is True

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is True
