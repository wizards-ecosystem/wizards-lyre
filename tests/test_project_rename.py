"""Renaming a project from the workspace (SPEC.md sec 9): the SPA's
workspace heading renames via the existing PATCH /api/projects/{id}. This
covers the server half -- the rename itself, and the empty/whitespace-only
title normalization that create_project and patch_project share (a rename
must not be able to produce a blank title the library renders as an empty
row).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import storage
from server.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    monkeypatch.setenv("BARD_WORKER", "mock")
    with TestClient(app) as c:
        yield c


def test_patch_title_renames_project_everywhere(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Before Rename"}).json()
    project_id = project["id"]

    resp = client.patch(f"/api/projects/{project_id}", json={"title": "After Rename"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "After Rename"

    # Persisted to project.json (survives any reload -- load_project reads
    # straight off disk) and visible through both read endpoints the SPA
    # uses: the detail view and the library list.
    assert storage.load_project(project_id)["title"] == "After Rename"
    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["title"] == "After Rename"
    listed = client.get("/api/projects").json()
    entry = next(p for p in listed if p["id"] == project_id)
    assert entry["title"] == "After Rename"


def test_patch_title_leaves_other_fields_untouched(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Rename Fields"}).json()
    project_id = project["id"]
    created_at = project["created_at"]
    dit_profile = project["dit_profile"]

    resp = client.patch(f"/api/projects/{project_id}", json={"title": "Renamed Fields"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Renamed Fields"
    assert body["created_at"] == created_at
    assert body["dit_profile"] == dit_profile
    assert body["favorite"] is False


@pytest.mark.parametrize("bad_title", ["", "   ", "\t \n"])
def test_patch_empty_or_whitespace_title_falls_back_to_untitled(
    client: TestClient, bad_title: str
) -> None:
    project = client.post("/api/projects", json={"title": "Real Title"}).json()
    project_id = project["id"]

    resp = client.patch(f"/api/projects/{project_id}", json={"title": bad_title})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Untitled"
    assert storage.load_project(project_id)["title"] == "Untitled"


def test_create_and_patch_normalize_titles_the_same(client: TestClient) -> None:
    # Whitespace-only falls back to 'Untitled' on both paths...
    created = client.post("/api/projects", json={"title": "   "}).json()
    assert created["title"] == "Untitled"

    created = client.post("/api/projects", json={"title": None}).json()
    assert created["title"] == "Untitled"

    # ...and real titles are stripped identically on both paths.
    created = client.post("/api/projects", json={"title": "  Padded  "}).json()
    assert created["title"] == "Padded"

    resp = client.patch(f"/api/projects/{created['id']}", json={"title": "  Padded Again  "})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Padded Again"
