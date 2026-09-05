"""Project-title normalization reconciled from the approved rename branch."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import storage


def test_patch_title_renames_project_everywhere(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Before Rename"}).json()
    project_id = project["id"]

    response = api_client.patch(f"/api/projects/{project_id}", json={"title": "After Rename"})

    assert response.status_code == 200
    assert response.json()["title"] == "After Rename"
    assert storage.load_project(project_id)["title"] == "After Rename"
    assert api_client.get(f"/api/projects/{project_id}").json()["project"]["title"] == (
        "After Rename"
    )
    listed = api_client.get("/api/projects").json()
    assert next(item for item in listed if item["id"] == project_id)["title"] == "After Rename"


def test_patch_title_leaves_other_fields_untouched(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Rename Fields"}).json()

    response = api_client.patch(f"/api/projects/{project['id']}", json={"title": "Renamed Fields"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Renamed Fields"
    assert body["created_at"] == project["created_at"]
    assert body["dit_profile"] == project["dit_profile"]
    assert body["favorite"] is False


@pytest.mark.parametrize("title", ["", "   ", "\t \n"])
def test_patch_blank_title_falls_back_to_untitled(api_client: TestClient, title: str) -> None:
    project = api_client.post("/api/projects", json={"title": "Real Title"}).json()

    response = api_client.patch(f"/api/projects/{project['id']}", json={"title": title})

    assert response.status_code == 200
    assert response.json()["title"] == "Untitled"
    assert storage.load_project(project["id"])["title"] == "Untitled"


@pytest.mark.parametrize(
    ("title", "expected"),
    [(None, "Untitled"), ("   ", "Untitled"), ("  Padded  ", "Padded")],
)
def test_create_and_patch_share_title_normalization(
    api_client: TestClient, title: str | None, expected: str
) -> None:
    created = api_client.post("/api/projects", json={"title": title}).json()
    assert created["title"] == expected

    response = api_client.patch(
        f"/api/projects/{created['id']}", json={"title": "  Padded Again  "}
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Padded Again"
