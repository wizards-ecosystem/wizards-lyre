"""Phase 6 (SPEC.md sec 12): project-level favorites. Library search is
api_client-side only (no backend endpoint), so it isn't covered here.
"""

from __future__ import annotations


from fastapi.testclient import TestClient



def test_new_project_is_not_favorited(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Fave Test"}).json()
    project_id = project["id"]

    listed = api_client.get("/api/projects").json()
    entry = next(p for p in listed if p["id"] == project_id)
    assert entry["favorite"] is False

    detail = api_client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is False


def test_patch_favorite_flips_it_in_list_and_detail(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Fave Test 2"}).json()
    project_id = project["id"]

    resp = api_client.patch(f"/api/projects/{project_id}", json={"favorite": True})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is True

    listed = api_client.get("/api/projects").json()
    entry = next(p for p in listed if p["id"] == project_id)
    assert entry["favorite"] is True

    detail = api_client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is True

    # flip back off
    resp = api_client.patch(f"/api/projects/{project_id}", json={"favorite": False})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    detail = api_client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is False


def test_patch_title_alone_leaves_favorite_unchanged(api_client: TestClient) -> None:
    project = api_client.post("/api/projects", json={"title": "Fave Test 3"}).json()
    project_id = project["id"]

    resp = api_client.patch(f"/api/projects/{project_id}", json={"favorite": True})
    assert resp.status_code == 200
    assert resp.json()["favorite"] is True

    resp = api_client.patch(f"/api/projects/{project_id}", json={"title": "renamed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "renamed"
    assert body["favorite"] is True

    detail = api_client.get(f"/api/projects/{project_id}").json()
    assert detail["project"]["favorite"] is True
