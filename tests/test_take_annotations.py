"""Phase 6 (SPEC.md sec 12): take-level favorites + free-text notes,
mirroring tests/test_project_favorites.py but scoped to a take's meta.json
via PATCH /api/projects/{id}/takes/{take_id}. Take generation itself is
mocked (worker.mock_worker) -- nothing here exercises real ACE-Step.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


def _make_take(client: TestClient) -> tuple[str, str]:
    project = client.post("/api/projects", json={"title": "Take Annotations"}).json()
    project_id = project["id"]
    resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    job = wait_for_job(client, resp.json()["id"])
    assert job["status"] == "done"
    return project_id, job["take_id"]


def test_new_take_is_not_favorited_and_has_no_notes(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is False
    assert take["notes"] == ""


def test_patch_favorite_and_notes_round_trip(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": True, "notes": "great take, keep this seed"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["favorite"] is True
    assert body["notes"] == "great take, keep this seed"

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is True
    assert take["notes"] == "great take, keep this seed"


def test_patch_one_field_leaves_the_other_untouched(client: TestClient) -> None:
    project_id, take_id = _make_take(client)

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"notes": "initial note"},
    )
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    assert resp.json()["notes"] == "initial note"

    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["favorite"] is True
    assert body["notes"] == "initial note"  # untouched by the favorite-only patch

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is True
    assert take["notes"] == "initial note"

    # flip favorite back off, notes still untouched
    resp = client.patch(
        f"/api/projects/{project_id}/takes/{take_id}",
        json={"favorite": False},
    )
    assert resp.status_code == 200
    assert resp.json()["favorite"] is False
    assert resp.json()["notes"] == "initial note"


def test_patch_unknown_take_404s(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "No Takes"}).json()
    resp = client.patch(
        f"/api/projects/{project['id']}/takes/does-not-exist",
        json={"favorite": True},
    )
    assert resp.status_code == 404


def test_concurrent_favorite_and_notes_patches_do_not_lose_updates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer-flagged: update_take_annotations used to do an unlocked
    read-modify-write, so a favorite PATCH and a notes PATCH racing each
    other could both read the same on-disk meta and then overwrite one
    another's write. Widen the race window artificially (sleep between the
    read and the write) so this fails reliably without the fix (storage's
    `_project_lock`) and passes reliably with it."""
    project_id, take_id = _make_take(client)

    # Patch the *defining* module: update_take_annotations calls get_take
    # through server.storage.takes' own globals, so patching the package
    # re-export would leave the real function running and never widen the
    # race this test exists to force.
    original_get_take = storage.takes.get_take

    def slow_get_take(pid: str, tid: str) -> dict:
        meta = original_get_take(pid, tid)
        time.sleep(0.05)
        return meta

    monkeypatch.setattr(storage.takes, "get_take", slow_get_take)

    responses: list = []

    def patch_favorite() -> None:
        responses.append(
            client.patch(f"/api/projects/{project_id}/takes/{take_id}", json={"favorite": True})
        )

    def patch_notes() -> None:
        responses.append(
            client.patch(
                f"/api/projects/{project_id}/takes/{take_id}",
                json={"notes": "concurrent note"},
            )
        )

    t1 = threading.Thread(target=patch_favorite)
    t2 = threading.Thread(target=patch_notes)
    t1.start()
    time.sleep(0.01)  # t1 must be inside its read before t2 starts, to force the race
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(responses) == 2
    assert all(r.status_code == 200 for r in responses)

    detail = client.get(f"/api/projects/{project_id}").json()
    take = next(t for t in detail["takes"] if t["id"] == take_id)
    assert take["favorite"] is True
    assert take["notes"] == "concurrent note"


def test_legacy_take_missing_favorite_and_notes_gets_defaults(
    client: TestClient, tmp_path: Path
) -> None:
    """A take written before this migration has neither key on disk (SPEC.md
    sec 12 Phase 6). Both list_takes and get_take must backfill defaults
    rather than the frontend receiving `undefined`/missing fields."""
    project_id, take_id = _make_take(client)

    meta_path = tmp_path / "projects" / project_id / "takes" / take_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["favorite"]
    del meta["notes"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    listed = client.get(f"/api/projects/{project_id}/takes").json()
    legacy = next(t for t in listed if t["id"] == take_id)
    assert legacy["favorite"] is False
    assert legacy["notes"] == ""

    detail = client.get(f"/api/projects/{project_id}").json()
    legacy_detail = next(t for t in detail["takes"] if t["id"] == take_id)
    assert legacy_detail["favorite"] is False
    assert legacy_detail["notes"] == ""

    # Patching just one field on a legacy take must not surface the other as
    # missing -- it should come back as its normalized default.
    resp = client.patch(f"/api/projects/{project_id}/takes/{take_id}", json={"favorite": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["favorite"] is True
    assert body["notes"] == ""
