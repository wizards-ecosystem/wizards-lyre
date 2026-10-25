"""SPEC.md sec 16.6: the export ZIP's manifest agrees with the archive and
with the take it delivers, and says what is missing instead of omitting it.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient

from server import storage
from tests.helpers import wait_for_job


def _project_with_take(client: TestClient, title: str = "Export") -> tuple[str, str]:
    project_id = client.post("/api/projects", json={"title": title}).json()["id"]
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    done = wait_for_job(client, job["id"])
    assert done["status"] == "done", done.get("error")
    return project_id, done["take_id"]


def _archive(client: TestClient, url: str) -> zipfile.ZipFile:
    resp = client.get(url)
    assert resp.status_code == 200, resp.text
    return zipfile.ZipFile(io.BytesIO(resp.content))


def test_every_member_is_hashed_and_matches_the_archive(client: TestClient) -> None:
    project_id, take_id = _project_with_take(client)
    zf = _archive(client, f"/v1/projects/{project_id}/export")
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["schema"] == "lyre.export-manifest/1"
    assert manifest["project"]["id"] == project_id
    assert manifest["take"]["id"] == take_id
    assert manifest["take"]["provenance_complete"] is True

    listed = {entry["path"]: entry for entry in manifest["files"]}
    assert set(listed) | {"manifest.json"} == set(zf.namelist())
    for path, entry in listed.items():
        data = zf.read(path)
        assert entry["bytes"] == len(data), path
        assert entry["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(), path
    mix = listed["mix.wav"]
    assert mix["role"] == "mix"
    assert (mix["sample_rate"], mix["channels"], mix["duration_sec"]) == (8000, 1, 0.5)
    # The manifest's take hash is the take's own recorded hash.
    take = storage.get_take(project_id, take_id)
    assert mix["sha256"] == take["provenance"]["files"]["mix.wav"]["sha256"]
    assert manifest["take"]["actual"]["seed"] == take["seed"]
    assert "can have changed since the take was rendered" in manifest["notes"]["plan"]
    assert "not a backup" in manifest["notes"]["scope"]


def test_missing_optional_files_are_named_not_hidden(client: TestClient) -> None:
    project_id, _take = _project_with_take(client)
    manifest = client.get(f"/v1/projects/{project_id}/export/manifest").json()
    roles = {(entry["role"], entry["reason"]) for entry in manifest["unavailable"]}
    assert ("lrc", "ACE-Step produced no lyric timestamps for this take") in roles
    assert ("stem", "no extract or lego takes") in roles

    without = client.get(f"/v1/projects/{project_id}/export/manifest?include_stems=false").json()
    assert {"role": "stem", "reason": "not requested"} in without["unavailable"]
    assert without["export_url"].endswith("include_stems=false")


def test_an_explicit_take_is_exported_instead_of_the_active_one(client: TestClient) -> None:
    project_id, first = _project_with_take(client)
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    second = wait_for_job(client, job["id"])["take_id"]
    assert storage.load_project(project_id)["active_take_id"] == first

    zf = _archive(client, f"/api/projects/{project_id}/export?take_id={second}")
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["take"]["id"] == second
    expected = storage.take_audio_path(project_id, second).read_bytes()
    assert zf.read("mix.wav") == expected


def test_a_kept_take_without_audio_is_an_error_not_a_silent_gap(client: TestClient) -> None:
    project_id, take_id = _project_with_take(client)
    storage.take_audio_path(project_id, take_id).unlink()
    (storage.take_dir(project_id, take_id) / "mix.mp3").unlink(missing_ok=True)
    resp = client.get(f"/v1/projects/{project_id}/export")
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "take_id"

    unknown = client.get(f"/v1/projects/{project_id}/export?take_id=nope")
    assert unknown.status_code == 400
    api_unknown = client.get(f"/api/projects/{project_id}/export?take_id=nope")
    assert api_unknown.status_code == 400
    assert "take not found" in api_unknown.json()["detail"]


def test_a_failed_take_cannot_be_exported(client: TestClient, monkeypatch) -> None:
    import worker.mock_worker as mock_worker_module

    project_id, _good = _project_with_take(client)

    def boom(**_kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(mock_worker_module, "run_job", boom)
    job = client.post(f"/api/projects/{project_id}/jobs", json={"action": "generate"}).json()
    failed = wait_for_job(client, job["id"])
    resp = client.get(f"/v1/projects/{project_id}/export?take_id={failed['take_id']}")
    assert resp.status_code == 400
    assert "failed" in resp.json()["error"]["message"]


def test_a_legacy_take_is_exported_with_unknown_provenance(client: TestClient) -> None:
    project_id, take_id = _project_with_take(client)
    meta_path = storage.take_dir(project_id, take_id) / "meta.json"
    legacy = json.loads(meta_path.read_text(encoding="utf-8"))
    legacy.pop("provenance")
    legacy.pop("job_id")
    meta_path.write_text(json.dumps(legacy), encoding="utf-8")

    manifest = json.loads(
        _archive(client, f"/v1/projects/{project_id}/export").read("manifest.json")
    )
    take = manifest["take"]
    assert take["id"] == take_id
    assert take["provenance_complete"] is False
    assert take["brief"] is None and take["actual"] is None and take["requested"] is None
    # The audio facts are still read from the file itself.
    mix = next(entry for entry in manifest["files"] if entry["role"] == "mix")
    assert mix["sample_rate"] == 8000


def test_stems_are_listed_with_their_take(client: TestClient) -> None:
    project_id, source = _project_with_take(client)
    job = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "extract", "source_take_id": source, "track_name": "../../drums"},
    ).json()
    stem_take = wait_for_job(client, job["id"])["take_id"]
    zf = _archive(client, f"/v1/projects/{project_id}/export")
    manifest = json.loads(zf.read("manifest.json"))
    stems = [entry for entry in manifest["files"] if entry["role"] == "stem"]
    assert [stem["take_id"] for stem in stems] == [stem_take]
    assert stems[0]["path"] == f"{stem_take}-extract-drums.wav"
    assert stems[0]["path"] in zf.namelist()
