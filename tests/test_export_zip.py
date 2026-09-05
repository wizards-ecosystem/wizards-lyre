"""SPEC.md sec 12 Phase 5 / sec 9.2: GET /api/projects/{id}/export builds an
in-memory zip of project.json, plan.json, the active take's mix, and
(optionally) every extract/lego take's audio -- see
storage.build_export_zip's docstring for why "stems" means "these
task_types" rather than a stems/ directory that nothing in the current
storage layout actually writes. `complete` is deliberately excluded: SPEC's
data-model diagram (sec 12 Phase 5 / sec 7) labels stems/ "extract / lego
outputs" specifically -- `complete` produces a full alternate mix, not an
isolated or added track.
"""

from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient
from helpers import wait_for_job

from server import storage


def test_export_with_no_takes_still_200s_with_minimal_zip(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Empty Export"}).json()
    project_id = project["id"]

    resp = client.get(f"/api/projects/{project_id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    assert "Empty Export-export.zip" in resp.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert names == {"project.json", "plan.json"}

    project_json = json.loads(zf.read("project.json"))
    assert project_json["id"] == project_id
    plan_json = json.loads(zf.read("plan.json"))
    assert plan_json == storage.default_plan()


def test_export_after_generate_includes_active_mix(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Generate Export"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    take_id = gen["take_id"]

    resp = client.get(f"/api/projects/{project_id}/export")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "project.json" in names
    assert "plan.json" in names
    assert "mix.wav" in names or "mix.mp3" in names

    project_json = json.loads(zf.read("project.json"))
    assert project_json["active_take_id"] == take_id


def test_export_includes_or_excludes_extract_stem_by_flag(client: TestClient) -> None:
    project = client.post("/api/projects", json={"title": "Stem Export"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    extract_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals",
        },
    )
    extract_job = wait_for_job(client, extract_resp.json()["id"])
    assert extract_job["status"] == "done", extract_job.get("error")
    stem_take_id = extract_job["take_id"]

    expected_stem_name = f"{stem_take_id}-extract-vocals.wav"

    resp_with_stems = client.get(f"/api/projects/{project_id}/export?include_stems=true")
    assert resp_with_stems.status_code == 200
    names_with_stems = set(zipfile.ZipFile(io.BytesIO(resp_with_stems.content)).namelist())
    assert expected_stem_name in names_with_stems

    resp_without_stems = client.get(f"/api/projects/{project_id}/export?include_stems=false")
    assert resp_without_stems.status_code == 200
    names_without_stems = set(zipfile.ZipFile(io.BytesIO(resp_without_stems.content)).namelist())
    assert expected_stem_name not in names_without_stems
    assert not any(name.startswith(f"{stem_take_id}-extract-") for name in names_without_stems)

    # default (no query param) includes stems
    resp_default = client.get(f"/api/projects/{project_id}/export")
    names_default = set(zipfile.ZipFile(io.BytesIO(resp_default.content)).namelist())
    assert expected_stem_name in names_default


def test_export_excludes_complete_takes_from_stems(client: TestClient) -> None:
    """`complete` ("Fill arrangement") produces a full alternate mix, not an
    isolated/added track -- SPEC's stems/ description covers extract/lego
    only (reviewer-flagged), so a complete take must never show up under a
    stem-style arcname even with include_stems=true."""
    project = client.post("/api/projects", json={"title": "Complete Not A Stem"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    complete_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "complete",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals, drums, bass",
        },
    )
    complete_job = wait_for_job(client, complete_resp.json()["id"])
    assert complete_job["status"] == "done", complete_job.get("error")
    complete_take_id = complete_job["take_id"]

    resp = client.get(f"/api/projects/{project_id}/export?include_stems=true")
    assert resp.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert not any(name.startswith(f"{complete_take_id}-complete-") for name in names)
    # the complete take is still the active take, so it's present once, under
    # the generic mix name
    assert "mix.wav" in names or "mix.mp3" in names


def test_export_always_includes_active_mix_even_when_it_is_also_a_stem(client: TestClient) -> None:
    """extract/lego promote their output to the active take (like every other
    job), so the common export-right-after-extract case still must write
    "mix<ext>" -- consumers need one predictable, unconditional place to find
    the active take, regardless of whether it also qualifies as a stem
    (reviewer-flagged: omitting "mix<ext>" here breaks that contract, even
    though the stem entry happens to hold identical bytes)."""
    project = client.post("/api/projects", json={"title": "Active Mix Also A Stem"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    extract_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": "vocals",
        },
    )
    extract_job = wait_for_job(client, extract_resp.json()["id"])
    assert extract_job["status"] == "done", extract_job.get("error")
    stem_take_id = extract_job["take_id"]

    project_after = client.get(f"/api/projects/{project_id}").json()
    assert project_after["project"]["active_take_id"] == stem_take_id

    resp = client.get(f"/api/projects/{project_id}/export?include_stems=true")
    assert resp.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()

    stem_name = f"{stem_take_id}-extract-vocals.wav"
    assert names.count(stem_name) == 1
    # the active mix entry must still be present, even though it's the same
    # audio as the stem entry above
    assert "mix.wav" in names or "mix.mp3" in names


def test_export_sanitizes_path_traversing_track_name(client: TestClient) -> None:
    """`track_name` is free text forwarded to ACE-Step's `instruction` field
    (SPEC.md sec 4.4) -- `_resolve_track_name` only trims whitespace, it
    doesn't restrict characters, so a request posted straight to the HTTP API
    (bypassing the web UI) can set track_name to something like
    `../../../evil`. The export archive must never turn that into a
    path-traversing zip member (zip-slip), regardless of what's stored on
    the take (reviewer-flagged)."""
    project = client.post("/api/projects", json={"title": "Zip Slip Guard"}).json()
    project_id = project["id"]
    client.put(
        f"/api/projects/{project_id}/plan",
        json={**storage.default_plan(), "caption": "test tone, sine wave"},
    )

    gen_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"action": "generate", "dit_profile": "iterate", "seed": -1},
    )
    gen = wait_for_job(client, gen_resp.json()["id"])
    assert gen["status"] == "done", gen.get("error")
    source_take_id = gen["take_id"]

    malicious_track_name = "../../../../evil"
    extract_resp = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "action": "extract",
            "dit_profile": "studio_ops",
            "source_take_id": source_take_id,
            "track_name": malicious_track_name,
        },
    )
    extract_job = wait_for_job(client, extract_resp.json()["id"])
    assert extract_job["status"] == "done", extract_job.get("error")
    stem_take_id = extract_job["take_id"]

    # confirm the raw malicious value really was persisted verbatim on the
    # take -- the guard has to live in the export path, not upstream of it
    take = next(
        t
        for t in client.get(f"/api/projects/{project_id}").json()["takes"]
        if t["id"] == stem_take_id
    )
    assert take["track_name"] == malicious_track_name

    resp = client.get(f"/api/projects/{project_id}/export?include_stems=true")
    assert resp.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()

    for name in names:
        assert not name.startswith("/"), name
        assert ".." not in name.split("/"), name
        assert "\\" not in name, name

    # the sanitized member is still present under a safe name derived from
    # the same take -- the malicious characters are stripped, not dropped
    # entirely
    stem_members = [n for n in names if n.startswith(f"{stem_take_id}-extract-")]
    assert len(stem_members) == 1
    assert stem_members[0] == f"{stem_take_id}-extract-evil.wav"
