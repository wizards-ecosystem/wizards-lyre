"""SPEC.md sec 15 phase 9: `POST /v1/songs`.

No GPU. The worker is the mock. The route creates a project, writes the
plan, and queues `generate`. It does not render inline. `/api` is unchanged.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from server import jobs as jobs_module
from server.api_keys import create_key
from server.app import app
from server.v1 import songs as songs_module
from server.v1.models import SongCreate
from tests.helpers import wait_for_job


def test_prompt_only_song_persists_a_filled_plan_and_a_take(client: TestClient) -> None:
    response = client.post(
        "/v1/songs",
        json={"prompt": "a rainy night jazz ballad", "seed": 17, "title": "  Rain  "},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert response.headers["location"] == f"/v1/jobs/{job['id']}"
    assert job["action"] == "generate"
    assert job["status"] == "queued"
    assert job["stage"] is None
    assert job["dit_profile"] == "iterate"
    assert job["artifacts"] == []
    assert job["project_id"]

    done = wait_for_job(client, job["id"])
    assert done["status"] == "done"
    viewed = client.get(f"/v1/jobs/{job['id']}").json()
    assert viewed["status"] == "done"
    assert viewed["take_id"]
    assert viewed["error"] is None
    assert viewed["seed"] == 17
    assert any(url.endswith("/mix.wav") for url in viewed["artifacts"])
    audio = client.get(viewed["artifacts"][0])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/")

    detail = client.get(f"/v1/projects/{job['project_id']}").json()
    assert detail["project"]["title"] == "Rain"
    assert detail["project"]["active_take_id"] == viewed["take_id"]
    assert detail["plan"]["query"] == "a rainy night jazz ballad"
    assert detail["plan"]["caption"] == "auto: a rainy night jazz ballad"
    assert "a rainy night jazz ballad" in detail["plan"]["lyrics"]
    assert detail["plan"]["caption_rewrite"] is True
    assert detail["takes"][0]["id"] == viewed["take_id"]

    listed = client.get("/v1/projects").json()["data"]
    assert any(row["id"] == job["project_id"] for row in listed)
    via_api = client.get(f"/api/projects/{job['project_id']}")
    assert via_api.status_code == 200
    assert via_api.json()["plan"]["caption"] == detail["plan"]["caption"]
    assert "detail" not in response.json()


def test_prompt_only_instrumental_is_filled_by_the_worker(client: TestClient) -> None:
    response = client.post(
        "/v1/songs",
        json={"prompt": "soft piano", "instrumental": True, "duration_sec": 10, "seed": 21},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    done = wait_for_job(client, job["id"])
    assert done["status"] == "done"
    plan = client.get(f"/v1/projects/{job['project_id']}").json()["plan"]
    assert plan["caption"] == "auto: soft piano"
    assert plan["lyrics"] == "[Instrumental]"
    assert plan["instrumental"] is True
    # The mock's plan patch fills bpm and key and leaves a provided duration.
    assert plan["duration_sec"] == 10
    assert plan["bpm"] == 120
    assert plan["keyscale"] == "C Major"


def test_caller_caption_is_kept_unless_rewrite_is_requested(client: TestClient) -> None:
    """Custom mode defaults `caption_rewrite` to false, and the caller's
    caption is still the plan after the mock worker runs.

    `caption_rewrite: true` is stored so the real worker turns thinking on
    (see `test_custom_mode_caption_rewrite_enabled_sets_thinking_true`). The
    mock does not invent a replacement caption, and custom mode does not
    write an LM rewrite back onto plan.json. The locked case is the default:
    a caller-supplied caption is not rewritten.
    """
    kept = client.post(
        "/v1/songs",
        json={
            "prompt": "night bus",
            "caption": "muted trumpet, rain",
            "lyrics": "[Verse]\nstay",
            "seed": 3,
        },
    )
    assert kept.status_code == 202, kept.text
    job = kept.json()
    plan = client.get(f"/v1/projects/{job['project_id']}").json()["plan"]
    assert plan["caption"] == "muted trumpet, rain"
    assert plan["lyrics"] == "[Verse]\nstay"
    assert plan["caption_rewrite"] is False
    assert plan["query"] == "night bus"
    done = wait_for_job(client, job["id"])
    assert done["status"] == "done"
    after = client.get(f"/v1/projects/{job['project_id']}").json()
    assert after["plan"]["caption"] == "muted trumpet, rain"
    assert after["plan"]["lyrics"] == "[Verse]\nstay"
    assert after["plan"]["caption_rewrite"] is False
    assert after["takes"][0]["caption"] == "muted trumpet, rain"

    rewritten = client.post(
        "/v1/songs",
        json={
            "prompt": "night bus",
            "caption": "muted trumpet, rain",
            "caption_rewrite": False,
            "seed": 5,
        },
    )
    assert rewritten.status_code == 202, rewritten.text
    explicit = rewritten.json()
    wait_for_job(client, explicit["id"])
    locked = client.get(f"/v1/projects/{explicit['project_id']}").json()["plan"]
    assert locked["caption_rewrite"] is False
    assert locked["caption"] == "muted trumpet, rain"
    # Caption with no lyrics is custom mode, so the lyrics stay empty. The
    # `[Instrumental]` placeholder is only for instrumental custom songs.
    assert locked["lyrics"] == ""

    opted_in = client.post(
        "/v1/songs",
        json={
            "prompt": "night bus",
            "caption": "muted trumpet, rain",
            "caption_rewrite": True,
            "seed": 4,
        },
    )
    assert opted_in.status_code == 202, opted_in.text
    body = opted_in.json()
    before = client.get(f"/v1/projects/{body['project_id']}").json()["plan"]
    assert before["caption_rewrite"] is True
    assert before["caption"] == "muted trumpet, rain"
    wait_for_job(client, body["id"])
    after_opt_in = client.get(f"/v1/projects/{body['project_id']}").json()["plan"]
    assert after_opt_in["caption_rewrite"] is True
    assert after_opt_in["caption"] == "muted trumpet, rain"


def test_song_plan_is_written_before_the_worker_runs(api_client: TestClient) -> None:
    simple = api_client.post(
        "/v1/songs",
        json={
            "prompt": "solo piano",
            "instrumental": True,
            "webhook_url": None,
            "title": None,
            "duration_sec": None,
        },
    )
    assert simple.status_code == 202, simple.text
    plan = api_client.get(f"/v1/projects/{simple.json()['project_id']}").json()["plan"]
    assert plan["query"] == "solo piano"
    assert plan["caption"] == ""
    assert plan["lyrics"] == ""
    assert plan["instrumental"] is True
    assert plan["duration_sec"] is None
    assert plan["bpm"] is None
    assert plan["keyscale"] is None
    assert plan["timesignature"] == "4/4"
    assert plan["caption_rewrite"] is True
    assert plan["vocal_language"] == "en"
    assert simple.json()["status"] == "queued"

    omitted = api_client.post("/v1/songs", json={"prompt": "  solo piano  "})
    empty = api_client.get(f"/v1/projects/{omitted.json()['project_id']}").json()
    assert empty["project"]["title"] == "Untitled"
    assert empty["plan"]["query"] == "solo piano"
    assert empty["plan"]["duration_sec"] is None

    custom = api_client.post(
        "/v1/songs",
        json={
            "prompt": "solo piano",
            "caption": "felt piano",
            "instrumental": True,
            "timesignature": "3/4",
            "vocal_language": " es ",
            "bpm": 72,
            "keyscale": " D Minor ",
            "duration_sec": 10,
            "dit_profile": "polish",
            "title": "Night",
            "seed": 9,
            "lyrics": "[Verse]\nkeep me",
        },
    )
    assert custom.status_code == 202, custom.text
    body = custom.json()
    assert body["dit_profile"] == "polish"
    assert body["seed"] == 9
    assert body["action"] == "generate"
    detail = api_client.get(f"/v1/projects/{body['project_id']}").json()
    assert detail["project"]["title"] == "Night"
    assert detail["project"]["dit_profile"] == "polish"
    # Caller-supplied lyrics win over the instrumental placeholder.
    assert detail["plan"]["lyrics"] == "[Verse]\nkeep me"
    assert detail["plan"]["caption"] == "felt piano"
    assert detail["plan"]["caption_rewrite"] is False
    assert detail["plan"]["timesignature"] == "3/4"
    assert detail["plan"]["vocal_language"] == "es"
    assert detail["plan"]["bpm"] == 72
    assert detail["plan"]["keyscale"] == "D Minor"
    assert detail["plan"]["duration_sec"] == 10
    assert detail["plan"]["instrumental"] is True

    placeholder = api_client.post(
        "/v1/songs",
        json={"prompt": "solo piano", "caption": "felt piano", "instrumental": True},
    )
    stored = api_client.get(f"/v1/projects/{placeholder.json()['project_id']}").json()["plan"]
    assert stored["lyrics"] == "[Instrumental]"
    assert stored["caption_rewrite"] is False

    blank_caption = api_client.post(
        "/v1/songs", json={"prompt": "rain", "caption": "   ", "lyrics": "  "}
    )
    treated_as_simple = api_client.get(f"/v1/projects/{blank_caption.json()['project_id']}").json()[
        "plan"
    ]
    assert treated_as_simple["caption"] == ""
    assert treated_as_simple["lyrics"] == ""
    assert treated_as_simple["caption_rewrite"] is True


def test_song_validation_uses_the_v1_error(api_client: TestClient) -> None:
    missing = api_client.post("/v1/songs", json={})
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "invalid_request"
    assert missing.json()["error"]["param"] == "prompt"
    assert "detail" not in missing.json()

    blank = api_client.post("/v1/songs", json={"prompt": "   "})
    assert blank.status_code == 400
    assert blank.json()["error"]["param"] == "prompt"
    not_text = api_client.post("/v1/songs", json={"prompt": 12})
    assert not_text.status_code == 400
    assert not_text.json()["error"]["param"] == "prompt"

    duration = api_client.post("/v1/songs", json={"prompt": "rain", "duration_sec": 0})
    assert duration.status_code == 400
    assert duration.json()["error"]["param"] == "duration_sec"
    negative = api_client.post("/v1/songs", json={"prompt": "rain", "duration_sec": -3})
    assert negative.status_code == 400
    assert negative.json()["error"]["param"] == "duration_sec"
    text = api_client.post("/v1/songs", json={"prompt": "rain", "duration_sec": "long"})
    assert text.status_code == 400
    assert text.json()["error"]["param"] == "duration_sec"

    profile = api_client.post("/v1/songs", json={"prompt": "rain", "dit_profile": "studio_ops"})
    assert profile.status_code == 400
    assert profile.json()["error"]["param"] == "dit_profile"

    webhook = api_client.post(
        "/v1/songs", json={"prompt": "rain", "webhook_url": "https://example.com/hook"}
    )
    assert webhook.status_code == 400
    assert webhook.json()["error"]["param"] == "webhook_url"
    assert "not available" in webhook.json()["error"]["message"]

    extra = api_client.post("/v1/songs", json={"prompt": "rain", "batch_size": 4})
    assert extra.status_code == 400
    assert extra.json()["error"]["param"] == "batch_size"

    bpm = api_client.post("/v1/songs", json={"prompt": "rain", "bpm": 90.5})
    assert bpm.status_code == 400
    assert bpm.json()["error"]["param"] == "bpm"

    flagged = api_client.post("/v1/songs", json={"prompt": "rain", "instrumental": "nope"})
    assert flagged.status_code == 400
    assert flagged.json()["error"]["param"] == "instrumental"

    assert api_client.get("/v1/projects").json()["data"] == []
    # `/api` does not gain this route, and it keeps its own error shape.
    # Without a built SPA the router answers 404 in `/api` shape. With
    # web/dist present the static mount owns unmatched paths and answers a
    # POST with 405. Either way no route handled it.
    unknown = api_client.post("/api/songs", json={"prompt": "rain"})
    assert unknown.status_code in (404, 405)
    if unknown.status_code == 404:
        assert "detail" in unknown.json()
    assert not any(getattr(route, "path", None) == "/api/songs" for route in app.routes)


def test_song_duration_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        SongCreate.model_validate({"prompt": "rain", "duration_sec": float("nan")})
    with pytest.raises(ValidationError):
        SongCreate.model_validate({"prompt": "rain", "duration_sec": float("inf")})
    with pytest.raises(ValidationError):
        SongCreate.model_validate({"prompt": "rain", "duration_sec": float("-inf")})


def test_failed_enqueue_removes_the_project(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(project_id: str, body: dict) -> dict:
        del project_id, body
        raise jobs_module.JobError("invalid dit_profile: nope")

    monkeypatch.setattr(songs_module.jobs, "enqueue_job", _boom)
    response = api_client.post("/v1/songs", json={"prompt": "rain", "dit_profile": "quality"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "invalid dit_profile" in response.json()["error"]["message"]
    assert "detail" not in response.json()
    assert api_client.get("/v1/projects").json()["data"] == []
    assert api_client.get("/v1/jobs").json()["data"] == []


def test_idempotent_song_replay(api_client: TestClient) -> None:
    raw = b'{"prompt":"rain","seed":4}'
    headers = {"Idempotency-Key": "song-1", "Content-Type": "application/json"}
    first = api_client.post("/v1/songs", content=raw, headers=headers)
    assert first.status_code == 202, first.text
    replay = api_client.post("/v1/songs", content=raw, headers=headers)
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.headers["location"] == first.headers["location"]
    assert len(api_client.get("/v1/projects").json()["data"]) == 1
    assert len(api_client.get("/v1/jobs").json()["data"]) == 1

    conflict = api_client.post(
        "/v1/songs",
        content=b'{"prompt":"sun","seed":4}',
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert conflict.json()["error"]["param"] == "Idempotency-Key"
    assert "detail" not in conflict.json()
    assert len(api_client.get("/v1/projects").json()["data"]) == 1

    # A key is optional. Two posts create two projects.
    one = api_client.post("/v1/songs", json={"prompt": "a"}).json()["project_id"]
    two = api_client.post("/v1/songs", json={"prompt": "a"}).json()["project_id"]
    assert one != two


def test_song_auth_on_and_off(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = api_client.post("/v1/songs", json={"prompt": "open"})
    assert opened.status_code == 202

    monkeypatch.setenv("LYRE_REQUIRE_AUTH", "1")
    missing = api_client.post("/v1/songs", json={"prompt": "nope"})
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"].lower().startswith("bearer")
    assert missing.json()["error"]["code"] == "unauthorized"
    assert "detail" not in missing.json()

    # An unauthenticated attempt does not consume the idempotency key.
    denied = api_client.post(
        "/v1/songs",
        json={"prompt": "later"},
        headers={"Idempotency-Key": "song-shared"},
    )
    assert denied.status_code == 401

    _reader, read_secret = create_key("reader", ["read"])
    _generator, generate_secret = create_key("generator", ["generate"])
    _writer, write_secret = create_key("writer", ["write"])
    read_headers = {"Authorization": f"Bearer {read_secret}"}
    generate_headers = {"Authorization": f"Bearer {generate_secret}"}
    write_headers = {"Authorization": f"Bearer {write_secret}"}

    created = api_client.post(
        "/v1/songs",
        json={"prompt": "later", "title": "Keyed"},
        headers={"Idempotency-Key": "song-shared", **generate_headers},
    )
    assert created.status_code == 202, created.text
    project_id = created.json()["project_id"]
    assert (
        api_client.get(f"/v1/projects/{project_id}", headers=read_headers).json()["project"][
            "title"
        ]
        == "Keyed"
    )
    assert (
        api_client.get(f"/v1/jobs/{created.json()['id']}", headers=generate_headers).status_code
        == 200
    )
    assert (
        api_client.post("/v1/songs", json={"prompt": "r"}, headers=read_headers).status_code == 403
    )
    assert (
        api_client.post("/v1/songs", json={"prompt": "w"}, headers=write_headers).status_code == 403
    )
    assert (
        api_client.get(f"/v1/jobs/{created.json()['id']}", headers=write_headers).status_code == 403
    )
    unknown = api_client.post(
        "/v1/songs",
        json={"prompt": "nope"},
        headers={"Authorization": "Bearer lyre_missing"},
    )
    assert unknown.status_code == 401

    # Distinct API keys do not share an idempotency slot.
    raw = b'{"prompt":"per-key"}'
    first = api_client.post(
        "/v1/songs",
        content=raw,
        headers={
            "Idempotency-Key": "song-per-key",
            "Content-Type": "application/json",
            **generate_headers,
        },
    )
    _other, other_secret = create_key("generator-2", ["generate"])
    second = api_client.post(
        "/v1/songs",
        content=raw,
        headers={
            "Idempotency-Key": "song-per-key",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {other_secret}",
        },
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["project_id"] != second.json()["project_id"]
    # Auth on still hides the studio surface.
    assert api_client.post("/api/projects", json={"title": "Studio"}).status_code == 404


def test_song_artifact_list_includes_mix_mp3_only_when_the_file_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 7: mix.mp3 is the playback copy. The job lists it only
    when the encode actually wrote the file, so GET cannot 404 an advertised
    artifact."""
    from pathlib import Path

    import worker.mock_worker as mock_worker_module

    def _write(wav_path: Path) -> bool:
        wav_path.with_name("mix.mp3").write_bytes(b"ID3-song")
        return True

    monkeypatch.setattr(mock_worker_module, "write_mix_mp3", _write)
    created = client.post("/v1/songs", json={"prompt": "rain on tin", "seed": 4})
    assert created.status_code == 202, created.text
    job = wait_for_job(client, created.json()["id"])
    assert job["status"] == "done"
    viewed = client.get(f"/v1/jobs/{job['id']}").json()
    assert any(url.endswith("/mix.wav") for url in viewed["artifacts"])
    assert any(url.endswith("/mix.mp3") for url in viewed["artifacts"])
    project_id = job["project_id"]
    take_id = viewed["take_id"]
    mp3 = client.get(f"/v1/artifacts/{project_id}/{take_id}/mix.mp3")
    assert mp3.status_code == 200
    assert mp3.headers["content-type"].startswith("audio/")
    assert mp3.content == b"ID3-song"

    def _skip(_wav_path: Path) -> bool:
        return False

    monkeypatch.setattr(mock_worker_module, "write_mix_mp3", _skip)
    skipped = client.post("/v1/songs", json={"prompt": "dry", "seed": 5})
    assert skipped.status_code == 202, skipped.text
    skipped_job = wait_for_job(client, skipped.json()["id"])
    assert skipped_job["status"] == "done"
    skipped_view = client.get(f"/v1/jobs/{skipped_job['id']}").json()
    assert any(url.endswith("/mix.wav") for url in skipped_view["artifacts"])
    assert not any(url.endswith("/mix.mp3") for url in skipped_view["artifacts"])
    missing = client.get(
        f"/v1/artifacts/{skipped_job['project_id']}/{skipped_view['take_id']}/mix.mp3"
    )
    assert missing.status_code == 404
    wav = client.get(f"/v1/artifacts/{skipped_job['project_id']}/{skipped_view['take_id']}/mix.wav")
    assert wav.status_code == 200


def test_song_openapi_schema() -> None:
    spec = app.openapi()
    operation = spec["paths"]["/v1/songs"]["post"]
    assert "202" in operation["responses"]
    assert "200" in operation["responses"]
    assert "Location" in operation["responses"]["202"]["headers"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert schema
    song = spec["components"]["schemas"]["SongCreate"]
    for name in (
        "prompt",
        "title",
        "caption",
        "lyrics",
        "instrumental",
        "vocal_language",
        "bpm",
        "keyscale",
        "timesignature",
        "duration_sec",
        "dit_profile",
        "seed",
        "caption_rewrite",
        "webhook_url",
    ):
        assert name in song["properties"]
    names = [param["name"].lower() for param in operation.get("parameters", [])]
    assert "idempotency-key" in names
    for code in ("400", "401", "403", "409"):
        body = operation["responses"][code]["content"]["application/json"]["schema"]
        assert body
    job = operation["responses"]["202"]["content"]["application/json"]["schema"]
    assert job
