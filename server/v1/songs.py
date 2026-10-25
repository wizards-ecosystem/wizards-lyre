"""One-shot song creation (SPEC.md sec 15.4).

`POST /v1/songs` calls `create_song`. The MCP `create_song` tool calls that
route over HTTP rather than this function, so the queue stays in one place.
It writes an ordinary project and queues `generate`. It does not render,
and it does not call the worker.
"""

from __future__ import annotations

from typing import Any

from server import jobs, storage
from server.generation import brief as creation
from server.v1.models import SongCreate


def song_plan(body: SongCreate) -> dict[str, Any]:
    """The plan.json a song request stores before the worker runs.

    Simple mode is a prompt with no caption and no lyrics: `from_idea`, so
    the worker's planner fills the plan (including `[Instrumental]` when that
    flag is set). Custom mode stores the caller's text as `from_brief`, and
    an instrumental with no lyrics becomes the `[Instrumental]` placeholder.
    `caption_rewrite` defaults to false only once the caller has supplied
    caption or lyrics. Every field the caller sent is origin `user`, so the
    planner cannot replace it (SPEC.md sec 16.3); omitted fields are
    `default`.
    """
    caption = body.caption or ""
    lyrics = body.lyrics or ""
    custom = bool(caption) or bool(lyrics)
    if body.instrumental and custom and not lyrics:
        lyrics = "[Instrumental]"
    # Simple mode keeps the studio default. Custom mode defaults to false.
    caption_rewrite = not custom if body.caption_rewrite is None else body.caption_rewrite
    sent = body.model_fields_set
    origins = creation.default_origins()
    if body.caption:
        origins["caption"] = "user"
    if body.lyrics:
        origins["lyrics"] = "user"
    for field in ("bpm", "keyscale", "timesignature", "duration_sec"):
        if getattr(body, field) is not None:
            origins[field] = "user"
    if "vocal_language" in sent:
        origins["vocal_language"] = "user"
    return {
        "query": body.prompt,
        "caption": caption,
        "negative": [],
        "lyrics": lyrics,
        "instrumental": body.instrumental,
        "vocal_language": body.vocal_language,
        "bpm": body.bpm,
        "keyscale": body.keyscale,
        "timesignature": body.timesignature or "4/4",
        "duration_sec": body.duration_sec,
        "sections": [],
        "caption_rewrite": caption_rewrite,
        "creation_mode": "from_brief" if custom else "from_idea",
        "field_origins": origins,
    }


def create_song(body: SongCreate) -> dict[str, Any]:
    """Create the project, write the plan, and queue `generate`.

    Returns the queued job row. A failure after the project exists removes
    that project, so a rejected song does not stay in the library.
    """
    project = storage.create_project(
        title=body.title,
        query=body.prompt,
        dit_profile=body.dit_profile,
    )
    project_id = str(project["id"])
    try:
        storage.replace_plan(project_id, song_plan(body))
        return jobs.enqueue_job(
            project_id,
            {
                "action": "generate",
                "dit_profile": body.dit_profile,
                "seed": body.seed,
            },
        )
    except Exception:
        storage.delete_project(project_id)
        raise
