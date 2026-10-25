"""Curated creative presets (SPEC.md sec 16.6).

Six independently written starting points: brief defaults, not a model
catalogue or a trained style pack. Applying one is deterministic and only
touches fields the user has not set themselves; it never changes lyrics,
models, media, or compute. The applied id and version go into the plan
(`preset`), the job's creation input, and the take's provenance.

Pure Python, like the rest of this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from server.generation import brief as creation

PRESET_SCHEMA = "lyre.preset/1"

# Fields a preset may set, beside the caption and the Song/Instrumental
# recommendation. Lyrics are never among them.
PRESET_FIELDS = ("bpm", "keyscale", "timesignature", "duration_sec")

PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "lyric_song",
        "version": 1,
        "name": "Lyric song",
        "purpose": "A song built around a sung lead and a clear verse and chorus.",
        "caption": (
            "song with a clear lead vocal, verse and chorus structure, memorable melodic "
            "hook, full band arrangement"
        ),
        "instrumental": False,
        "fields": {"timesignature": "4/4"},
        "example": "a hopeful song about leaving home at dawn",
    },
    {
        "id": "intimate_acoustic_song",
        "version": 1,
        "name": "Intimate acoustic song",
        "purpose": "A quiet, close song for voice and acoustic guitar.",
        "caption": (
            "intimate acoustic song, fingerpicked acoustic guitar, soft close vocal, warm room "
            "sound, gentle dynamics, sparse arrangement"
        ),
        "instrumental": False,
        "fields": {"bpm": 84, "timesignature": "4/4"},
        "example": "a letter to an old friend, sung late at night",
    },
    {
        "id": "energetic_electronic_song",
        "version": 1,
        "name": "Energetic electronic song",
        "purpose": "Upbeat electronic pop with a driving beat and a lifted chorus.",
        "caption": (
            "energetic electronic pop, four-on-the-floor kick, bright synth leads, punchy bass, "
            "confident vocal, big chorus lift"
        ),
        "instrumental": False,
        "fields": {"bpm": 124, "timesignature": "4/4"},
        "example": "running through a city at night with the lights coming on",
    },
    {
        "id": "exploration_ambience",
        "version": 1,
        "name": "Exploration ambience",
        "purpose": "Calm, evolving background music for wandering a game world.",
        "caption": (
            "ambient exploration music, slowly evolving pads, sparse plucked textures, distant "
            "soft percussion, slow harmonic movement, calm and curious"
        ),
        "instrumental": True,
        "fields": {"bpm": 70, "duration_sec": 90},
        "example": "a misty forest at the edge of the map",
    },
    {
        "id": "combat_cue",
        "version": 1,
        "name": "Combat cue",
        "purpose": "Driving, tense music for a fight or a chase.",
        "caption": (
            "intense combat music, driving drums and taiko hits, urgent string ostinato, low "
            "brass stabs, rising tension, relentless energy"
        ),
        "instrumental": True,
        "fields": {"bpm": 150, "duration_sec": 90, "timesignature": "4/4"},
        "example": "a boss fight on a collapsing bridge",
    },
    {
        "id": "menu_theme",
        "version": 1,
        "name": "Menu theme",
        "purpose": "A welcoming title or menu theme with a recognisable motif.",
        "caption": (
            "main menu theme, memorable melodic motif, warm blend of orchestra and synths, "
            "inviting and steady, clear ending"
        ),
        "instrumental": True,
        "fields": {"bpm": 96, "duration_sec": 60},
        "example": "the title screen of a cosy farming game",
    },
)

_BY_ID = {preset["id"]: preset for preset in PRESETS}


class PresetNotFound(LookupError):
    pass


def list_presets() -> list[dict[str, Any]]:
    """The registry as served (`GET .../presets`, capabilities)."""
    return [{"schema": PRESET_SCHEMA, **preset} for preset in PRESETS]


def get_preset(preset_id: str) -> dict[str, Any]:
    try:
        return _BY_ID[preset_id]
    except KeyError:
        raise PresetNotFound(preset_id) from None


def apply_preset(plan: Mapping[str, Any], preset_id: str) -> dict[str, Any]:
    """Propose the plan after applying `preset_id`, without writing it.

    Returns `{"plan", "changes", "warnings", "preset"}`. A field whose origin
    is `user` keeps its value; a changed field gets origin `preset`. Lyrics
    are never touched, and an Instrumental recommendation is not applied
    over lyrics someone wrote (that would be the conflict section 16.3
    refuses). The same plan and preset always give the same result."""
    preset = get_preset(preset_id)
    origins = creation.plan_origins(plan)
    proposed = dict(plan)
    proposed_origins = dict(origins)
    changes: list[dict[str, Any]] = []
    warnings: list[str] = []

    def change(field: str, value: Any, *, tracked: bool = True) -> None:
        if proposed.get(field) == value:
            return
        changes.append({"field": field, "from": proposed.get(field), "to": value})
        proposed[field] = value
        if tracked:
            proposed_origins[field] = "preset"

    if origins["caption"] == "user" and str(plan.get("caption") or "").strip():
        warnings.append("kept your caption; the preset's caption was not applied")
    else:
        change("caption", preset["caption"])

    for field, value in preset["fields"].items():
        if origins[field] == "user" and not creation.is_unset(field, plan.get(field)):
            warnings.append(f"kept your {field}")
        else:
            change(field, value)

    lyrics = str(plan.get("lyrics") or "")
    if (
        preset["instrumental"]
        and lyrics.strip()
        and not creation.is_instrumental_placeholder(lyrics)
    ):
        if not plan.get("instrumental"):
            warnings.append(
                "the preset suggests an instrumental, but the plan has lyrics, so it stays a song"
            )
    else:
        change("instrumental", preset["instrumental"], tracked=False)

    identity = {"id": preset["id"], "version": preset["version"]}
    proposed["preset"] = identity
    proposed["field_origins"] = proposed_origins
    return {"plan": proposed, "changes": changes, "warnings": warnings, "preset": identity}
