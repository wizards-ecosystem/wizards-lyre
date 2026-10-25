"""The creation contract (SPEC.md sec 16.3): one resolver for every surface.

The browser (`/api`), `/v1`, MCP, the mock worker, the real ACE-Step adapter,
and the Remote GPU host all decide "what will this generate render?" here, so
precedence cannot drift between them. Three records flow through it:

- the *creation input* (`lyre.creation/1`): a frozen snapshot of the plan
  fields, their origins, the creation mode, and the plan revision, taken at
  enqueue;
- the *resolved brief* (`lyre.brief/1`): caption, lyrics, and metas after any
  planning, with each field's origin and a stable content hash;
- the *render flags*: the LM switches `generate_music` reads, set explicitly
  because upstream defaults `use_cot_caption` and `use_cot_language` to true.

Pure Python: no FastAPI, SQLite, storage, `acestep`, or `torch` imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from typing import Any

CREATION_SCHEMA = "lyre.creation/1"
BRIEF_SCHEMA = "lyre.brief/1"
PLAN_BRIEF_SCHEMA = "lyre.plan-brief/1"

MODES = ("from_idea", "from_brief")
TRACKED_FIELDS = (
    "caption",
    "lyrics",
    "vocal_language",
    "bpm",
    "keyscale",
    "timesignature",
    "duration_sec",
)
META_FIELDS = ("vocal_language", "bpm", "keyscale", "timesignature", "duration_sec")
ORIGINS = ("user", "planner", "preset", "default", "unknown")
# Origins a planner result may replace in `from_idea`. `user` and `preset`
# are deliberate choices and survive planning (precedence rule 1). `unknown`
# is legacy data, which keeps its old Simple-mode behavior.
_REPLACEABLE = {"planner", "default", "unknown"}
_KEPT = {"user", "preset"}

# Upstream acestep/constants.py bounds (ACE_STEP_REVISION). The conformance
# test compares these against the installed package.
BPM_MIN, BPM_MAX = 30, 300
DURATION_MIN, DURATION_MAX = 10, 600
TIME_SIGNATURES = ("2", "3", "4", "6", "2/4", "3/4", "4/4", "6/8")
# Upstream's VALID_LANGUAGES, minus its Cantonese code: that token is on the
# forbidden-engine keyword list in tests/test_spec_lock.py, so Lyre cannot
# spell it in source. Cantonese lyrics can use "zh".
VALID_LANGUAGES = (
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "zh", "unknown",
)  # fmt: skip
# Upstream documents these as limits; past them it may truncate. Lyre warns.
CAPTION_SOFT_LIMIT = 512
LYRICS_SOFT_LIMIT = 4096
INSTRUMENTAL_LYRICS = "[Instrumental]"

SEED_MIN, SEED_MAX = 1, 2**31 - 1


class BriefError(ValueError):
    """An invalid creation request, raised before any GPU work. `param`
    names the field and `code` is the `/v1` error code."""

    def __init__(self, message: str, *, param: str | None, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.param = param
        self.code = code


def draw_seed() -> int:
    """A concrete synthesis seed for a `-1` request (SPEC.md sec 16.4)."""
    return random.SystemRandom().randint(SEED_MIN, SEED_MAX)


def is_unset(field: str, value: Any) -> bool:
    """True when `value` leaves the field for the planner or the model."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        if field == "vocal_language" and text.lower() == "unknown":
            return True
    return False


def is_instrumental_placeholder(lyrics: str) -> bool:
    return lyrics.strip().lower() == INSTRUMENTAL_LYRICS.lower()


def legacy_mode(plan: Mapping[str, Any]) -> str:
    """The pre-section-16 rule: Simple mode iff an idea and no brief."""
    query = str(plan.get("query") or "").strip()
    if query and not plan.get("caption") and not plan.get("lyrics"):
        return "from_idea"
    return "from_brief"


def plan_mode(plan: Mapping[str, Any]) -> str:
    mode = plan.get("creation_mode")
    return mode if mode in MODES else legacy_mode(plan)


def plan_origins(plan: Mapping[str, Any]) -> dict[str, str]:
    """Every tracked field's origin. Missing entries are `unknown`."""
    raw = plan.get("field_origins")
    out: dict[str, str] = {}
    for field in TRACKED_FIELDS:
        origin = raw.get(field) if isinstance(raw, Mapping) else None
        out[field] = origin if origin in ORIGINS else "unknown"
    return out


def default_origins() -> dict[str, str]:
    return dict.fromkeys(TRACKED_FIELDS, "default")


def apply_write_semantics(
    current: Mapping[str, Any] | None,
    new_plan: dict[str, Any],
    sent: Mapping[str, Any],
) -> dict[str, Any]:
    """Set `creation_mode`, `field_origins`, `revision`, and `brief` on a
    plan about to be written (SPEC.md sec 16.3 "Origins on write").

    `current` is the plan on disk (None for a new project), `new_plan` the
    validated plan, and `sent` the raw body, so an echoed origin or mode can
    be told apart from a deliberate one. A tracked field whose value changed
    becomes `user` (or `default` when cleared) unless the body names a
    different origin for it than the one on disk. A changed idea switches to
    `from_idea` unless the body sets the mode itself. `brief` is
    server-managed: clients cannot write it.
    """
    current = current or {}
    current_origins = plan_origins(current) if current else default_origins()
    claimed = sent.get("field_origins")
    claimed_origins = claimed if isinstance(claimed, Mapping) else {}

    origins: dict[str, str] = {}
    for field in TRACKED_FIELDS:
        before = current_origins[field]
        claim = claimed_origins.get(field)
        if claim in ORIGINS and claim != before:
            origins[field] = claim
        elif new_plan.get(field) != current.get(field):
            origins[field] = "default" if is_unset(field, new_plan.get(field)) else "user"
        else:
            origins[field] = before

    current_mode = plan_mode(current) if current else "from_idea"
    claimed_mode = sent.get("creation_mode")
    if claimed_mode in MODES and claimed_mode != current_mode:
        mode = claimed_mode
    elif str(new_plan.get("query") or "") != str(current.get("query") or ""):
        mode = "from_idea"
    else:
        mode = current_mode

    new_plan["creation_mode"] = mode
    new_plan["field_origins"] = origins
    new_plan["revision"] = int(current.get("revision") or 0) + 1
    new_plan["brief"] = current.get("brief") if isinstance(current.get("brief"), dict) else None
    return new_plan


def _effective_mode(mode: str, query: str, caption: str, lyrics: str) -> str:
    has_idea = bool(query.strip())
    has_brief = bool(caption.strip() or lyrics.strip())
    if mode == "from_idea" and not has_idea and has_brief:
        return "from_brief"
    if mode == "from_brief" and not has_brief and has_idea:
        return "from_idea"
    return mode


def creation_input(
    plan: Mapping[str, Any],
    *,
    action: str = "generate",
    mode_override: str | None = None,
) -> dict[str, Any]:
    """Freeze what one job will render from (SPEC.md sec 16.4). No
    validation; `validate_creation` does that."""
    if mode_override is not None and mode_override not in MODES:
        raise BriefError(
            f"creation_mode must be one of {list(MODES)} (got {mode_override!r})",
            param="creation_mode",
        )
    query = str(plan.get("query") or "")
    caption = str(plan.get("caption") or "")
    lyrics = str(plan.get("lyrics") or "")
    if action == "generate":
        mode = _effective_mode(mode_override or plan_mode(plan), query, caption, lyrics)
    else:
        mode = "from_brief"
    preset = plan.get("preset")
    return {
        "schema": CREATION_SCHEMA,
        "action": action,
        "mode": mode,
        "query": query,
        "caption": caption,
        "lyrics": lyrics,
        "instrumental": bool(plan.get("instrumental")),
        "vocal_language": plan.get("vocal_language"),
        "bpm": plan.get("bpm"),
        "keyscale": plan.get("keyscale"),
        "timesignature": plan.get("timesignature"),
        "duration_sec": plan.get("duration_sec"),
        "caption_rewrite": bool(plan.get("caption_rewrite")),
        "origins": plan_origins(plan),
        "plan_revision": int(plan.get("revision") or 0),
        "preset": dict(preset) if isinstance(preset, Mapping) else None,
    }


def parse_keyscale(value: str) -> str | None:
    """Canonical `"<note><accidental> <major|minor>"`, or None when invalid.
    Accepts `C major`, `c Major`, `Am`, `F#m`, `Bb`, `Eb min`."""
    match = re.fullmatch(r"([A-Ga-g])([#b♯♭]?)\s*(.*)", value.strip())
    if match is None:
        return None
    note, accidental, rest = match.group(1).upper(), match.group(2), match.group(3).strip()
    if rest.lower() in ("", "maj", "major"):
        mode = "major"
    elif rest == "m" or rest.lower() in ("min", "minor"):
        mode = "minor"
    else:
        return None
    return f"{note}{accidental} {mode}"


def _lyric_warnings(lyrics: str) -> list[str]:
    warnings: list[str] = []
    for number, line in enumerate(lyrics.split("\n"), start=1):
        stripped = line.strip()
        if stripped.startswith("[") and "]" not in stripped:
            warnings.append(f"lyrics line {number}: section tag is not closed with ']'")
    if len(lyrics) > LYRICS_SOFT_LIMIT:
        warnings.append(
            f"lyrics are {len(lyrics)} characters; ACE-Step documents a {LYRICS_SOFT_LIMIT} "
            "character limit and may truncate the rest"
        )
    return warnings


def validate_creation(intent: dict[str, Any]) -> dict[str, Any]:
    """Reject what cannot render, before it reaches the queue (precedence
    rule 3). Returns the intent with transport normalizations applied and a
    `warnings` list. Only `generate` is checked: cover, repaint, and the
    studio operations keep their existing validation."""
    intent = dict(intent)
    warnings: list[str] = []
    if intent["action"] != "generate":
        intent["warnings"] = warnings
        return intent

    lyrics = intent["lyrics"]
    normalized = lyrics.replace("\r\n", "\n").replace("\r", "\n")
    if normalized != lyrics:
        intent["lyrics_original"] = lyrics
        intent["lyrics"] = normalized
        warnings.append("lyrics line endings were normalized to LF; the words are unchanged")

    if intent["mode"] == "from_idea" and not intent["query"].strip():
        # Nothing to plan and nothing written: render with every field left
        # to ACE-Step, as an empty Custom plan always has.
        intent["mode"] = "from_brief"
    if not (intent["query"].strip() or intent["caption"].strip() or intent["lyrics"].strip()):
        warnings.append("no idea, caption, or lyrics: ACE-Step chooses everything")
    if (
        intent["instrumental"]
        and intent["lyrics"].strip()
        and not is_instrumental_placeholder(intent["lyrics"])
    ):
        raise BriefError(
            "Instrumental is on but the plan has lyrics. Clear the lyrics, or turn "
            "Instrumental off to keep them; Lyre will not delete them for you.",
            param="lyrics",
            code="instrumental_lyrics_conflict",
        )

    bpm = intent["bpm"]
    if not is_unset("bpm", bpm) and (
        isinstance(bpm, bool) or not isinstance(bpm, int) or not BPM_MIN <= bpm <= BPM_MAX
    ):
        raise BriefError(
            f"bpm must be a whole number from {BPM_MIN} to {BPM_MAX} (got {bpm!r})",
            param="bpm",
        )
    duration = intent["duration_sec"]
    if not is_unset("duration_sec", duration) and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not DURATION_MIN <= duration <= DURATION_MAX
    ):
        raise BriefError(
            f"duration_sec must be from {DURATION_MIN} to {DURATION_MAX} seconds "
            f"(got {duration!r})",
            param="duration_sec",
        )
    keyscale = intent["keyscale"]
    if not is_unset("keyscale", keyscale):
        canonical = parse_keyscale(str(keyscale))
        if canonical is None:
            raise BriefError(
                f"keyscale must be a note A-G, an optional #/b, and major or minor "
                f"(got {keyscale!r})",
                param="keyscale",
            )
        intent["keyscale"] = canonical
    meter = intent["timesignature"]
    if not is_unset("timesignature", meter):
        if str(meter).strip() not in TIME_SIGNATURES:
            raise BriefError(
                f"timesignature must be one of {', '.join(TIME_SIGNATURES)} (got {meter!r})",
                param="timesignature",
            )
        intent["timesignature"] = str(meter).strip()
    language = intent["vocal_language"]
    if not is_unset("vocal_language", language):
        code = str(language).strip().lower()
        if code not in VALID_LANGUAGES:
            raise BriefError(
                f"vocal_language {language!r} is not a language ACE-Step supports",
                param="vocal_language",
            )
        intent["vocal_language"] = code

    if len(intent["caption"]) > CAPTION_SOFT_LIMIT:
        warnings.append(
            f"caption is {len(intent['caption'])} characters; ACE-Step documents a "
            f"{CAPTION_SOFT_LIMIT} character limit"
        )
    warnings.extend(_lyric_warnings(intent["lyrics"]))
    intent["warnings"] = warnings
    return intent


def _user_text(intent: Mapping[str, Any], field: str) -> str:
    """The caption or lyrics the user (or a preset, or legacy data) wrote,
    which planning must keep. Planner and default text is replaceable."""
    value = str(intent[field] or "")
    if intent["origins"][field] in _REPLACEABLE - {"unknown"}:
        return ""
    if field == "lyrics" and is_instrumental_placeholder(value):
        return ""
    return value if value.strip() else ""


def _explicit_metas(intent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: intent[field]
        for field in META_FIELDS
        if intent["origins"][field] in _KEPT and not is_unset(field, intent[field])
    }


def planner_request(intent: Mapping[str, Any]) -> dict[str, Any] | None:
    """Which upstream planner `from_idea` calls, and with what. None means
    no planning (`from_brief`, or not a generate).

    Without user lyrics: `create_sample(query, instrumental, vocal_language)`
    drafts caption, lyrics, and metas. With user lyrics: `format_sample`
    plans caption and metas around them, constrained by the explicit metas.
    """
    if intent["action"] != "generate" or intent["mode"] != "from_idea":
        return None
    language = intent["vocal_language"]
    lyrics = _user_text(intent, "lyrics")
    if lyrics:
        metas = _explicit_metas(intent)
        user_metadata: dict[str, Any] = {}
        for field, upstream in (
            ("bpm", "bpm"),
            ("duration_sec", "duration"),
            ("keyscale", "keyscale"),
            ("timesignature", "timesignature"),
            ("vocal_language", "language"),
        ):
            if field in metas:
                user_metadata[upstream] = metas[field]
        return {
            "planner": "format_sample",
            "caption": _user_text(intent, "caption") or intent["query"],
            "lyrics": lyrics,
            "user_metadata": user_metadata or None,
        }
    return {
        "planner": "create_sample",
        "query": intent["query"],
        "instrumental": intent["instrumental"],
        "vocal_language": None if is_unset("vocal_language", language) else language,
    }


def _brief_record(
    intent: Mapping[str, Any],
    values: Mapping[str, Any],
    origins: Mapping[str, str],
    planner: Mapping[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    brief: dict[str, Any] = {
        "schema": BRIEF_SCHEMA,
        "mode": intent["mode"],
        "query": intent["query"],
        **{field: values[field] for field in TRACKED_FIELDS},
        "instrumental": intent["instrumental"],
        "caption_rewrite": intent["caption_rewrite"],
        "origins": dict(origins),
        "planner": dict(planner) if planner is not None else None,
        "preset": intent.get("preset"),
        "warnings": list(warnings),
    }
    if "lyrics_original" in intent:
        brief["lyrics_original"] = intent["lyrics_original"]
    if brief["instrumental"] and not brief["lyrics"].strip():
        brief["lyrics"] = INSTRUMENTAL_LYRICS
    brief["content_hash"] = content_hash(brief)
    return brief


def brief_without_planning(intent: Mapping[str, Any]) -> dict[str, Any]:
    """`from_brief`, and every non-generate action: render the snapshot."""
    values = {field: intent[field] for field in TRACKED_FIELDS}
    return _brief_record(intent, values, intent["origins"], None, intent.get("warnings", []))


def apply_planner(
    intent: Mapping[str, Any],
    sample: Mapping[str, Any],
    planner: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a planner result onto the frozen intent by precedence.

    `sample` uses upstream's result names (`caption`, `lyrics`, `bpm`,
    `duration`, `keyscale`, `language`, `timesignature`). A field the user or
    a preset set keeps its value whatever the planner said; user lyrics are
    never replaced; a planner value fills only a replaceable field, and is
    recorded with origin `planner`.
    """
    upstream_names = {
        "caption": "caption",
        "lyrics": "lyrics",
        "bpm": "bpm",
        "duration_sec": "duration",
        "keyscale": "keyscale",
        "vocal_language": "language",
        "timesignature": "timesignature",
    }
    origins = dict(intent["origins"])
    values: dict[str, Any] = {}
    warnings = list(intent.get("warnings", []))
    for field in TRACKED_FIELDS:
        current = intent[field]
        suggestion = sample.get(upstream_names[field])
        if field in ("caption", "lyrics"):
            kept = _user_text(intent, field)
            if kept:
                values[field] = current
                continue
            if field == "lyrics" and intent["instrumental"]:
                values[field] = INSTRUMENTAL_LYRICS
                origins[field] = "default"
                continue
        elif origins[field] in _KEPT and not is_unset(field, current):
            values[field] = current
            if not is_unset(field, suggestion) and suggestion != current:
                warnings.append(f"planner suggested {field}={suggestion!r}; kept your {current!r}")
            continue
        if not is_unset(field, suggestion):
            values[field] = suggestion
            origins[field] = "planner"
        else:
            values[field] = current
    return _brief_record(intent, values, origins, planner, warnings)


def render_flags(brief: Mapping[str, Any], action: str) -> dict[str, bool]:
    """The LM switches for `GenerationParams`. Every one is explicit:
    upstream defaults `use_cot_caption` and `use_cot_language` to true, and
    `generate_music` runs the LM whenever any CoT flag is set."""
    if action != "generate":
        # Upstream skips the LM for every direct-conditioning task anyway.
        return {
            "thinking": False,
            "use_cot_caption": False,
            "use_cot_language": False,
            "use_cot_metas": False,
        }
    origins = brief["origins"]
    planned = brief.get("planner") is not None or "planner" in origins.values()
    caption_empty = not str(brief["caption"] or "").strip()
    return {
        "thinking": brief["mode"] == "from_idea" or planned or bool(brief["caption_rewrite"]),
        "use_cot_caption": caption_empty
        or (bool(brief["caption_rewrite"]) and origins["caption"] != "planner"),
        "use_cot_language": is_unset("vocal_language", brief["vocal_language"])
        and not brief["instrumental"],
        "use_cot_metas": any(
            is_unset(field, brief[field])
            for field in ("bpm", "keyscale", "timesignature", "duration_sec")
        ),
    }


def generation_fields(brief: Mapping[str, Any], action: str) -> dict[str, Any]:
    """The creative half of the `GenerationParams` kwargs, identical for the
    real adapter and the mock (the parity tests compare them)."""
    flags = render_flags(brief, action)
    return {
        "caption": str(brief["caption"] or ""),
        "lyrics": str(brief["lyrics"] or ""),
        "bpm": None if is_unset("bpm", brief["bpm"]) else brief["bpm"],
        "keyscale": "" if is_unset("keyscale", brief["keyscale"]) else brief["keyscale"],
        "duration": None
        if is_unset("duration_sec", brief["duration_sec"])
        else brief["duration_sec"],
        "instrumental": bool(brief["instrumental"]),
        "vocal_language": "unknown"
        if is_unset("vocal_language", brief["vocal_language"])
        else brief["vocal_language"],
        # ACE-Step calls .strip() on timesignature; never forward None.
        "timesignature": ""
        if is_unset("timesignature", brief["timesignature"])
        else str(brief["timesignature"]),
        **flags,
    }


def actual_values(
    brief: Mapping[str, Any],
    flags: Mapping[str, bool],
    audio_params: Mapping[str, Any] | None,
    lm_metadata: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """What the DiT actually rendered from, with origins. Fields the brief
    left unset and upstream's CoT filled (`audio["params"]["cot_*"]`) are
    origin `model`; a caption or language CoT rewrote comes from the LM
    metadata. Upstream's `extra_outputs` carries no caption or metas of its
    own, so nothing else is trusted."""
    values = {field: brief[field] for field in TRACKED_FIELDS}
    origins = dict(brief["origins"])
    params = audio_params or {}
    lm = lm_metadata or {}
    cot_names = {
        "bpm": "cot_bpm",
        "keyscale": "cot_keyscale",
        "timesignature": "cot_timesignature",
        "duration_sec": "cot_duration",
        "vocal_language": "cot_vocal_language",
    }
    for field, cot in cot_names.items():
        if is_unset(field, values[field]) and not is_unset(field, params.get(cot)):
            values[field] = params[cot]
            origins[field] = "model"
    lm_caption = str(lm.get("caption") or "")
    if flags.get("use_cot_caption") and lm_caption.strip() and lm_caption != values["caption"]:
        values["caption"] = lm_caption
        origins["caption"] = "model"
    if flags.get("use_cot_language") and not is_unset("vocal_language", lm.get("vocal_language")):
        values["vocal_language"] = lm["vocal_language"]
        origins["vocal_language"] = "model"
    return values, origins


def plan_patch_from_brief(brief: Mapping[str, Any], job_id: str) -> dict[str, Any] | None:
    """The planner-filled fields to merge into the editable draft after a
    planned render (SPEC.md sec 16.3 "After a planned render"). None when
    nothing was planned. `field_origins` here is a partial map that storage
    merges key by key."""
    if brief.get("planner") is None:
        return None
    filled = [field for field in TRACKED_FIELDS if brief["origins"][field] == "planner"]
    # The instrumental placeholder is an application default, but it is
    # still part of what was rendered, so the draft shows it too.
    if (
        brief["instrumental"]
        and "lyrics" not in filled
        and brief["origins"]["lyrics"] == "default"
        and is_instrumental_placeholder(brief["lyrics"])
    ):
        filled.append("lyrics")
    patch: dict[str, Any] = {field: brief[field] for field in filled}
    patch["field_origins"] = {field: brief["origins"][field] for field in filled}
    patch["creation_mode"] = "from_brief"
    patch["brief"] = {
        "schema": PLAN_BRIEF_SCHEMA,
        "source_query": brief["query"],
        "planner": brief["planner"],
        "content_hash": brief["content_hash"],
        "job_id": job_id,
    }
    return patch


_HASH_KEYS = (*TRACKED_FIELDS, "instrumental", "caption_rewrite", "origins", "mode")


def content_hash(brief: Mapping[str, Any]) -> str:
    """Stable sha256 over what the brief renders, independent of key order."""
    payload = {key: brief.get(key) for key in _HASH_KEYS}
    payload["planned"] = brief.get("planner") is not None
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
