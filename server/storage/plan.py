"""plan.json defaults, lenient read normalization, and strict write validation.

Reads stay lenient (`_normalize_plan`) so plan files written before a field
existed keep loading; only the PUT write path runs `validate_plan`
(SPEC.md sec 7.2).
"""

from __future__ import annotations

import math

from server.storage import jsonio
from server.storage.paths import plan_json_path


def _normalize_plan(plan: dict) -> dict:
    """Fill in plan fields added after some plan.json files were already
    written to disk (SPEC.md sec 9.2 `caption_rewrite`), same pattern as
    `_normalize_take_meta` for takes: a plan.json saved before this field
    existed has no key at all, and must keep behaving exactly like it did
    then -- worker.acestep_worker.run_job (and the frontend's Plan type /
    checkbox) would otherwise have to special-case a missing key at every
    read site instead of once here."""
    plan.setdefault("caption_rewrite", False)
    return plan


def _read_plan_or_default(project_id: str) -> dict:
    path = plan_json_path(project_id)
    if not path.exists():
        return default_plan()
    return _normalize_plan(jsonio._read_json(path))


def default_plan() -> dict:
    return {
        "query": "",
        "caption": "",
        "negative": [],
        "lyrics": "",
        "instrumental": False,
        "vocal_language": "en",
        "bpm": None,
        "keyscale": None,
        "timesignature": "4/4",
        "duration_sec": 120,
        "sections": [],
        # SPEC.md sec 9.2/7.2: Custom-mode checkbox controlling whether
        # ACE-Step's LM ("thinking") is allowed to rewrite the user's caption.
        # New plans allow rewriting until the user disables it, as required
        # by SPEC.md sec 7.2. _normalize_plan separately keeps legacy plans
        # without this field on their historical False behavior.
        "caption_rewrite": True,
    }


# plan.json field types per SPEC.md sec 7.2. `bool` is a subclass of `int` in
# Python, so every numeric check must reject it explicitly -- a JSON `true`
# would otherwise pass `isinstance(..., int)` and reach the worker as a
# number. `_plan_number` additionally rejects NaN/+-inf, which `json.loads`
# accepts but which would serialize back out as non-standard JSON tokens and
# confuse downstream consumers.
_PLAN_STRING_FIELDS = ("query", "caption", "lyrics", "vocal_language", "timesignature")
_PLAN_BOOL_FIELDS = ("instrumental", "caption_rewrite")
# The only keys a section dict may carry (SPEC.md sec 7.2 sections[]). Unknown
# keys inside a section are dropped for the same reason unknown top-level keys
# are -- they'd otherwise be persisted verbatim and flow to worker/SPA.
_PLAN_SECTION_KEYS = ("name", "start_sec", "end_sec", "lyrics")


def _plan_type_name(value: object) -> str:
    return type(value).__name__


def _plan_type_error(field: str, expected: str, value: object) -> ValueError:
    """ValueError naming the offending field (server/app.py maps it to 400)."""
    return ValueError(
        f"invalid plan field '{field}': expected {expected}, got {_plan_type_name(value)}"
    )


def _plan_string(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise _plan_type_error(field, "a string", value)
    return value


def _plan_bool(field: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise _plan_type_error(field, "a boolean", value)
    return value


def _plan_number(field: str, value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _plan_type_error(field, "a number", value)
    if isinstance(value, float) and not math.isfinite(value):
        raise _plan_type_error(field, "a finite number", value)
    return value


def validate_plan(plan: object) -> dict:
    """Validate and normalize a client-submitted plan for `PUT /plan`
    (SPEC.md sec 7.2), returning a fresh dict safe to persist:

    - missing keys are filled from `default_plan()`, so a partial body can
      never produce a plan the worker (`.get()` reads) or the SPA (Plan type
      assumes exact shapes) has to special-case;
    - field types are enforced -- `query`/`caption`/`lyrics`/
      `vocal_language`/`timesignature` strings, `negative` a list of strings,
      `instrumental`/`caption_rewrite` booleans, `bpm` int-or-null,
      `keyscale` string-or-null, `duration_sec` a number, `sections` a list
      of dicts each with `name` (str), `start_sec`/`end_sec` (numbers) and an
      optional `lyrics` (str). Any invalid value raises `ValueError` (->
      HTTP 400 via server/app.py) naming the offending field;
    - unknown top-level keys (and unknown keys inside a section) are dropped
      rather than persisted.

    Only the PUT write path runs this (`save_plan`). Reads stay on the
    lenient `_normalize_plan`, so plan.json files written before this
    validation existed keep loading exactly as before.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"invalid plan: expected a JSON object, got {_plan_type_name(plan)}")

    defaults = default_plan()
    # Copy only known keys (drops unknown top-level keys), filling any that
    # are missing from default_plan().
    normalized: dict = {key: plan.get(key, defaults[key]) for key in defaults}

    for field in _PLAN_STRING_FIELDS:
        _plan_string(field, normalized[field])
    for field in _PLAN_BOOL_FIELDS:
        _plan_bool(field, normalized[field])

    negative = normalized["negative"]
    if not isinstance(negative, list):
        raise _plan_type_error("negative", "a list of strings", negative)
    for i, item in enumerate(negative):
        _plan_string(f"negative[{i}]", item)

    bpm = normalized["bpm"]
    if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, int)):
        raise _plan_type_error("bpm", "an integer or null", bpm)

    keyscale = normalized["keyscale"]
    if keyscale is not None and not isinstance(keyscale, str):
        raise _plan_type_error("keyscale", "a string or null", keyscale)

    _plan_number("duration_sec", normalized["duration_sec"])

    sections = normalized["sections"]
    if not isinstance(sections, list):
        raise _plan_type_error("sections", "a list", sections)
    cleaned_sections: list[dict] = []
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise _plan_type_error(f"sections[{i}]", "an object", section)
        cleaned = {key: section[key] for key in _PLAN_SECTION_KEYS if key in section}
        if "name" not in cleaned:
            raise ValueError(f"invalid plan field 'sections[{i}].name': missing required key")
        _plan_string(f"sections[{i}].name", cleaned["name"])
        for bound in ("start_sec", "end_sec"):
            if bound not in cleaned:
                raise ValueError(
                    f"invalid plan field 'sections[{i}].{bound}': missing required key"
                )
            _plan_number(f"sections[{i}].{bound}", cleaned[bound])
        if "lyrics" in cleaned:
            _plan_string(f"sections[{i}].lyrics", cleaned["lyrics"])
        cleaned_sections.append(cleaned)
    normalized["sections"] = cleaned_sections
    return normalized
