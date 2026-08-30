"""Reading what came back: plan fill-in, quality score, and lyric timestamps.

Upstream returns results as loosely-shaped dicts or objects depending on
version, so every field is read defensively and mapped onto Lyre's own
plan/meta shapes (SPEC.md sec 7.2/7.3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worker.acestep_worker.api import _api_call
from worker.acestep_worker.settings import _LOCK, GENERATION_STEPS


def _repaint_meta(job: dict[str, Any]) -> dict | None:
    if job.get("action") != "repaint":
        return None
    return {
        "start": job.get("repainting_start", 0),
        "end": job.get("repainting_end", -1),
    }


def _is_simple_mode(job: dict[str, Any], plan: dict[str, Any]) -> bool:
    """SPEC.md sec 7.2: Simple mode is "user types query"; caption/lyrics
    stay empty until the LM fills them in."""
    return (
        job.get("action") == "generate"
        and bool(plan.get("query"))
        and not plan.get("caption")
        and not plan.get("lyrics")
    )


def _plan_from_query(create_sample_fn: Any, lm: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Simple mode (SPEC.md sec 7.2): turn `query` into a full plan via
    ACE-Step's module-level `create_sample` (it takes the LM handler, it is
    not a method on it), then merge the result onto `plan`."""
    sample = _api_call(
        "create_sample",
        create_sample_fn,
        # Upstream spells this `llm_handler`, with two Ls -- unlike
        # `AceStepHandler`/`LLMHandler.initialize`, which take neither. Passing
        # `lm_handler` raised TypeError on every real simple-mode generate.
        llm_handler=lm,
        query=plan.get("query"),
        instrumental=plan.get("instrumental", False),
        # Without this, create_sample is free to pick any language for the
        # lyrics it writes -- the plan's own vocal_language (default "en",
        # SPEC.md sec 7.2) is what's supposed to constrain that.
        vocal_language=plan.get("vocal_language"),
    )

    def _field(name: str, fallback: Any) -> Any:
        if isinstance(sample, dict):
            return sample.get(name, fallback)
        return getattr(sample, name, fallback)

    # Mirrors generate_music's GenerationResult.success handling further
    # down in this module: absent means the installed acestep predates this
    # field (treated as success, matching the fakes/tests below), but an
    # explicit False must fail the job instead of silently generating from
    # an empty/fallback caption and lyrics.
    if not _field("success", True):
        detail = _field("error", None) or _field("message", None) or _field("status", None)
        raise RuntimeError(f"create_sample reported failure: {detail or 'no detail provided'}")

    return {
        **plan,
        "caption": _field("caption", plan.get("caption", "")),
        "lyrics": _field("lyrics", plan.get("lyrics", "")),
        "bpm": _field("bpm", plan.get("bpm")),
        "keyscale": _field("keyscale", plan.get("keyscale")),
        "duration_sec": _field("duration", plan.get("duration_sec")),
        # create_sample's result field is "language", not "vocal_language"
        # (that's Lyre's own plan.json field name -- see storage.default_plan).
        "vocal_language": _field("language", plan.get("vocal_language")),
        "timesignature": _field("timesignature", plan.get("timesignature")),
    }


def _result_field(container: Any, name: str, fallback: Any) -> Any:
    if isinstance(container, dict):
        return container.get(name, fallback)
    return getattr(container, name, fallback)


def _quality_score(
    handler: Any,
    extra_lookup: Callable[[str, Any], Any],
    lyrics: Any,
    vocal_language: Any,
    dit_profile: str,
) -> float | None:
    """SPEC.md sec 12 Phase 4: "ACE-Step quality score on takes". Checked
    upstream `acestep.inference.generate_music`/`GenerationResult` directly
    (ace-step/ACE-Step-1.5 main branch, since this isn't documented in
    INFERENCE.md/Tutorial.md at the field level): neither `GenerationResult`
    nor its `extra_outputs`/per-audio dict carries a score field on its own.
    There *is* a real, callable scoring API though -- `AceStepHandler.
    get_lyric_score(...)` (the DiT Lyrics Alignment Score Tutorial.md sec
    "Automatic Scoring Mechanism" calls upstream's own favorite metric) --
    it just isn't a value `generate_music` hands back; it's a second forward
    pass through the already-loaded model, reusing decoder-attention tensors
    (`pred_latents`, `encoder_hidden_states`, `encoder_attention_mask`,
    `context_latents`, `lyric_token_idss`) that `generate_music` *does* put
    in `extra_outputs` -- unless ACE-Step's own save-memory mode is on, in
    which case they're absent and there's nothing to score from. Requires
    non-empty lyrics (mirrors upstream's own scoring UI, which skips
    alignment scoring for instrumental/lyric-less takes the same way).
    Best-effort like every other `extra_outputs` field pulled in `run_job`:
    a scoring failure must never fail an otherwise-successful take, so this
    never raises `WorkerUnavailable` (or anything else) -- unlike the
    surrounding `_api_call`-wrapped calls that *are* allowed to fail the
    job, this one only ever falls back to `None`. Runs under `_LOCK`
    because it's a real extra GPU forward pass through the shared loaded
    model (SPEC.md sec 4.3: "one GPU occupant... jobs serialize"), same as
    the `generate_music` call above.
    """
    if not lyrics or not str(lyrics).strip():
        return None
    pred_latents = extra_lookup("pred_latents", None)
    encoder_hidden_states = extra_lookup("encoder_hidden_states", None)
    encoder_attention_mask = extra_lookup("encoder_attention_mask", None)
    context_latents = extra_lookup("context_latents", None)
    lyric_token_ids = extra_lookup("lyric_token_idss", None)
    if None in (
        pred_latents,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        lyric_token_ids,
    ):
        return None
    try:
        with _LOCK:
            score_result = handler.get_lyric_score(
                pred_latent=pred_latents,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                lyric_token_ids=lyric_token_ids,
                vocal_language=vocal_language or "en",
                inference_steps=GENERATION_STEPS[dit_profile],
                seed=42,
            )
        if not _result_field(score_result, "success", False):
            return None
        dit_score = _result_field(score_result, "dit_score", None)
        return float(dit_score) if dit_score is not None else None
    except Exception:
        return None


def _lyric_timestamps(
    handler: Any,
    extra_lookup: Callable[[str, Any], Any],
    lyrics: Any,
    vocal_language: Any,
    dit_profile: str,
    duration_sec: Any,
) -> str | None:
    """SPEC.md sec 12 Phase 4: "LRC if upstream provides timestamps" -- SPEC
    hedges this as conditional on upstream support. Checked upstream directly
    (ace-step/ACE-Step-1.5 main branch; same gap as `_quality_score` above --
    not documented at the field level in INFERENCE.md/Tutorial.md):
    `AceStepHandler` mixes in `LyricTimestampMixin`
    (`acestep/core/generation/handler/lyric_timestamp.py`), and its
    `get_lyric_timestamp(...)` genuinely exists and returns real per-line
    timing -- upstream support is real, not absent. Its signature mirrors
    `get_lyric_score` almost exactly: the same five decoder-attention tensors
    (`pred_latents`/`encoder_hidden_states`/`encoder_attention_mask`/
    `context_latents`/`lyric_token_idss`) `generate_music` already puts in
    `extra_outputs` when lyrics are non-empty (absent under ACE-Step's
    save-memory mode, same as scoring), plus `total_duration_seconds` to
    scale the timestamps and the usual `vocal_language`/`inference_steps`/
    `seed`. Internally it delegates to
    `MusicStampsAligner.get_timestamps_and_lrc`
    (`acestep/core/scoring/dit_alignment.py`), which already formats
    `[mm:ss.xx]line text` per lyric line (structure tags like [Verse]/
    [Chorus] are stripped before alignment, not emitted as timed lines) --
    so the returned `lrc_text` is used as-is here rather than re-derived from
    `sentence_timestamps`/`token_timestamps`. Best-effort like
    `_quality_score`: a timestamp failure must never fail an otherwise-
    successful take, so this never raises -- it only ever falls back to
    `None`, meaning no `lyrics.lrc` is written for the take."""
    if not lyrics or not str(lyrics).strip():
        return None
    if not duration_sec:
        return None
    pred_latents = extra_lookup("pred_latents", None)
    encoder_hidden_states = extra_lookup("encoder_hidden_states", None)
    encoder_attention_mask = extra_lookup("encoder_attention_mask", None)
    context_latents = extra_lookup("context_latents", None)
    lyric_token_ids = extra_lookup("lyric_token_idss", None)
    if None in (
        pred_latents,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        lyric_token_ids,
    ):
        return None
    try:
        with _LOCK:
            result = handler.get_lyric_timestamp(
                pred_latent=pred_latents,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                lyric_token_ids=lyric_token_ids,
                total_duration_seconds=float(duration_sec),
                vocal_language=vocal_language or "en",
                inference_steps=GENERATION_STEPS[dit_profile],
                seed=42,
            )
        if not _result_field(result, "success", False):
            return None
        lrc_text = _result_field(result, "lrc_text", None)
        if not lrc_text or not str(lrc_text).strip():
            return None
        return str(lrc_text)
    except Exception:
        return None
