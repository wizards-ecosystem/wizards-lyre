"""Reading what came back: plan fill-in, quality score, and lyric timestamps.

Upstream returns results as loosely-shaped dicts or objects depending on
version, so every field is read defensively and mapped onto Lyre's own
plan/meta shapes (SPEC.md sec 7.2/7.3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from server.generation import brief as creation
from worker.acestep_worker.api import _api_call
from worker.acestep_worker.settings import _LOCK, GENERATION_STEPS


def _repaint_meta(job: dict[str, Any]) -> dict | None:
    if job.get("action") != "repaint":
        return None
    return {
        "start": job.get("repainting_start", 0),
        "end": job.get("repainting_end", -1),
    }


def _run_planner(
    request: dict[str, Any],
    create_sample_fn: Any,
    format_sample_fn: Any,
    lm: Any,
) -> dict[str, Any]:
    """Run the upstream planner `server.generation.brief.planner_request`
    chose, and return its result under upstream's own field names
    (`caption`, `lyrics`, `bpm`, `duration`, `keyscale`, `language`,
    `timesignature`). Precedence is applied afterwards by
    `brief.apply_planner`, never here. Both planners are module-level
    functions that take the LM handler as `llm_handler` (two Ls)."""
    if request["planner"] == "format_sample":
        sample = _api_call(
            "format_sample",
            format_sample_fn,
            llm_handler=lm,
            caption=request["caption"],
            lyrics=request["lyrics"],
            user_metadata=request["user_metadata"],
        )
    else:
        sample = _api_call(
            "create_sample",
            create_sample_fn,
            llm_handler=lm,
            query=request["query"],
            instrumental=request["instrumental"],
            vocal_language=request["vocal_language"],
        )

    def _field(name: str) -> Any:
        if isinstance(sample, dict):
            return sample.get(name)
        return getattr(sample, name, None)

    # Absent `success` means an older acestep (treated as success, as for
    # generate_music below); an explicit False fails the job instead of
    # rendering from an empty caption and lyrics.
    success = _field("success")
    if success is not None and not success:
        detail = _field("error") or _field("message") or _field("status_message")
        raise RuntimeError(
            f"{request['planner']} reported failure: {detail or 'no detail provided'}"
        )
    names = ("caption", "lyrics", "bpm", "duration", "keyscale", "language", "timesignature")
    return {name: _field(name) for name in names}


def _result_field(container: Any, name: str, fallback: Any) -> Any:
    if isinstance(container, dict):
        return container.get(name, fallback)
    return getattr(container, name, fallback)


def _has_sung_words(lyrics: Any) -> bool:
    """Alignment scoring and LRC timestamps need words to align. The
    `[Instrumental]` placeholder has none, and scoring it cost a second
    forward pass per take -- about 11 s on the CPU-offloaded quality profile,
    measured on the 16 GB target (docs/COMPLETION_BENCHMARK.md)."""
    text = str(lyrics or "").strip()
    return bool(text) and not creation.is_instrumental_placeholder(text)


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
    if not _has_sung_words(lyrics):
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
    if not _has_sung_words(lyrics):
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
