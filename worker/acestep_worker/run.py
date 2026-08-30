"""`run_job`: one Lyre job -> one ACE-Step generation -> one take on disk.

This is the entry point `server.jobs.runner` calls for every generate, cover,
repaint, extract, lego, and complete.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worker.acestep_worker.api import _api_call, _import_acestep
from worker.acestep_worker.errors import WorkerUnavailable
from worker.acestep_worker.loading import _ensure_loaded, _ensure_lora_adapter
from worker.acestep_worker.results import (
    _is_simple_mode,
    _lyric_timestamps,
    _plan_from_query,
    _quality_score,
    _repaint_meta,
    _result_field,
)
from worker.acestep_worker.settings import (
    _LOCK,
    GENERATION_GUIDANCE_SCALE,
    GENERATION_STEPS,
    LORA_BASE_DIT_PROFILE,
    TASK_TYPE_BY_ACTION,
    TRACK_INSTRUCTION_ACTIONS,
)


def run_job(
    job: dict[str, Any],
    plan: dict[str, Any],
    take_id: str,
    take_dir: Path,
    on_dit_loaded: Callable[[str], None] | None = None,
) -> tuple[dict, dict | None, str | None]:
    """Run one real ACE-Step job. Returns `(take_meta, plan_patch, lrc_text)`;
    `plan_patch` is a delta of the LM-filled fields to persist when this was
    a simple-mode generation, else None (SPEC.md sec 7.2: "Persist the
    filled plan."). `server.jobs` merges it onto the plan that's current on
    disk when the job finishes, not the stale snapshot passed in as `plan`.
    `lrc_text` is standard `[mm:ss.xx]line`-formatted LRC text when ACE-Step
    could produce lyric timestamps for this take (SPEC.md sec 12 Phase 4;
    see `_lyric_timestamps`), else None. This module only returns it, same
    as `plan_patch` -- `server.jobs` is what actually persists it, via
    `server.storage.write_take_lrc`, to `takes/<take_id>/lyrics.lrc`, only
    when not None.

    `job["src_audio"]` must already be a resolved, jailed filesystem path (or
    None for `generate`) -- `server.jobs` resolves `source_take_id` /
    `upload_path` before calling this, so the worker never has to reach back
    into project storage itself.

    `on_dit_loaded`, if given, is called with the now-loaded dit_profile
    immediately after `_ensure_loaded` returns -- i.e. once any base-model
    swap has actually finished, but before the (potentially long-running)
    generation call below starts. This lets a caller publish worker_status
    right away instead of only after the whole job completes (SPEC.md sec
    4.3: a client polling worker status should see "loading" only for the
    swap itself, not for the inference that follows it).

    `job["lora_id"]`/`job["lora_adapter_path"]`, if set, request a trained
    style-pack LoRA be applied for this generation (SPEC.md sec 4.4 "LoRA
    train / load") -- see `_ensure_lora_adapter` for the real API this
    wraps. `server.jobs` resolves and validates `lora_id` and the jailed
    `lora_adapter_path` before the job ever reaches this worker, same
    division of labor as `job["src_audio"]` above.

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable or its API no
    longer matches this adapter; `server.jobs` catches that and marks the
    job `error` without crashing the HTTP process.
    """
    _, GenerationParams, GenerationConfig, _, generate_music, create_sample = _import_acestep()
    take_dir.mkdir(parents=True, exist_ok=True)

    simple_mode = _is_simple_mode(job, plan)

    with _LOCK:
        handler, lm, dit_profile = _ensure_loaded(job["dit_profile"])
        if on_dit_loaded is not None:
            on_dit_loaded(dit_profile)

        lora_id = job.get("lora_id")
        lora_adapter_path = job.get("lora_adapter_path")
        if lora_adapter_path is not None and dit_profile != LORA_BASE_DIT_PROFILE:
            # Belt and suspenders: server.jobs._resolve_dit_profile should
            # already have coerced/rejected this at enqueue time (SPEC.md
            # sec 4.4), but a LoRA's weight deltas are only valid against
            # the exact base checkpoint it was trained on -- refuse to apply
            # one against any other profile rather than silently loading it
            # onto the wrong base model.
            raise WorkerUnavailable(
                f"job requested lora_id '{lora_id}' but resolved dit_profile is "
                f"'{dit_profile}', not '{LORA_BASE_DIT_PROFILE}' -- a LoRA's weights are "
                "only valid against the base checkpoint it was trained on (SPEC.md sec 4.4)"
            )
        _ensure_lora_adapter(handler, lora_id, lora_adapter_path)

        effective_plan = _plan_from_query(create_sample, lm, plan) if simple_mode else plan
        # Null/omitted bpm, key, or duration: let ACE-Step's CoT fill them in
        # (SPEC.md sec 7.2, use_cot_metas).
        use_cot_metas = simple_mode or any(
            effective_plan.get(k) is None for k in ("bpm", "keyscale", "duration_sec")
        )

        # SPEC.md sec 7.2/9.2: Simple mode always runs with thinking=true (the
        # LM fills the whole plan from `query`). Custom mode leaves thinking
        # off by default -- unlike simple mode, the caption here is
        # human-written, so ACE-Step's LM must not silently rewrite it unless
        # the user opted in via the plan's `caption_rewrite` checkbox. Gated
        # on `job["action"] == "generate"` (equivalently: whenever
        # `simple_mode` could be true) rather than plan.get(...) alone, so a
        # `caption_rewrite: true` left over from an earlier Custom generate
        # can never leak into a cover/repaint/extract/lego/complete job on
        # the same plan -- those actions' thinking must stay exactly as
        # before this field existed (SPEC.md sec 4.2: "thinking=false is
        # allowed for cover/repaint/extract... upstream ignores LM for those
        # anyway").
        thinking = simple_mode or (
            job["action"] == "generate" and bool(plan.get("caption_rewrite", False))
        )

        # SPEC.md sec 7.3: "-1 from the user means worker picks and records
        # it" -- any other value is a fixed seed the user expects to be able
        # to reproduce. Passing `seed` on GenerationParams is not enough by
        # itself: ACE-Step's GenerationConfig.use_random_seed defaults True
        # upstream and overrides it, so a fixed seed must also flip that off
        # or a "regenerate with this seed" request can silently come back
        # different every time.
        seed = job.get("seed", -1)
        use_random_seed = seed == -1

        # plan.json's `negative` (negative prompts) has no confirmed
        # equivalent in the installed ACE-Step's GenerationParams -- the
        # previous `negative_tags` kwarg didn't exist and raised TypeError
        # on every real call. Not sent until upstream confirms a field name
        # (SPEC.md sec 13: adapt as the real API is confirmed).
        params = _api_call(
            "GenerationParams",
            GenerationParams,
            task_type=TASK_TYPE_BY_ACTION[job["action"]],
            caption=effective_plan.get("caption") or "",
            lyrics=effective_plan.get("lyrics") or "",
            bpm=effective_plan.get("bpm"),
            keyscale=effective_plan.get("keyscale") or "",
            duration=effective_plan.get("duration_sec"),
            instrumental=effective_plan.get("instrumental", False),
            vocal_language=effective_plan.get("vocal_language") or "unknown",
            # ACE-Step calls .strip() on timesignature; None crashes inside
            # generate_music (AttributeError), so never forward a missing
            # plan field as None -- empty string is the upstream default.
            timesignature=effective_plan.get("timesignature") or "",
            thinking=thinking,
            use_cot_metas=use_cot_metas,
            seed=seed,
            src_audio=job.get("src_audio"),
            audio_cover_strength=(
                1.0 if job.get("audio_cover_strength") is None else job.get("audio_cover_strength")
            ),
            repainting_start=job.get("repainting_start")
            if job.get("repainting_start") is not None
            else 0.0,
            repainting_end=job.get("repainting_end")
            if job.get("repainting_end") is not None
            else -1,
            # extract/lego/complete's track selection goes through the
            # task-specific `instruction` field, not a `track_name` kwarg
            # (GenerationParams has no such field -- passing it raised
            # TypeError on every real call, converted to WorkerUnavailable).
            instruction=(
                job.get("track_name")
                if job["action"] in TRACK_INSTRUCTION_ACTIONS
                else "Fill the audio semantic mask based on the given conditions:"
            ),
            inference_steps=GENERATION_STEPS[dit_profile],
            guidance_scale=GENERATION_GUIDANCE_SCALE[dit_profile],
            # SPEC.md sec 4.3: loudness normalization is "default on, no
            # extra mastering chain in v1" -- requested explicitly rather
            # than relying on ACE-Step's own default staying True upstream.
            # Lives on GenerationParams, not GenerationConfig.
            enable_normalization=True,
        )
        config = _api_call(
            "GenerationConfig",
            GenerationConfig,
            batch_size=job.get("batch_size", 1),
            # ACE-Step defaults to FLAC; request WAV explicitly so the
            # archive we rename to mix.wav below actually is one (SPEC.md
            # sec 7: mix.wav is "preferred archive").
            audio_format="wav",
            use_random_seed=use_random_seed,
        )
        result = _api_call(
            "generate_music",
            generate_music,
            dit_handler=handler,
            llm_handler=lm,
            params=params,
            config=config,
            save_dir=str(take_dir),
        )

    success = getattr(result, "success", True)
    audios = getattr(result, "audios", None) or []
    if not success or not audios:
        detail = getattr(result, "error", None) or getattr(result, "message", None)
        raise RuntimeError(f"ACE-Step generation failed: {detail or 'no audio produced'}")

    audio = audios[0]
    src_path_raw = _result_field(audio, "path", None) or _result_field(audio, "audio_path", None)
    if not src_path_raw:
        raise RuntimeError("ACE-Step reported success but returned no audio path")
    src_path = Path(src_path_raw).resolve()

    # The returned path is untrusted: an upstream bug, API drift, or a
    # compromised acestep install could point it anywhere on disk. Require
    # it stay inside the take directory we handed to save_dir= above
    # *before* even checking existence -- every generated-audio filesystem
    # operation must stay jailed under projects/ or output/ (SPEC.md sec
    # 8.1 / 11), and shutil.move below must never touch an arbitrary local
    # file.
    take_dir_resolved = take_dir.resolve()
    try:
        src_path.relative_to(take_dir_resolved)
    except ValueError:
        raise RuntimeError(
            f"ACE-Step returned an audio path outside its allocated take directory "
            f"('{src_path}' is not under '{take_dir_resolved}'); refusing to touch it."
        ) from None

    if not src_path.exists():
        raise RuntimeError(f"ACE-Step reported audio at '{src_path}' but the file does not exist")

    # We requested audio_format="wav" above; storage.take_audio_path only
    # recognizes mix.wav / mix.mp3 (SPEC.md sec 7). Verify ACE-Step actually
    # returned one of those instead of blindly renaming whatever it wrote --
    # a FLAC file (ACE-Step's default) renamed to .wav is not a WAV file and
    # won't play/download reliably as one.
    suffix = src_path.suffix.lower()
    if suffix not in (".wav", ".mp3"):
        raise RuntimeError(
            f"ACE-Step returned audio in an unexpected format ('{suffix}', requested 'wav') "
            f"at '{src_path}'. Update worker/acestep_worker.py's GenerationConfig audio_format "
            "handling to match the installed acestep version."
        )
    dest_path = take_dir / ("mix.mp3" if suffix == ".mp3" else "mix.wav")
    if src_path != dest_path:
        shutil.move(str(src_path), str(dest_path))

    extra = getattr(result, "extra_outputs", None) or {}

    def _extra(name: str, fallback: Any) -> Any:
        if isinstance(extra, dict):
            return extra.get(name, fallback)
        return getattr(extra, name, fallback)

    caption = _extra("caption", effective_plan.get("caption", ""))
    lyrics = _extra("lyrics", effective_plan.get("lyrics", ""))
    bpm = _extra("bpm", effective_plan.get("bpm"))
    keyscale = _extra("keyscale", effective_plan.get("keyscale"))
    duration_sec = _extra("duration", effective_plan.get("duration_sec"))
    vocal_language = _extra("vocal_language", effective_plan.get("vocal_language"))
    timesignature = _extra("timesignature", effective_plan.get("timesignature"))
    score = _quality_score(
        handler,
        extra_lookup=_extra,
        lyrics=lyrics,
        vocal_language=vocal_language,
        dit_profile=dit_profile,
    )
    lrc_text = _lyric_timestamps(
        handler,
        extra_lookup=_extra,
        lyrics=lyrics,
        vocal_language=vocal_language,
        dit_profile=dit_profile,
        duration_sec=duration_sec,
    )

    # The actual seed used lives nested under audio["params"]["seed"] --
    # not top-level on the audio record -- with progressively looser
    # fallbacks for other upstream versions. Falling through to the
    # requested seed would silently record -1 for random generation instead
    # of the seed ACE-Step actually used (SPEC.md sec 7.3).
    audio_params = _result_field(audio, "params", None)
    seed = _result_field(audio_params, "seed", None)
    if seed is None:
        seed = _result_field(audio, "seed", None)
    if seed is None:
        seed = _extra("seed", None)
    if seed is None:
        seed = job.get("seed", -1)

    plan_patch = None
    if simple_mode:
        # The LM filled caption/lyrics/metas from `query` -- persist them
        # onto plan.json instead of discarding them (SPEC.md sec 7.2),
        # including duration and any other metadata ACE-Step filled in.
        #
        # This is a patch, not a full plan: `plan` here was loaded before
        # this (possibly long-running) generation started, so spreading it
        # into the patch would let a stale snapshot clobber any edits saved
        # while the job was running. server.jobs merges this onto whatever
        # plan is current on disk when the job finishes instead.
        plan_patch = {
            "caption": caption,
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": keyscale,
            "duration_sec": duration_sec,
            "vocal_language": vocal_language,
            "timesignature": timesignature,
        }

    meta = {
        "id": take_id,
        "parent_take_id": job.get("source_take_id"),
        "task_type": TASK_TYPE_BY_ACTION[job["action"]],
        "dit_profile": dit_profile,
        "seed": seed,
        "duration_sec": duration_sec,
        "caption": caption,
        "lyrics": lyrics,
        "bpm": bpm,
        "keyscale": keyscale,
        "created_at": datetime.now(UTC).isoformat(),
        # ACE-Step 1.5's DiT Lyrics Alignment Score (see `_quality_score`
        # above for exactly what was checked upstream and why this is a
        # second handler call rather than a `GenerationResult` field). None
        # when lyrics are empty, the decoder-attention tensors weren't in
        # `extra_outputs` (e.g. ACE-Step's save-memory mode), or scoring
        # itself failed -- a missing score must never fail an otherwise-
        # successful take.
        "score": score,
        # SPEC.md sec 7 `lyrics.lrc ... optional, phase 4`: True only when
        # `_lyric_timestamps` actually produced LRC text -- the UI must not
        # guess a lyrics.lrc exists from this alone and hit a 404 (see
        # `_lyric_timestamps` for exactly when this is None).
        "has_lrc": lrc_text is not None,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
        # SPEC.md sec 4.4 "LoRA train / load": which style-pack lora (if
        # any) was applied to this take, for the same reason track_name is
        # recorded here -- an auditable record on the take itself, not just
        # in the job row.
        "lora_id": job.get("lora_id"),
        "favorite": False,
        "notes": "",
    }
    return meta, plan_patch, lrc_text
