"""Signature-level test for worker/acestep_worker.py's ACE-Step 1.5 adapter.

`tests/test_phase1_api.py` only ever exercises `worker.mock_worker`, so a
call-contract mismatch against the real ACE-Step 1.5 Python API (wrong
method name, wrong argument, a field on the wrong class) would go
undetected. This module installs fake `acestep.*` modules with the
signatures ACE-Step 1.5 actually exposes -- `AceStepHandler` loaded via
`initialize_service(project_root=..., config_path=<checkpoint name>,
device=...)`, which the fake below resolves the same way ACE-Step itself
does (`<project_root>/checkpoints/<config_path>`) so a test can assert the
*real* weights directory is found, not just that some locally-assumed kwarg
value was passed through unchanged, `LLMHandler` loaded via a *different*
method with *different*
argument names, `initialize(checkpoint_dir=..., lm_model_path=...,
backend="pt", device=...)`, returning `(status_message, success)` (a falsy
`success` must be treated as a failed load, not cached as ready), module-
level `create_sample` whose result carries the language under `language`
(mapped onto Bard's own `vocal_language` plan field), `inference_steps`/
`guidance_scale`/`vocal_language`/`timesignature`/`instruction` on
`GenerationParams` (not `GenerationConfig`, and *no* `negative_tags` or
`track_name` field -- those don't exist upstream and previously raised
TypeError on every real call), `generate_music(dit_handler=...)` (not
`handler=`), audio results as dicts with the actual seed nested at
`audio["params"]["seed"]`, FLAC-by-default output unless
`GenerationConfig(audio_format="wav")` is requested, and loudness
normalization requested explicitly (`enable_normalization=True`, SPEC.md
sec 4.3: "default on") rather than left to whatever ACE-Step's own default
is -- then runs `run_job` against them, so `worker/acestep_worker.py`'s
exact calling convention is verified without requiring CUDA or the real
acestep package. See SPEC.md sec 13.
"""

from __future__ import annotations

import re
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from worker import acestep_worker


@pytest.fixture(autouse=True)
def _reset_worker_state():
    acestep_worker._STATE.update({"dit_profile": None, "handler": None, "lm": None})
    yield
    acestep_worker._STATE.update({"dit_profile": None, "handler": None, "lm": None})


def _write_tiny_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 400)


class FakeGenerationParams:
    """Mirrors ACE-Step 1.5's real signature: every per-request field,
    including inference_steps/guidance_scale, lives here -- not on
    GenerationConfig. There is no negative_tags or track_name field (the
    real GenerationParams has neither -- passing them raised TypeError on
    every real call); track selection for extract/lego/complete goes
    through `instruction` instead. A missing/extra/misplaced kwarg raises
    TypeError, same as it would against the real class."""

    def __init__(
        self,
        *,
        task_type: str,
        caption: str,
        lyrics: str,
        bpm: Any,
        keyscale: Any,
        duration: Any,
        instrumental: bool,
        vocal_language: Any,
        timesignature: Any,
        thinking: bool,
        use_cot_metas: bool,
        seed: int,
        src_audio: Any,
        audio_cover_strength: Any,
        repainting_start: Any,
        repainting_end: Any,
        instruction: Any,
        inference_steps: int,
        guidance_scale: float,
    ) -> None:
        self.task_type = task_type
        self.caption = caption
        self.lyrics = lyrics
        self.bpm = bpm
        self.keyscale = keyscale
        self.duration = duration
        self.instrumental = instrumental
        self.vocal_language = vocal_language
        self.timesignature = timesignature
        self.thinking = thinking
        self.use_cot_metas = use_cot_metas
        self.seed = seed
        self.src_audio = src_audio
        self.audio_cover_strength = audio_cover_strength
        self.repainting_start = repainting_start
        self.repainting_end = repainting_end
        self.instruction = instruction
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale


class FakeGenerationConfig:
    """batch_size/audio_format/use_random_seed/enable_normalization --
    inference_steps/guidance_scale belong on GenerationParams instead (this
    is exactly what the reviewer flagged). use_random_seed defaults True
    upstream and must be explicitly False for a fixed (non--1) seed to
    actually be honored -- also reviewer-flagged. enable_normalization must
    be requested explicitly True (SPEC.md sec 4.3: "default on") rather than
    left unset."""

    def __init__(
        self,
        *,
        batch_size: int,
        audio_format: str,
        use_random_seed: bool,
        enable_normalization: bool,
    ) -> None:
        self.batch_size = batch_size
        self.audio_format = audio_format
        self.use_random_seed = use_random_seed
        self.enable_normalization = enable_normalization


class FakeResult:
    def __init__(self, audios: list[dict], extra_outputs: dict[str, Any]) -> None:
        self.success = True
        self.audios = audios
        self.extra_outputs = extra_outputs


def _install_fake_acestep(
    monkeypatch: pytest.MonkeyPatch,
    log: list[tuple],
    handler_cls: type | None = None,
    returned_audio_suffix: str = ".wav",
    lm_init_result: tuple[str, bool] = ("lm ready", True),
    handler_init_result: tuple[str, bool] = ("dit ready", True),
    audio_path_override: Path | None = None,
    create_sample_result: dict | None = None,
    lyric_score_tensors: dict[str, Any] | None = None,
    lyric_score_result: dict[str, Any] | None = None,
    lyric_timestamp_result: dict[str, Any] | None = None,
) -> None:
    """Register fake `acestep.*` modules so `worker.acestep_worker`'s lazy
    `from acestep... import ...` statements resolve to them instead of the
    real (uninstalled) package. `returned_audio_suffix` simulates what
    `generate_music` hands back regardless of the requested
    `audio_format` -- default `.wav` (the request honored); pass `.flac`
    to simulate ACE-Step ignoring it and still defaulting to FLAC.
    `lm_init_result`/`handler_init_result` simulate LLMHandler.initialize /
    AceStepHandler.initialize_service's `(status_message, success)` return;
    pass a falsy `success` to simulate a failed load that must not be
    cached as ready. `audio_path_override` simulates ACE-Step reporting an
    audio file at an arbitrary path instead of writing into `save_dir`.
    `create_sample_result` overrides module-level `create_sample`'s default
    successful return -- pass e.g. `{"success": False, "error": "..."}` to
    simulate a documented planning failure. `lyric_score_tensors`, when
    given, is merged into `generate_music`'s `extra_outputs` -- mirrors
    ACE-Step actually populating the decoder-attention tensors
    (`pred_latents`/`encoder_hidden_states`/`encoder_attention_mask`/
    `context_latents`/`lyric_token_idss`) `AceStepHandler.get_lyric_score`
    needs; omitted (the default) matches ACE-Step's save-memory mode / no
    tensors captured, so the adapter must not attempt scoring.
    `lyric_score_result` overrides the fake handler's `get_lyric_score`
    return; defaults to a canned success payload. `lyric_timestamp_result`
    is the same override for `get_lyric_timestamp` (SPEC.md sec 12 Phase 4;
    also gated on `lyric_score_tensors`, since ACE-Step's real
    `LyricTimestampMixin.get_lyric_timestamp` consumes the exact same five
    decoder-attention tensors as `LyricScoreMixin.get_lyric_score`);
    defaults to a canned success payload with `[mm:ss.xx]`-formatted
    `lrc_text`."""

    class DefaultFakeAceStepHandler:
        def __init__(self) -> None:
            self.config_path: str | None = None
            self.offload_to_cpu = False

        def get_lyric_score(self, **kwargs: Any) -> dict[str, Any]:
            # Real signature: pred_latent, encoder_hidden_states,
            # encoder_attention_mask, context_latents, lyric_token_ids,
            # vocal_language, inference_steps, seed -- returns a dict with
            # lm_score/dit_score/success/error (worker/acestep_worker.py's
            # `_quality_score` uses dit_score, upstream's Tutorial.md
            # "favorite" metric).
            log.append(("handler.get_lyric_score", kwargs))
            if lyric_score_result is not None:
                return lyric_score_result
            return {"lm_score": 0.5, "dit_score": 0.8123, "success": True, "error": None}

        def get_lyric_timestamp(self, **kwargs: Any) -> dict[str, Any]:
            # Real signature: pred_latent, encoder_hidden_states,
            # encoder_attention_mask, context_latents, lyric_token_ids,
            # total_duration_seconds, vocal_language, inference_steps, seed
            # -- returns a dict with lrc_text (already `[mm:ss.xx]line`
            # formatted)/sentence_timestamps/token_timestamps/success/error
            # (worker/acestep_worker.py's `_lyric_timestamps` uses lrc_text
            # as-is, per acestep/core/scoring/dit_alignment.py's
            # MusicStampsAligner.get_timestamps_and_lrc).
            log.append(("handler.get_lyric_timestamp", kwargs))
            if lyric_timestamp_result is not None:
                return lyric_timestamp_result
            return {
                "lrc_text": "[00:00.00]We were born to run\n[00:02.50]Down the neon highway",
                "sentence_timestamps": [
                    {"text": "We were born to run", "start": 0.0, "end": 2.5},
                    {"text": "Down the neon highway", "start": 2.5, "end": 5.0},
                ],
                "token_timestamps": [],
                "success": True,
                "error": None,
            }

        def initialize_service(
            self, *, project_root: str, config_path: str, device: str, offload_to_cpu: bool = False
        ) -> tuple[str, bool]:
            # Mirrors ACE-Step 1.5's real resolution: the DiT checkpoint
            # lives at <project_root>/checkpoints/<config_path>, not at
            # <project_root>/<config_path> and not by treating project_root
            # itself as the checkpoint directory. Exposing the resolved path
            # lets tests assert the real weights location is found instead
            # of merely accepting whatever project_root value was passed
            # (exactly what the reviewer flagged the previous version of
            # this test as unable to catch).
            resolved_checkpoint_dir = Path(project_root) / "checkpoints" / config_path
            log.append(
                (
                    "handler.initialize_service",
                    {
                        "project_root": project_root,
                        "config_path": config_path,
                        "device": device,
                        "offload_to_cpu": offload_to_cpu,
                        "resolved_checkpoint_dir": resolved_checkpoint_dir,
                    },
                )
            )
            self.project_root = project_root
            self.config_path = config_path
            self.device = device
            self.offload_to_cpu = offload_to_cpu
            return handler_init_result

    class FakeLLMHandler:
        def __init__(self) -> None:
            self.checkpoint_dir: str | None = None
            self.lm_model_path: str | None = None

        def initialize(
            self, *, checkpoint_dir: str, lm_model_path: str, backend: str, device: str
        ) -> tuple[str, bool]:
            log.append(
                (
                    "lm.initialize",
                    {
                        "checkpoint_dir": checkpoint_dir,
                        "lm_model_path": lm_model_path,
                        "backend": backend,
                        "device": device,
                    },
                )
            )
            self.checkpoint_dir = checkpoint_dir
            self.lm_model_path = lm_model_path
            return lm_init_result

    def create_sample(
        *, lm_handler: Any, query: str, instrumental: bool, vocal_language: Any
    ) -> dict:
        log.append(("create_sample", lm_handler, query, instrumental, vocal_language))
        if create_sample_result is not None:
            return create_sample_result
        return {
            "caption": f"auto: {query}",
            "lyrics": "[Instrumental]" if instrumental else f"[Verse]\n{query}",
            "bpm": 120,
            "keyscale": "C Major",
            "duration": 30,
            # ACE-Step's own field name is "language", not "vocal_language"
            # (Bard's plan.json field) -- distinct from extra_outputs'
            # "en" below so the mapping in _plan_from_query is unambiguous.
            "language": "ja",
        }

    def generate_music(
        *, dit_handler: Any, lm_handler: Any, params: Any, config: Any, save_dir: str
    ) -> FakeResult:
        log.append(("generate_music", dit_handler, lm_handler, params, config, save_dir))
        # ACE-Step writes its own output filename -- not mix.wav -- so the
        # adapter has to find and rename it (SPEC.md sec 7.3).
        if audio_path_override is not None:
            out_path = audio_path_override
        else:
            out_path = Path(save_dir) / f"output_0{returned_audio_suffix}"
        _write_tiny_wav(out_path)  # fine as fixture bytes regardless of extension
        # Real ACE-Step audio entries are dicts, and the actual seed used is
        # nested under "params", not top-level -- exactly what the reviewer
        # flagged as misread.
        audio = {"path": str(out_path), "params": {"seed": 999}}
        extra_outputs = {
            "caption": params.caption,
            "lyrics": params.lyrics,
            "bpm": params.bpm,
            "keyscale": params.keyscale,
            "duration": params.duration,
            "vocal_language": "en",
            "timesignature": "4/4",
        }
        if lyric_score_tensors is not None:
            extra_outputs.update(lyric_score_tensors)
        return FakeResult(audios=[audio], extra_outputs=extra_outputs)

    acestep_pkg = types.ModuleType("acestep")
    handler_mod = types.ModuleType("acestep.handler")
    inference_mod = types.ModuleType("acestep.inference")
    llm_mod = types.ModuleType("acestep.llm_inference")

    handler_mod.AceStepHandler = handler_cls or DefaultFakeAceStepHandler
    llm_mod.LLMHandler = FakeLLMHandler
    inference_mod.GenerationParams = FakeGenerationParams
    inference_mod.GenerationConfig = FakeGenerationConfig
    inference_mod.generate_music = generate_music
    inference_mod.create_sample = create_sample

    monkeypatch.setitem(sys.modules, "acestep", acestep_pkg)
    monkeypatch.setitem(sys.modules, "acestep.handler", handler_mod)
    monkeypatch.setitem(sys.modules, "acestep.inference", inference_mod)
    monkeypatch.setitem(sys.modules, "acestep.llm_inference", llm_mod)


def test_run_job_matches_installed_api_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    plan = {
        "query": "",
        "caption": "orchestral swell, cinematic strings",
        "lyrics": "[Instrumental]",
        "negative": ["distorted", "lo-fi"],
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 45,
        "vocal_language": "fr",
        "timesignature": "3/4",
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, plan_patch, lrc_text = acestep_worker.run_job(
        job=job, plan=plan, take_id="t1", take_dir=tmp_path / "take1"
    )

    assert plan_patch is None  # not simple mode: nothing to persist onto plan.json
    assert meta["task_type"] == "text2music"
    assert meta["dit_profile"] == "iterate"
    assert meta["caption"] == plan["caption"]
    # the actual seed used, read from audio["params"]["seed"] -- not the -1
    # request, and not silently left as -1 (SPEC.md sec 7.3)
    assert meta["seed"] == 999
    assert (tmp_path / "take1" / "mix.wav").exists()  # renamed from output_0.wav
    # No decoder-attention tensors in extra_outputs here (the default fake
    # matches ACE-Step's save-memory mode / no tensors captured) -- scoring
    # must not be attempted, not fabricated (SPEC.md sec 12 Phase 4).
    assert meta["score"] is None
    assert not any(e[0] == "handler.get_lyric_score" for e in log)
    # Same reasoning applies to lyric timestamps (SPEC.md sec 12 Phase 4:
    # "LRC if upstream provides timestamps") -- no tensors, no attempt.
    assert meta["has_lrc"] is False
    assert lrc_text is None
    assert not any(e[0] == "handler.get_lyric_timestamp" for e in log)
    # SPEC.md sec 12 Phase 6: every newly-created take gets these fields so
    # older takes (written before this migration) are the only ones a
    # reader ever needs `.get("favorite", False)` / `.get("notes", "")` for.
    assert meta["favorite"] is False
    assert meta["notes"] == ""

    handler_init = next(e for e in log if e[0] == "handler.initialize_service")
    kwargs = handler_init[1]
    # The real assertion that matters (reviewer-flagged): ACE-Step resolves
    # the checkpoint at <project_root>/checkpoints/<config_path>, so this
    # must land on Bard's actual weights directory -- not
    # checkpoints/checkpoints/acestep-v15-turbo, which is what passing
    # CHECKPOINTS_ROOT itself as project_root used to produce.
    assert kwargs["resolved_checkpoint_dir"] == acestep_worker.CHECKPOINTS_ROOT / "acestep-v15-turbo"
    assert kwargs["project_root"] == str(acestep_worker.CHECKPOINTS_ROOT.parent)
    assert kwargs["config_path"] == "acestep-v15-turbo"
    assert kwargs["device"] == acestep_worker.DEVICE
    assert kwargs["offload_to_cpu"] is False  # iterate does not need offload_to_cpu

    # LLMHandler is loaded through a *different* method with *different*
    # argument names than the DiT handler (the bug the reviewer flagged),
    # including backend="pt" (SPEC.md sec 4.2 -- ACE-Step otherwise
    # defaults to "vllm") and device.
    lm_init = next(e for e in log if e[0] == "lm.initialize")
    assert lm_init[1]["checkpoint_dir"] == str(acestep_worker.CHECKPOINTS_ROOT)
    assert lm_init[1]["lm_model_path"] == acestep_worker.DEFAULT_LM
    assert lm_init[1]["backend"] == "pt"
    assert lm_init[1]["device"] == acestep_worker.DEVICE

    generate_call = next(e for e in log if e[0] == "generate_music")
    params, config = generate_call[3], generate_call[4]
    # SPEC.md sec 4.1: iterate = 8 steps, no CFG -- and these live on
    # GenerationParams, not GenerationConfig (the bug the reviewer flagged).
    assert params.inference_steps == 8
    assert params.guidance_scale == 1.0
    assert config.batch_size == 1
    assert not hasattr(config, "inference_steps")
    assert not hasattr(config, "guidance_scale")
    # ACE-Step defaults to FLAC; the adapter must request WAV explicitly
    # instead of silently relabeling whatever comes back as .wav.
    assert config.audio_format == "wav"
    # SPEC.md sec 4.3: loudness normalization is "default on" -- requested
    # explicitly rather than relying on whatever ACE-Step's own default is.
    assert config.enable_normalization is True
    # Custom-mode plan metadata (language, time signature) must actually
    # reach the renderer, not be dropped.
    assert params.vocal_language == "fr"
    assert params.timesignature == "3/4"
    # GenerationParams has no negative_tags/track_name field; plan.json's
    # "negative" is not forwarded (no confirmed upstream field), and
    # "generate" doesn't send an instruction (that's extract/lego/complete
    # only -- see test_track_name_maps_to_instruction_for_studio_ops).
    assert not hasattr(params, "negative_tags")
    assert params.instruction is None
    # SPEC.md sec 7.3: -1 means "worker picks" -- ACE-Step's own random-seed
    # path, not a fixed request.
    assert params.seed == -1
    assert config.use_random_seed is True


def test_dit_lyric_score_flows_into_take_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 12 Phase 4: "ACE-Step quality score on takes". There's no
    score field on GenerationResult/extra_outputs itself, but ACE-Step 1.5
    does expose a real scoring call -- `AceStepHandler.get_lyric_score` --
    that reuses the decoder-attention tensors `generate_music` puts in
    `extra_outputs` when lyrics are present. This fake mirrors that: once
    those tensors show up in `extra_outputs`, `run_job` must call
    `get_lyric_score` with them (plus vocal_language/inference_steps/seed)
    and round-trip its `dit_score` into the take's `score` field."""
    log: list[tuple] = []
    tensors = {
        "pred_latents": "fake-pred-latents",
        "encoder_hidden_states": "fake-encoder-hidden-states",
        "encoder_attention_mask": "fake-encoder-attention-mask",
        "context_latents": "fake-context-latents",
        "lyric_token_idss": "fake-lyric-token-ids",
    }
    _install_fake_acestep(monkeypatch, log, lyric_score_tensors=tensors)

    plan = {
        "query": "",
        "caption": "anthemic pop chorus",
        "lyrics": "[Verse]\nWe were born to run",
        "instrumental": False,
        "bpm": 120,
        "keyscale": "C Major",
        "duration_sec": 30,
        "vocal_language": "en",
        "timesignature": "4/4",
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, _, _ = acestep_worker.run_job(
        job=job, plan=plan, take_id="t-score", take_dir=tmp_path / "take-score"
    )

    assert meta["score"] == 0.8123  # dit_score from the fake's canned success payload

    score_call = next(e for e in log if e[0] == "handler.get_lyric_score")
    kwargs = score_call[1]
    assert kwargs["pred_latent"] == tensors["pred_latents"]
    assert kwargs["encoder_hidden_states"] == tensors["encoder_hidden_states"]
    assert kwargs["encoder_attention_mask"] == tensors["encoder_attention_mask"]
    assert kwargs["context_latents"] == tensors["context_latents"]
    # extra_outputs' real key is "lyric_token_idss" (plural), mapped onto
    # get_lyric_score's "lyric_token_ids" kwarg.
    assert kwargs["lyric_token_ids"] == tensors["lyric_token_idss"]
    assert kwargs["vocal_language"] == "en"
    # SPEC.md sec 4.1: iterate = 8 steps -- the same inference_steps this
    # take was actually generated with, not a hardcoded default.
    assert kwargs["inference_steps"] == 8
    assert kwargs["seed"] == 42


def test_dit_lyric_score_failure_falls_back_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scoring failure (e.g. attentions unavailable) must fall back to
    `None`, not fail the whole take -- scoring is best-effort, unlike the
    surrounding `_api_call`-wrapped calls that raise `WorkerUnavailable`."""
    log: list[tuple] = []
    tensors = {
        "pred_latents": "fake-pred-latents",
        "encoder_hidden_states": "fake-encoder-hidden-states",
        "encoder_attention_mask": "fake-encoder-attention-mask",
        "context_latents": "fake-context-latents",
        "lyric_token_idss": "fake-lyric-token-ids",
    }
    _install_fake_acestep(
        monkeypatch,
        log,
        lyric_score_tensors=tensors,
        lyric_score_result={"success": False, "error": "attentions unavailable"},
    )

    plan = {
        "query": "",
        "caption": "anthemic pop chorus",
        "lyrics": "[Verse]\nWe were born to run",
        "instrumental": False,
        "bpm": 120,
        "keyscale": "C Major",
        "duration_sec": 30,
        "vocal_language": "en",
        "timesignature": "4/4",
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, _, _ = acestep_worker.run_job(
        job=job, plan=plan, take_id="t-score-fail", take_dir=tmp_path / "take-score-fail"
    )

    assert meta["score"] is None
    assert any(e[0] == "handler.get_lyric_score" for e in log)


def test_lyric_timestamps_flow_into_take_meta_and_lrc_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 12 Phase 4: "LRC if upstream provides timestamps" -- it
    does. ACE-Step 1.5 exposes a real per-line timestamp call --
    `AceStepHandler.get_lyric_timestamp` -- that reuses the exact same
    decoder-attention tensors `generate_music` puts in `extra_outputs` as
    `get_lyric_score`. This fake mirrors that: once those tensors show up,
    `run_job` must call `get_lyric_timestamp` with them (plus
    total_duration_seconds/vocal_language/inference_steps/seed) and
    round-trip its already-`[mm:ss.xx]`-formatted `lrc_text` as the take's
    returned lrc text, with `has_lrc: True` on the take meta."""
    log: list[tuple] = []
    tensors = {
        "pred_latents": "fake-pred-latents",
        "encoder_hidden_states": "fake-encoder-hidden-states",
        "encoder_attention_mask": "fake-encoder-attention-mask",
        "context_latents": "fake-context-latents",
        "lyric_token_idss": "fake-lyric-token-ids",
    }
    _install_fake_acestep(monkeypatch, log, lyric_score_tensors=tensors)

    plan = {
        "query": "",
        "caption": "anthemic pop chorus",
        "lyrics": "[Verse]\nWe were born to run",
        "instrumental": False,
        "bpm": 120,
        "keyscale": "C Major",
        "duration_sec": 30,
        "vocal_language": "en",
        "timesignature": "4/4",
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, _, lrc_text = acestep_worker.run_job(
        job=job, plan=plan, take_id="t-lrc", take_dir=tmp_path / "take-lrc"
    )

    assert meta["has_lrc"] is True
    assert lrc_text is not None
    lines = [line for line in lrc_text.splitlines() if line.strip()]
    assert lines, "expected at least one timed lyric line"
    for line in lines:
        assert re.match(r"^\[\d{2}:\d{2}\.\d{2}\].+", line), line

    ts_call = next(e for e in log if e[0] == "handler.get_lyric_timestamp")
    kwargs = ts_call[1]
    assert kwargs["pred_latent"] == tensors["pred_latents"]
    assert kwargs["encoder_hidden_states"] == tensors["encoder_hidden_states"]
    assert kwargs["encoder_attention_mask"] == tensors["encoder_attention_mask"]
    assert kwargs["context_latents"] == tensors["context_latents"]
    assert kwargs["lyric_token_ids"] == tensors["lyric_token_idss"]
    assert kwargs["total_duration_seconds"] == 30
    assert kwargs["vocal_language"] == "en"
    # SPEC.md sec 4.1: iterate = 8 steps -- the same inference_steps this
    # take was actually generated with, not a hardcoded default.
    assert kwargs["inference_steps"] == 8
    assert kwargs["seed"] == 42


def test_lyric_timestamps_failure_falls_back_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timestamp-alignment failure must fall back to `None` (no
    `lyrics.lrc`), not fail the whole take -- best-effort, unlike the
    surrounding `_api_call`-wrapped calls that raise `WorkerUnavailable`."""
    log: list[tuple] = []
    tensors = {
        "pred_latents": "fake-pred-latents",
        "encoder_hidden_states": "fake-encoder-hidden-states",
        "encoder_attention_mask": "fake-encoder-attention-mask",
        "context_latents": "fake-context-latents",
        "lyric_token_idss": "fake-lyric-token-ids",
    }
    _install_fake_acestep(
        monkeypatch,
        log,
        lyric_score_tensors=tensors,
        lyric_timestamp_result={"success": False, "error": "alignment failed"},
    )

    plan = {
        "query": "",
        "caption": "anthemic pop chorus",
        "lyrics": "[Verse]\nWe were born to run",
        "instrumental": False,
        "bpm": 120,
        "keyscale": "C Major",
        "duration_sec": 30,
        "vocal_language": "en",
        "timesignature": "4/4",
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, _, lrc_text = acestep_worker.run_job(
        job=job, plan=plan, take_id="t-lrc-fail", take_dir=tmp_path / "take-lrc-fail"
    )

    assert lrc_text is None
    assert meta["has_lrc"] is False
    assert any(e[0] == "handler.get_lyric_timestamp" for e in log)


def test_checkpoints_project_root_matches_ace_step_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct unit coverage for _checkpoints_project_root() (reviewer-
    flagged): AceStepHandler.initialize_service resolves the checkpoint at
    <project_root>/checkpoints/<config_path>, so project_root must be
    CHECKPOINTS_ROOT's parent for that to land on CHECKPOINTS_ROOT itself --
    and a BARD_CHECKPOINTS_DIR that isn't literally named 'checkpoints' can
    never satisfy that upstream convention, so it must fail clearly instead
    of silently resolving to the wrong directory."""
    monkeypatch.setattr(acestep_worker, "CHECKPOINTS_ROOT", Path("some/where/checkpoints"))
    project_root = acestep_worker._checkpoints_project_root()
    assert project_root == Path("some/where")
    assert project_root / "checkpoints" / "acestep-v15-turbo" == acestep_worker.CHECKPOINTS_ROOT / (
        "acestep-v15-turbo"
    )

    monkeypatch.setattr(acestep_worker, "CHECKPOINTS_ROOT", Path("some/where/weights"))
    with pytest.raises(acestep_worker.WorkerUnavailable, match="checkpoints"):
        acestep_worker._checkpoints_project_root()


def test_fixed_seed_disables_use_random_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive job seed must actually be reproducible: GenerationParams.seed
    alone is not enough upstream -- GenerationConfig.use_random_seed defaults
    True and overrides it unless explicitly turned off (reviewer-flagged:
    'consequently regenerating with a recorded seed can produce different
    audio'). -1 keeps ACE-Step's own random path."""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    plan = {
        "query": "",
        "caption": "orchestral swell",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 45,
        "vocal_language": "en",
        "timesignature": "3/4",
    }

    acestep_worker.run_job(
        job={"action": "generate", "dit_profile": "iterate", "seed": 12345, "src_audio": None},
        plan=plan,
        take_id="tfixed",
        take_dir=tmp_path / "take_fixed",
    )
    generate_call = next(e for e in log if e[0] == "generate_music")
    params, config = generate_call[3], generate_call[4]
    assert params.seed == 12345
    assert config.use_random_seed is False

    log.clear()
    acestep_worker.run_job(
        job={"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None},
        plan=plan,
        take_id="trandom",
        take_dir=tmp_path / "take_random",
    )
    generate_call = next(e for e in log if e[0] == "generate_music")
    params, config = generate_call[3], generate_call[4]
    assert params.seed == -1
    assert config.use_random_seed is True


def test_simple_mode_uses_module_level_create_sample_and_persists_full_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    plan = {
        "query": "dreamy synthwave drive",
        "caption": "",
        "lyrics": "",
        "instrumental": False,
        "bpm": None,
        "keyscale": None,
        "duration_sec": None,
        "vocal_language": "es",
        "timesignature": None,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, plan_patch, _ = acestep_worker.run_job(
        job=job, plan=plan, take_id="t2", take_dir=tmp_path / "take2"
    )

    create_call = next(e for e in log if e[0] == "create_sample")
    assert create_call[2] == "dreamy synthwave drive"  # query passed through
    assert create_call[3] is False  # instrumental
    # the plan's vocal_language must constrain create_sample's own lyrics
    # generation, not just get applied after the fact (reviewer-flagged).
    assert create_call[4] == "es"

    assert plan_patch is not None
    assert "dreamy synthwave drive" in plan_patch["caption"]
    assert meta["caption"] == plan_patch["caption"]
    # SPEC.md sec 7.2: persist everything the LM/ACE-Step filled in, not
    # just caption/lyrics/bpm/keyscale -- duration and other plan metadata
    # (language, time signature) must round-trip too.
    assert plan_patch["duration_sec"] == 30
    assert plan_patch["vocal_language"] == "en"
    assert plan_patch["timesignature"] == "4/4"
    assert meta["duration_sec"] == 30

    generate_call = next(e for e in log if e[0] == "generate_music")
    params = generate_call[3]
    assert params.thinking is True
    assert params.use_cot_metas is True
    # create_sample's result carries the language under "language", not
    # "vocal_language" (Bard's own plan.json field name) -- confirms
    # _plan_from_query maps it correctly before generation even runs.
    assert params.vocal_language == "ja"


def test_create_sample_failure_fails_job_instead_of_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If create_sample reports success=False (e.g. ACE-Step's planner
    itself failed), the adapter must fail the job with that detail instead
    of proceeding to generate_music from an empty/fallback caption and
    lyrics -- mirrors how generate_music's own GenerationResult.success is
    already checked below."""
    log: list[tuple] = []
    _install_fake_acestep(
        monkeypatch,
        log,
        create_sample_result={"success": False, "error": "planner overloaded"},
    )

    plan = {
        "query": "dreamy synthwave drive",
        "caption": "",
        "lyrics": "",
        "instrumental": False,
        "bpm": None,
        "keyscale": None,
        "duration_sec": None,
        "vocal_language": None,
        "timesignature": None,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    with pytest.raises(RuntimeError, match="planner overloaded"):
        acestep_worker.run_job(job=job, plan=plan, take_id="t2b", take_dir=tmp_path / "take2b")

    # Must fail before ever reaching generate_music -- no audio should have
    # been produced from the empty/fallback plan.
    assert not any(e[0] == "generate_music" for e in log)


def test_quality_profile_requests_cpu_offload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    ok, reason = acestep_worker.supports_dit_profile("quality")
    assert ok, reason

    plan = {
        "query": "",
        "caption": "orchestral",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 20,
    }
    job = {"action": "generate", "dit_profile": "quality", "seed": -1, "src_audio": None}

    meta, _, _ = acestep_worker.run_job(
        job=job, plan=plan, take_id="t3", take_dir=tmp_path / "take3"
    )
    assert meta["dit_profile"] == "quality"

    handler_init = next(e for e in log if e[0] == "handler.initialize_service")
    assert handler_init[1]["offload_to_cpu"] is True  # requested for XL

    generate_call = next(e for e in log if e[0] == "generate_music")
    params = generate_call[3]
    assert params.inference_steps == 8
    assert params.guidance_scale == 1.0


def test_track_name_maps_to_instruction_for_studio_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 4.4: extract/lego/complete's track selection must reach
    GenerationParams' task-specific `instruction` field -- the real class
    has no `track_name` kwarg (passing one raised TypeError on every real
    call, exactly what the reviewer flagged)."""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    plan = {
        "query": "",
        "caption": "orchestral",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 20,
    }

    for action in ("extract", "lego", "complete"):
        log.clear()
        job = {
            "action": action,
            "dit_profile": "studio_ops",
            "seed": -1,
            "src_audio": "/some/source.wav",
            "track_name": "vocals",
        }
        acestep_worker.run_job(
            job=job, plan=plan, take_id=f"t-{action}", take_dir=tmp_path / f"take-{action}"
        )
        generate_call = next(e for e in log if e[0] == "generate_music")
        params = generate_call[3]
        assert params.instruction == "vocals", action


def test_api_mismatch_raises_worker_unavailable_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If acestep's real API drifts from this adapter again, a job must
    fail cleanly (WorkerUnavailable -> job `error`), not crash the worker
    process (SPEC.md sec 10 point 5)."""

    class BrokenHandler:
        def __init__(self) -> None:
            pass

        # No initialize_service -- simulates a further upstream API change.

    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log, handler_cls=BrokenHandler)

    plan = {
        "query": "",
        "caption": "x",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 20,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    with pytest.raises(acestep_worker.WorkerUnavailable):
        acestep_worker.run_job(job=job, plan=plan, take_id="t4", take_dir=tmp_path / "take4")


def test_initialize_worker_preloads_default_iterate_dit_and_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC.md sec 10 point 1: 'On start: detect CUDA, log VRAM, load
    default iterate DiT + 1.7B LM with pt backend.'"""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)

    ready, message = acestep_worker.initialize_worker()

    assert ready is True
    assert "iterate" in message

    handler_init = next(e for e in log if e[0] == "handler.initialize_service")
    assert handler_init[1]["config_path"] == "acestep-v15-turbo"

    lm_init = next(e for e in log if e[0] == "lm.initialize")
    assert lm_init[1]["lm_model_path"] == acestep_worker.DEFAULT_LM

    assert acestep_worker._STATE["dit_profile"] == "iterate"
    assert acestep_worker._STATE["handler"] is not None
    assert acestep_worker._STATE["lm"] is not None


def test_initialize_worker_reports_failure_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing/broken ACE-Step install must be a reported, non-fatal
    startup failure (SPEC.md sec 10 point 5), not an unhandled crash --
    `worker/run_worker.py` uses this to publish honest capability state
    instead of an optimistic guess."""

    class BrokenHandler:
        def __init__(self) -> None:
            pass

        # No initialize_service -- simulates ACE-Step/CUDA being unavailable.

    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log, handler_cls=BrokenHandler)

    ready, message = acestep_worker.initialize_worker()

    assert ready is False
    assert "iterate" in message
    assert acestep_worker._STATE["handler"] is None


def test_initialize_worker_reports_ordinary_runtime_failure_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary ACE-Step/CUDA failures during a real load -- a missing
    checkpoint (OSError), a CUDA driver error or OOM (RuntimeError) -- are
    not TypeError/AttributeError, so `_api_method_call` doesn't convert them
    to WorkerUnavailable. initialize_worker must still catch and report
    them (not let them propagate and crash worker.run_worker)."""

    class OomHandler:
        def __init__(self) -> None:
            pass

        def initialize_service(self, **kwargs: Any) -> None:
            raise RuntimeError("CUDA out of memory")

    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log, handler_cls=OomHandler)

    ready, message = acestep_worker.initialize_worker()

    assert ready is False
    assert "out of memory" in message
    assert acestep_worker._STATE["handler"] is None


def test_initialize_worker_reports_lm_init_failure_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMHandler.initialize returns (status_message, success) -- a falsy
    success (e.g. missing LM weights) must not be cached in _STATE and
    reported as a successfully preloaded worker (exactly what the reviewer
    flagged: the return value was previously ignored entirely)."""
    log: list[tuple] = []
    _install_fake_acestep(
        monkeypatch, log, lm_init_result=("lm checkpoint not found", False)
    )

    ready, message = acestep_worker.initialize_worker()

    assert ready is False
    assert "lm checkpoint not found" in message
    assert acestep_worker._STATE["lm"] is None
    # the DiT handler *did* load fine -- only the LM failed
    assert acestep_worker._STATE["handler"] is not None


def test_initialize_worker_reports_dit_init_failure_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AceStepHandler.initialize_service also returns (status_message,
    success) -- a falsy success (e.g. missing/incompatible DiT weights)
    must not be cached in _STATE and reported as a successfully preloaded
    worker (the return value was previously discarded entirely)."""
    log: list[tuple] = []
    _install_fake_acestep(
        monkeypatch, log, handler_init_result=("dit checkpoint incompatible", False)
    )

    ready, message = acestep_worker.initialize_worker()

    assert ready is False
    assert "dit checkpoint incompatible" in message
    assert acestep_worker._STATE["handler"] is None
    assert acestep_worker._STATE["dit_profile"] is None
    assert acestep_worker._STATE["lm"] is None  # never reached the LM step


def test_unexpected_audio_format_is_a_clean_error_not_mislabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ACE-Step ignores audio_format='wav' and still returns its FLAC
    default (or anything else), the adapter must fail loudly instead of
    renaming a non-WAV file to mix.wav -- that file wouldn't reliably
    play/download as the WAV the API and storage layer promise."""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log, returned_audio_suffix=".flac")

    plan = {
        "query": "",
        "caption": "orchestral swell, cinematic strings",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 45,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    with pytest.raises(RuntimeError, match="unexpected format"):
        acestep_worker.run_job(job=job, plan=plan, take_id="t5", take_dir=tmp_path / "take5")

    # nothing got renamed/mislabeled
    assert not (tmp_path / "take5" / "mix.wav").exists()


def test_audio_path_outside_take_dir_is_rejected_not_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC.md sec 8.1/11: generated-audio filesystem operations must stay
    jailed under the take directory ACE-Step was told to use. The returned
    path is untrusted -- an upstream bug, API drift, or a compromised
    acestep install could point it anywhere -- so the adapter must refuse
    to shutil.move a path outside take_dir instead of moving/deleting an
    arbitrary local file."""
    log: list[tuple] = []
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_path = outside_dir / "evil.wav"
    _install_fake_acestep(monkeypatch, log, audio_path_override=outside_path)

    plan = {
        "query": "",
        "caption": "orchestral swell, cinematic strings",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 45,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}
    take_dir = tmp_path / "take-outside"

    with pytest.raises(RuntimeError, match="outside its allocated take directory"):
        acestep_worker.run_job(job=job, plan=plan, take_id="t-outside", take_dir=take_dir)

    # the file outside the jail was left exactly where it was ...
    assert outside_path.exists()
    # ... and nothing was moved into (or created under) take_dir
    assert not (take_dir / "mix.wav").exists()
    assert not (take_dir / "evil.wav").exists()


def _install_fake_lora_training(
    monkeypatch: pytest.MonkeyPatch,
    log: list[tuple],
    *,
    dataset_builder_cls: type | None = None,
    train_steps: list[tuple] | None = None,
    write_final: bool = True,
    label_count_override: int | None = None,
) -> None:
    """Register fake `acestep.training.*` modules so `worker.acestep_worker.
    train_lora`'s lazy imports resolve to them (SPEC.md sec 4.4/12; see the
    module docstring's LoRA section for the real call chain this mirrors:
    `DatasetBuilder.scan_directory` -> `get_labeled_count` ->
    `preprocess_to_tensors` -> `LoRAConfig`/`TrainingConfig` -> `LoRATrainer`
    -> `train_from_preprocessed`). Caller must also call
    `_install_fake_acestep` first -- `train_lora` still calls `_ensure_loaded`
    to get a DiT handler, same as `run_job`.

    The default `DatasetBuilder` fake mirrors the real
    `ScanMixin.scan_directory`/`PreprocessMixin.preprocess_to_tensors`
    contract closely enough to exercise the caption-sidecar labeling logic
    `train_lora` actually depends on: a sample is only "labeled" (and only
    then preprocessed) if a `<file>.caption.txt` sidecar exists next to it.
    `label_count_override`, if given, replaces the real computed count --
    used to simulate ACE-Step labeling fewer than `MIN_LORA_SOURCES` sources.
    `train_steps` are the `(step, loss, status)` tuples the fake
    `LoRATrainer.train_from_preprocessed` yields; `write_final`, if True
    (the default), writes a fake adapter file under
    `<training_config.output_dir>/final/` once the generator is exhausted --
    pass False to simulate ACE-Step finishing without ever producing one."""

    class DefaultFakeDatasetBuilder:
        def __init__(self) -> None:
            self.metadata = types.SimpleNamespace(name="", all_instrumental=False)
            self.samples: list[dict] = []

        def scan_directory(self, directory: str):
            log.append(("builder.scan_directory", directory))
            found = []
            for p in sorted(Path(directory).iterdir()):
                if p.name.endswith(".caption.txt"):
                    continue
                caption_path = p.with_name(p.name + ".caption.txt")
                found.append({"path": str(p), "labeled": caption_path.exists()})
            self.samples = found
            if not found:
                return [], f"no audio files in {directory}"
            return found, f"found {len(found)} audio files in {directory}"

        def get_labeled_count(self) -> int:
            if label_count_override is not None:
                return label_count_override
            return sum(1 for s in self.samples if s["labeled"])

        def preprocess_to_tensors(
            self, *, dit_handler: Any, output_dir: str, skip_existing: bool, progress_callback: Any
        ):
            log.append(
                ("builder.preprocess_to_tensors", dit_handler, output_dir, skip_existing, progress_callback)
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_paths = []
            for i, sample in enumerate(self.samples):
                if not sample["labeled"]:
                    continue
                out_path = Path(output_dir) / f"sample_{i}.pt"
                out_path.write_bytes(b"fake-tensor")
                output_paths.append(str(out_path))
            return output_paths, f"preprocessed {len(output_paths)} samples"

    class FakeLoRAConfig:
        def __init__(self, *, r: int, alpha: int, dropout: float) -> None:
            log.append(("LoRAConfig", {"r": r, "alpha": alpha, "dropout": dropout}))
            self.r = r
            self.alpha = alpha
            self.dropout = dropout

    class FakeTrainingConfig:
        def __init__(
            self,
            *,
            learning_rate: float,
            max_epochs: int,
            save_every_n_epochs: int,
            gradient_accumulation_steps: int,
            seed: int,
            output_dir: str,
        ) -> None:
            log.append(
                (
                    "TrainingConfig",
                    {
                        "learning_rate": learning_rate,
                        "max_epochs": max_epochs,
                        "save_every_n_epochs": save_every_n_epochs,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "seed": seed,
                        "output_dir": output_dir,
                    },
                )
            )
            self.learning_rate = learning_rate
            self.max_epochs = max_epochs
            self.save_every_n_epochs = save_every_n_epochs
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.seed = seed
            self.output_dir = output_dir

    class FakeLoRATrainer:
        def __init__(self, *, dit_handler: Any, lora_config: Any, training_config: Any) -> None:
            log.append(("LoRATrainer", dit_handler, lora_config, training_config))
            self._training_config = training_config

        def train_from_preprocessed(self, tensor_dir: str):
            log.append(("trainer.train_from_preprocessed", tensor_dir))
            for step, loss, status in train_steps or [(1, 0.5, "epoch 1/10"), (2, 0.3, "epoch 2/10")]:
                yield step, loss, status
            if write_final:
                final_dir = Path(self._training_config.output_dir) / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                (final_dir / "adapter_model.bin").write_bytes(b"fake-lora-adapter-weights")

    training_pkg = types.ModuleType("acestep.training")
    configs_mod = types.ModuleType("acestep.training.configs")
    dataset_builder_mod = types.ModuleType("acestep.training.dataset_builder")
    trainer_mod = types.ModuleType("acestep.training.trainer")

    configs_mod.LoRAConfig = FakeLoRAConfig
    configs_mod.TrainingConfig = FakeTrainingConfig
    dataset_builder_mod.DatasetBuilder = dataset_builder_cls or DefaultFakeDatasetBuilder
    trainer_mod.LoRATrainer = FakeLoRATrainer

    monkeypatch.setitem(sys.modules, "acestep.training", training_pkg)
    monkeypatch.setitem(sys.modules, "acestep.training.configs", configs_mod)
    monkeypatch.setitem(sys.modules, "acestep.training.dataset_builder", dataset_builder_mod)
    monkeypatch.setitem(sys.modules, "acestep.training.trainer", trainer_mod)


def _make_source_files(tmp_path: Path, count: int) -> list[Path]:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = sources_dir / f"song{i}.wav"
        p.write_bytes(b"fake-source-audio")
        paths.append(p)
    return paths


def _call_train_lora(source_paths: list[Path], name: str, lora_dir: Path, lora_id: str = "lora1"):
    return acestep_worker.train_lora(
        job={"name": name},
        project_id="proj",
        lora_id=lora_id,
        lora_dir=lora_dir,
        source_paths=source_paths,
    )


def test_train_lora_matches_installed_training_api_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)
    _install_fake_lora_training(monkeypatch, log)

    source_paths = _make_source_files(tmp_path, acestep_worker.MIN_LORA_SOURCES)
    lora_dir = tmp_path / "loras" / "lora1"

    meta = _call_train_lora(source_paths, "dreamy synthwave", lora_dir, lora_id="lora1")

    # Sources staged with a `.caption.txt` sidecar per file (SPEC.md sec
    # 4.4/12: real ACE-Step only labels -- and therefore trains on -- a
    # sample when a caption source exists next to it).
    staged = sorted((lora_dir / "dataset").glob("*.wav"))
    assert len(staged) == acestep_worker.MIN_LORA_SOURCES
    for staged_file in staged:
        caption_path = staged_file.with_name(staged_file.name + ".caption.txt")
        assert caption_path.exists()
        assert caption_path.read_text(encoding="utf-8") == "dreamy synthwave"

    scan_call = next(e for e in log if e[0] == "builder.scan_directory")
    assert scan_call[1] == str(lora_dir / "dataset")

    preprocess_call = next(e for e in log if e[0] == "builder.preprocess_to_tensors")
    _, dit_handler, output_dir, skip_existing, progress_callback = preprocess_call
    assert dit_handler is not None
    assert output_dir == str(lora_dir / "tensors")
    assert skip_existing is False
    assert progress_callback is None

    # LoRA/training hyperparameters mirror the upstream training route's own
    # request defaults (see worker.acestep_worker's LORA_* constants).
    lora_config_call = next(e for e in log if e[0] == "LoRAConfig")
    assert lora_config_call[1] == {"r": 64, "alpha": 128, "dropout": 0.1}
    training_config_call = next(e for e in log if e[0] == "TrainingConfig")
    assert training_config_call[1]["output_dir"] == str(lora_dir / "adapter")
    assert training_config_call[1]["max_epochs"] == 10

    trainer_call = next(e for e in log if e[0] == "LoRATrainer")
    assert trainer_call[1] is dit_handler  # same handler _ensure_loaded produced

    train_call = next(e for e in log if e[0] == "trainer.train_from_preprocessed")
    assert train_call[1] == str(lora_dir / "tensors")

    # The base checkpoint LoRA trains against is studio_ops (the real,
    # non-distilled base model) -- see LORA_BASE_DIT_PROFILE.
    assert meta["dit_profile"] == "studio_ops"
    assert meta["base_checkpoint"] == acestep_worker.DIT_CHECKPOINTS["studio_ops"]
    assert meta["name"] == "dreamy synthwave"
    assert meta["source_take_count"] == acestep_worker.MIN_LORA_SOURCES
    assert meta["id"] == "lora1"
    # The generator's last yielded (step, loss, status) is what gets
    # reported, not just "it finished".
    assert meta["final_step"] == 2
    assert meta["final_loss"] == 0.3
    assert meta["status"] == "epoch 2/10"

    final_dir = lora_dir / "adapter" / "final"
    assert final_dir.exists()
    assert any(final_dir.iterdir())


def test_train_lora_requires_minimum_source_count(tmp_path: Path) -> None:
    """SPEC.md sec 4.4: 'Style pack | LoRA train / load | 8+ songs'. Must
    reject clearly *before* touching acestep at all -- no fake acestep.*
    modules are installed for this test, so any attempt to import would
    raise ImportError/WorkerUnavailable for the wrong reason if the count
    check didn't come first."""
    source_paths = _make_source_files(tmp_path, acestep_worker.MIN_LORA_SOURCES - 1)
    with pytest.raises(acestep_worker.WorkerUnavailable, match="at least 8"):
        _call_train_lora(source_paths, "too few", tmp_path / "loras" / "lora2")


def test_train_lora_fails_when_too_few_samples_get_labeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ACE-Step's own scan only labels some of the staged sources (e.g. a
    caption sidecar failed to write), preprocessing must not silently proceed
    on too little data -- SPEC.md sec 4.4's '8+ songs' is about what actually
    gets trained on, not just what was requested."""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)
    _install_fake_lora_training(monkeypatch, log, label_count_override=3)

    source_paths = _make_source_files(tmp_path, acestep_worker.MIN_LORA_SOURCES)

    with pytest.raises(RuntimeError, match="only labeled 3"):
        _call_train_lora(source_paths, "x", tmp_path / "loras" / "lora3")

    assert not any(e[0] == "builder.preprocess_to_tensors" for e in log)


def test_train_lora_fails_when_no_final_adapter_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A training run that finishes without ever writing
    `<output_dir>/final/` must be a clean failure, not a "successful" lora
    with no actual weights."""
    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)
    _install_fake_lora_training(monkeypatch, log, write_final=False)

    source_paths = _make_source_files(tmp_path, acestep_worker.MIN_LORA_SOURCES)

    with pytest.raises(RuntimeError, match="wrote no adapter weights"):
        _call_train_lora(source_paths, "x", tmp_path / "loras" / "lora4")


def test_train_lora_api_mismatch_raises_worker_unavailable_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same contract as generate: if acestep's real training API drifts from
    this adapter, a job must fail cleanly (WorkerUnavailable -> job `error`),
    not crash the worker process."""

    class BrokenDatasetBuilder:
        def __init__(self) -> None:
            pass

        # No scan_directory -- simulates an upstream API change.

    log: list[tuple] = []
    _install_fake_acestep(monkeypatch, log)
    _install_fake_lora_training(monkeypatch, log, dataset_builder_cls=BrokenDatasetBuilder)

    source_paths = _make_source_files(tmp_path, acestep_worker.MIN_LORA_SOURCES)

    with pytest.raises(acestep_worker.WorkerUnavailable):
        _call_train_lora(source_paths, "x", tmp_path / "loras" / "lora5")


def test_importing_worker_module_never_touches_acestep_or_cuda() -> None:
    """SPEC.md sec 10 point 4 / sec 11: importing `worker.acestep_worker`
    must stay safe on a machine with no GPU and no ACE-Step install -- only
    calling `run_job`/`initialize_worker`/`supports_dit_profile` may reach
    for `acestep` or CUDA (`torch`). Runs in a fresh subprocess, with no fake
    `acestep.*` modules installed, so a stray module-level `import acestep`
    or `import torch` would fail the import for real instead of merely being
    shadowed by another test's monkeypatched fake modules or an
    already-imported `sys.modules` entry left over from earlier in this same
    process."""
    script = (
        "import sys\n"
        "import worker.acestep_worker\n"
        "assert 'acestep' not in sys.modules, sorted(sys.modules)\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
