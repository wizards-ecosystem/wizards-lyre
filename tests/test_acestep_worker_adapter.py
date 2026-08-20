"""Signature-level test for worker/acestep_worker.py's ACE-Step 1.5 adapter.

`tests/test_phase1_api.py` only ever exercises `worker.mock_worker`, so a
call-contract mismatch against the real ACE-Step 1.5 Python API (wrong
method name, wrong argument, a field on the wrong class) would go
undetected. This module installs fake `acestep.*` modules with the
signatures ACE-Step 1.5 actually exposes -- `AceStepHandler` loaded via
`initialize_service(project_root=..., config_path=<checkpoint name>,
device=...)`, `LLMHandler` loaded via a *different* method with *different*
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
`audio["params"]["seed"]`, and FLAC-by-default output unless
`GenerationConfig(audio_format="wav")` is requested -- then runs `run_job`
against them, so `worker/acestep_worker.py`'s exact calling convention is
verified without requiring CUDA or the real acestep package. See SPEC.md
sec 13.
"""

from __future__ import annotations

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
    """Only batch_size -- inference_steps/guidance_scale belong on
    GenerationParams instead (this is exactly what the reviewer flagged)."""

    def __init__(self, *, batch_size: int, audio_format: str) -> None:
        self.batch_size = batch_size
        self.audio_format = audio_format


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
    simulate a documented planning failure."""

    class DefaultFakeAceStepHandler:
        def __init__(self) -> None:
            self.config_path: str | None = None
            self.offload_to_cpu = False

        def initialize_service(
            self, *, project_root: str, config_path: str, device: str, offload_to_cpu: bool = False
        ) -> tuple[str, bool]:
            log.append(
                (
                    "handler.initialize_service",
                    {
                        "project_root": project_root,
                        "config_path": config_path,
                        "device": device,
                        "offload_to_cpu": offload_to_cpu,
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

    def create_sample(*, lm_handler: Any, query: str, instrumental: bool) -> dict:
        log.append(("create_sample", lm_handler, query, instrumental))
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
        return FakeResult(
            audios=[audio],
            extra_outputs={
                "caption": params.caption,
                "lyrics": params.lyrics,
                "bpm": params.bpm,
                "keyscale": params.keyscale,
                "duration": params.duration,
                "vocal_language": "en",
                "timesignature": "4/4",
            },
        )

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

    meta, plan_patch = acestep_worker.run_job(
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

    handler_init = next(e for e in log if e[0] == "handler.initialize_service")
    kwargs = handler_init[1]
    assert kwargs["project_root"] == str(acestep_worker.CHECKPOINTS_ROOT)
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
        "vocal_language": None,
        "timesignature": None,
    }
    job = {"action": "generate", "dit_profile": "iterate", "seed": -1, "src_audio": None}

    meta, plan_patch = acestep_worker.run_job(
        job=job, plan=plan, take_id="t2", take_dir=tmp_path / "take2"
    )

    create_call = next(e for e in log if e[0] == "create_sample")
    assert create_call[2] == "dreamy synthwave drive"  # query passed through
    assert create_call[3] is False  # instrumental

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

    meta, _ = acestep_worker.run_job(job=job, plan=plan, take_id="t3", take_dir=tmp_path / "take3")
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
