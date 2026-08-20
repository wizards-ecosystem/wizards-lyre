"""Signature-level test for worker/acestep_worker.py's ACE-Step 1.5 adapter.

`tests/test_phase1_api.py` only ever exercises `worker.mock_worker`, so a
call-contract mismatch against the real ACE-Step 1.5 Python API (wrong
method name, wrong argument, a field on the wrong class) would go
undetected. This module installs fake `acestep.*` modules with the
signatures ACE-Step 1.5 actually exposes -- `AceStepHandler` loaded via
`initialize_service(project_root=..., config_path=<checkpoint name>,
device=...)`, `LLMHandler` loaded via a *different* method,
`initialize(project_root=..., config_path=..., device=..., backend=...)`,
module-level `create_sample`, `inference_steps`/`guidance_scale` on
`GenerationParams` (not `GenerationConfig`), `generate_music(dit_handler=...)`
(not `handler=`), and audio results as dicts with the actual seed nested at
`audio["params"]["seed"]` -- then runs `run_job` against them, so
`worker/acestep_worker.py`'s exact calling convention is verified without
requiring CUDA or the real acestep package. See SPEC.md sec 13.
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


class FakeLLMHandler:
    def __init__(self) -> None:
        self.config_path: str | None = None

    def initialize(self, *, project_root: str, config_path: str, device: str, backend: str) -> None:
        self.project_root = project_root
        self.config_path = config_path
        self.device = device
        self.backend = backend


class FakeGenerationParams:
    """Mirrors ACE-Step 1.5's real signature: every per-request field,
    including inference_steps/guidance_scale, lives here -- not on
    GenerationConfig. A missing/extra/misplaced kwarg raises TypeError,
    same as it would against the real class."""

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
        thinking: bool,
        use_cot_metas: bool,
        seed: int,
        src_audio: Any,
        audio_cover_strength: Any,
        repainting_start: Any,
        repainting_end: Any,
        track_name: Any,
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
        self.thinking = thinking
        self.use_cot_metas = use_cot_metas
        self.seed = seed
        self.src_audio = src_audio
        self.audio_cover_strength = audio_cover_strength
        self.repainting_start = repainting_start
        self.repainting_end = repainting_end
        self.track_name = track_name
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale


class FakeGenerationConfig:
    """Only batch_size -- inference_steps/guidance_scale belong on
    GenerationParams instead (this is exactly what the reviewer flagged)."""

    def __init__(self, *, batch_size: int) -> None:
        self.batch_size = batch_size


class FakeResult:
    def __init__(self, audios: list[dict], extra_outputs: dict[str, Any]) -> None:
        self.success = True
        self.audios = audios
        self.extra_outputs = extra_outputs


def _install_fake_acestep(
    monkeypatch: pytest.MonkeyPatch, log: list[tuple], handler_cls: type | None = None
) -> None:
    """Register fake `acestep.*` modules so `worker.acestep_worker`'s lazy
    `from acestep... import ...` statements resolve to them instead of the
    real (uninstalled) package."""

    class DefaultFakeAceStepHandler:
        def __init__(self) -> None:
            self.config_path: str | None = None
            self.cpu_offload = False

        def initialize_service(
            self, *, project_root: str, config_path: str, device: str, cpu_offload: bool = False
        ) -> None:
            log.append(
                (
                    "handler.initialize_service",
                    {
                        "project_root": project_root,
                        "config_path": config_path,
                        "device": device,
                        "cpu_offload": cpu_offload,
                    },
                )
            )
            self.project_root = project_root
            self.config_path = config_path
            self.device = device
            self.cpu_offload = cpu_offload

    def create_sample(*, lm_handler: Any, query: str, instrumental: bool) -> dict:
        log.append(("create_sample", lm_handler, query, instrumental))
        return {
            "caption": f"auto: {query}",
            "lyrics": "[Instrumental]" if instrumental else f"[Verse]\n{query}",
            "bpm": 120,
            "keyscale": "C Major",
            "duration": 30,
        }

    def generate_music(
        *, dit_handler: Any, lm_handler: Any, params: Any, config: Any, save_dir: str
    ) -> FakeResult:
        log.append(("generate_music", dit_handler, lm_handler, params, config, save_dir))
        # ACE-Step writes its own output filename -- not mix.wav -- so the
        # adapter has to find and rename it (SPEC.md sec 7.3).
        out_path = Path(save_dir) / "output_0.wav"
        _write_tiny_wav(out_path)
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
        "instrumental": True,
        "bpm": 90,
        "keyscale": "D Minor",
        "duration_sec": 45,
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
    assert kwargs["cpu_offload"] is False  # iterate does not need cpu_offload

    generate_call = next(e for e in log if e[0] == "generate_music")
    params, config = generate_call[3], generate_call[4]
    # SPEC.md sec 4.1: iterate = 8 steps, no CFG -- and these live on
    # GenerationParams, not GenerationConfig (the bug the reviewer flagged).
    assert params.inference_steps == 8
    assert params.guidance_scale == 1.0
    assert config.batch_size == 1
    assert not hasattr(config, "inference_steps")
    assert not hasattr(config, "guidance_scale")


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
    assert handler_init[1]["cpu_offload"] is True  # requested for XL

    generate_call = next(e for e in log if e[0] == "generate_music")
    params = generate_call[3]
    assert params.inference_steps == 8
    assert params.guidance_scale == 1.0


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
