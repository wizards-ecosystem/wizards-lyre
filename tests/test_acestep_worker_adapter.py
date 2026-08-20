"""Signature-level test for worker/acestep_worker.py's ACE-Step 1.5 adapter.

`tests/test_phase1_api.py` only ever exercises `worker.mock_worker`, so a
call-contract mismatch against the real ACE-Step 1.5 Python API (wrong
method name, wrong argument, a field on the wrong class) would go
undetected. This module installs fake `acestep.*` modules with the
signatures ACE-Step 1.5 actually exposes -- `AceStepHandler`/`LLMHandler`
constructed bare and loaded via `initialize_service(config_path=...)`,
module-level `create_sample`, `num_inference_steps`/`use_cfg` on
`GenerationParams` (not `GenerationConfig`) -- and runs `run_job` against
them, so `worker/acestep_worker.py`'s exact calling convention is verified
without requiring CUDA or the real acestep package. See SPEC.md sec 13.
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

    def initialize_service(self, *, config_path: str) -> None:
        self.config_path = config_path


class FakeGenerationParams:
    """Mirrors ACE-Step 1.5's real signature: every per-request field,
    including num_inference_steps/use_cfg, lives here -- not on
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
        num_inference_steps: int,
        use_cfg: bool,
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
        self.num_inference_steps = num_inference_steps
        self.use_cfg = use_cfg


class FakeGenerationConfig:
    """Only batch_size -- num_inference_steps/use_cfg belong on
    GenerationParams instead (this is exactly what the reviewer flagged)."""

    def __init__(self, *, batch_size: int) -> None:
        self.batch_size = batch_size


class FakeAudio:
    def __init__(self, path: str, seed: int) -> None:
        self.path = path
        self.seed = seed


class FakeResult:
    def __init__(self, audios: list[FakeAudio], extra_outputs: dict[str, Any]) -> None:
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

        def initialize_service(self, *, config_path: str, cpu_offload: bool = False) -> None:
            log.append(("handler.initialize_service", config_path, cpu_offload))
            self.config_path = config_path
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
        *, handler: Any, lm_handler: Any, params: Any, config: Any, save_dir: str
    ) -> FakeResult:
        log.append(("generate_music", handler, lm_handler, params, config, save_dir))
        # ACE-Step writes its own output filename -- not mix.wav -- so the
        # adapter has to find and rename it (SPEC.md sec 7.3).
        out_path = Path(save_dir) / "output_0.wav"
        _write_tiny_wav(out_path)
        return FakeResult(
            audios=[FakeAudio(path=str(out_path), seed=999)],
            extra_outputs={
                "caption": params.caption,
                "lyrics": params.lyrics,
                "bpm": params.bpm,
                "keyscale": params.keyscale,
                "duration": params.duration,
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
    assert meta["seed"] == 999  # from the fake audio record, not the -1 request
    assert (tmp_path / "take1" / "mix.wav").exists()  # renamed from output_0.wav

    handler_init = next(e for e in log if e[0] == "handler.initialize_service")
    assert handler_init[1] == acestep_worker._config_path("acestep-v15-turbo")
    assert handler_init[2] is False  # iterate does not need cpu_offload

    generate_call = next(e for e in log if e[0] == "generate_music")
    params, config = generate_call[3], generate_call[4]
    # SPEC.md sec 4.1: iterate = 8 steps, no CFG -- and these live on
    # GenerationParams, not GenerationConfig (the bug the reviewer flagged).
    assert params.num_inference_steps == 8
    assert params.use_cfg is False
    assert config.batch_size == 1
    assert not hasattr(config, "num_inference_steps")
    assert not hasattr(config, "use_cfg")


def test_simple_mode_uses_module_level_create_sample(
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
    assert handler_init[2] is True  # cpu_offload=True was requested for XL

    generate_call = next(e for e in log if e[0] == "generate_music")
    params = generate_call[3]
    assert params.num_inference_steps == 8
    assert params.use_cfg is False


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
