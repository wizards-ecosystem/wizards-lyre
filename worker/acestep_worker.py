"""Real ACE-Step 1.5 worker (SPEC.md sec 4 and sec 10).

This is the production job backend (`server.jobs` selects it whenever
`BARD_WORKER` is unset or `acestep`). `worker.mock_worker` is the test/local
-dev-only stand-in -- see SPEC.md sec 11.

`acestep` and CUDA are imported lazily, inside functions, never at module
scope. Importing this module (e.g. because `server.jobs` looked up the
backend) must stay safe on a machine with no GPU and no ACE-Step install;
only calling `run_job` reaches for either. On a machine without ACE-Step
installed, `run_job` raises `WorkerUnavailable`, which `server.jobs` catches
and records as the job's `error` -- a missing/broken GPU stack fails the job,
not the HTTP process (sec 10 point 5). Run this worker as its own OS process
(`python -m worker.run_worker`, see that module) so a native crash inside
ACE-Step/CUDA can't take the FastAPI server down with it either.

Adapter notes (installed ACE-Step 1.5 Python API, SPEC.md sec 13 -- "follow
ACE-Step's current API and keep Bard's HTTP schema stable with an
adapter"): `AceStepHandler`/`LLMHandler` are constructed with no arguments.
`AceStepHandler` is loaded via `initialize_service(project_root=...,
config_path=<checkpoint name>, device=..., offload_to_cpu=...)` -- CPU
offload for `quality` (XL) is `offload_to_cpu`, not `cpu_offload`. Upstream
resolves the DiT checkpoint at `<project_root>/checkpoints/<config_path>`,
*not* at `<project_root>/<config_path>` -- `project_root` and the checkpoint
directory are two different things to ACE-Step, even though Bard only has
one (`CHECKPOINTS_ROOT`, SPEC.md sec 6). Passing `CHECKPOINTS_ROOT` itself as
`project_root` (an earlier version of this adapter did) resolves as
`checkpoints/checkpoints/<name>` and can't find real weights; `project_root`
must be `CHECKPOINTS_ROOT`'s *parent* instead, so see `_checkpoints_project_root()`.
`LLMHandler` is loaded via `initialize(checkpoint_dir=..., lm_model_path=
<lm name>, backend="pt", device=...)` -- a different method *and*
different argument names than the DiT handler, and `backend="pt"` must be
passed explicitly (SPEC.md sec 4.2: ACE-Step otherwise defaults it to
`"vllm"`, not a reliable native-Windows backend). Both
`initialize_service` and `LLMHandler.initialize` return `(status_message,
success)`; a falsy `success` is treated as a failed load, not cached as
ready. Simple-mode planning goes through the module-level `acestep.
inference.create_sample` (not a handler method); its result carries the
language under `language`, which this adapter maps onto Bard's own
`vocal_language` plan field. `GenerationParams` carries every per-request
field -- `inference_steps` and `guidance_scale` (not `num_inference_steps`/
`use_cfg`), `duration` (not `duration_sec`), plus `vocal_language` and
`timesignature` so Custom-mode plan metadata actually reaches the
renderer. It has no `negative_tags` or `track_name` field (both raised
TypeError on every real call): plan.json's `negative` list is not sent
until a real ACE-Step field name is confirmed, and `track_name` maps to
the task-specific `instruction` field, sent only for extract/lego/complete
(SPEC.md sec 4.4). `GenerationConfig` carries `batch_size`,
`audio_format="wav"` (ACE-Step defaults to FLAC otherwise, which this
adapter must not silently relabel as `.wav`), `use_random_seed` --
`GenerationParams.seed` alone is not enough to reproduce a take: this must
be explicitly `False` whenever a fixed (non--1) seed is requested, or
ACE-Step's own default (`True`) ignores it -- and `enable_normalization=True`
(SPEC.md sec 4.3: "default on", always requested explicitly rather than
trusting whatever ACE-Step's own default happens to be); `generate_music`
takes both plus `dit_handler` (not `handler`)
and `lm_handler`. Its `GenerationResult` reports `success` plus generated
files in `audios` (dicts) and LM/CoT-filled metadata in `extra_outputs`;
the actual seed used lives nested at `audio["params"]["seed"]`, not
top-level. Every acestep call below is wrapped so a further API drift
raises `WorkerUnavailable` (job `error`, not a crash) instead of an
unhandled exception; `tests/test_acestep_worker_adapter.py` exercises this
whole call contract against fake acestep modules so a mismatch is caught
without needing CUDA installed. There's no quality-score field on
`GenerationResult`/`extra_outputs` itself (SPEC.md sec 12 Phase 4); ACE-Step
1.5's real scoring path is a second call, `AceStepHandler.get_lyric_score`,
against decoder-attention tensors `generate_music` already returns in
`extra_outputs` when lyrics are non-empty -- see `_quality_score` for the
full contract. Unlike the calls above, a scoring failure never raises
`WorkerUnavailable`; it only ever falls back to `score: None`.

SPEC.md sec 12 Phase 4 also lists "LRC if upstream provides timestamps",
hedged as conditional on upstream support -- it is supported: `AceStepHandler`
also mixes in `LyricTimestampMixin`
(`acestep/core/generation/handler/lyric_timestamp.py`, not documented at the
field level in INFERENCE.md/Tutorial.md either), whose `get_lyric_timestamp`
takes the same five decoder-attention tensors as `get_lyric_score` (plus
`total_duration_seconds`) and returns an already-`[mm:ss.xx]`-formatted
`lrc_text` (via `MusicStampsAligner.get_timestamps_and_lrc` in
`acestep/core/scoring/dit_alignment.py`) alongside raw
`sentence_timestamps`/`token_timestamps` -- see `_lyric_timestamps` for the
full contract. Same as scoring: best-effort, never fails the take, falls
back to `lrc_text: None` (`meta["has_lrc"] = False`, no `lyrics.lrc`
written) rather than fabricating timestamps from a guessed duration.

LoRA training (SPEC.md sec 4.4/12 "Style pack | LoRA train / load | 8+
songs... GPU exclusive"; sec 13 "wrapped -- not [the upstream UI] itself"):
researched the same way as the two sections above -- this path isn't
documented at the field level in docs/en/*.md either (the upstream LoRA
training tutorial documents its own web UI's "Train LoRA" tab, a one-click
workflow, not a Python call), so this was read directly from
ace-step/ACE-Step-1.5's source on GitHub. That web UI turns out to be a thin
client of a real, separately-importable training pipeline: an internal
FastAPI app under `acestep/api/train_api_*.py` that the UI process talks to
over HTTP. Reading `acestep/api/train_api_lora_start_route.py` (its
`/v1/training/start` handler) surfaces the actual Python entry points behind
that HTTP hop, which `train_lora` below calls directly -- skipping both the
UI and that internal HTTP layer, not just the UI (SPEC.md sec 2's "no [
upstream UI]" rule and sec 3's "fully local" both still apply to it).

The real pipeline is three stages, each a real class/function, not CLI-only:
`acestep.training.dataset_builder.DatasetBuilder` (`scan_directory(dir)` walks
a directory of audio files into `AudioSample` records, then
`preprocess_to_tensors(dit_handler=..., output_dir=...)` runs each *labeled*
sample through the already-loaded DiT handler's VAE/text/lyric encoders and
writes one `<sample id>.pt` tensor file per song); `acestep.training.configs.
LoRAConfig`/`TrainingConfig` (plain dataclasses -- `LoRAConfig(r=..., alpha=
..., dropout=...)`, `TrainingConfig(learning_rate=..., max_epochs=...,
save_every_n_epochs=..., output_dir=...)`); and `acestep.training.trainer.
LoRATrainer(dit_handler=..., lora_config=..., training_config=...)`, whose
`train_from_preprocessed(tensor_dir)` is a *generator* yielding `(step, loss,
status)` -- the training loop only actually runs while this generator is
being iterated, and it writes the final adapter weights to
`<training_config.output_dir>/final/` itself once exhausted (no separate
export call needed for a first cut). `dit_handler` is the same `AceStepHandler`
instance `_ensure_loaded` already produces for generation -- LoRA training
reuses whichever DiT is currently loaded rather than a second handler.
Reusing it has a consequence `_STATE`'s own lora bookkeeping (`lora_id`/
`lora_adapter_path`, tracked for `_ensure_lora_adapter` below) never
observes: `LoRATrainer` injects the in-training PEFT adapter directly into
`dit_handler`'s decoder as a side effect of training, entirely outside
`add_lora`/`set_active_lora_adapter`/`set_use_lora`. `train_lora` below
therefore invalidates the whole cached handler once training finishes
(success or failure) rather than leaving it cached with `_STATE["lora_id"]`
still `None` -- otherwise a later plain generation would see that `None`,
believe nothing is loaded, no-op, and silently run through the
just-trained adapter anyway; a later explicit lora load would call
`add_lora` against a decoder that's already PEFT-wrapped from training.

LoRA loading (SPEC.md sec 4.4/12 "LoRA train / load" -- this is the "load"
half; `train_lora` above is "train"): the tutorials in docs/en/*.md only
cover the upstream web UI's "Use LoRA" checkbox, not a Python call, so this
too was read directly from ace-step/ACE-Step-1.5's source on GitHub --
specifically `LoraManagerMixin`
(`acestep/core/generation/handler/lora_manager.py`), which `AceStepHandler`
mixes in, and its `acestep/core/generation/handler/lora/lifecycle.py` +
`.../lora/controls.py` implementations. `add_lora(lora_path, adapter_name)`
injects a PEFT adapter directory (the same `<...>/adapter/final/` layout
`train_lora` above writes, containing `adapter_config.json`) into the
decoder, wrapping it in a PEFT `PeftModel` the first time it's called
without disturbing the base weights; `set_active_lora_adapter(adapter_name)`
selects which loaded adapter subsequent inference calls actually use, and
`set_use_lora(bool)` is LoRA's master on/off switch. `unload_lora()` removes
every currently loaded adapter and restores the plain base decoder -- used
by `_ensure_lora_adapter` below to swap cleanly (unload, then load the new
one) instead of stacking a second adapter on top of the first when a later
job asks for a different lora_id, or leaving a stale adapter active when a
later job asks for none at all. None of these four return a
`(status_message, success)` tuple like `initialize_service`/`initialize`
above -- but they are *not* silent either: ACE-Step reports an ordinary
operational failure (a missing/corrupt adapter path, the base-decoder
backup being unavailable to restore on unload, ...) as a returned status
string starting with "❌" rather than raising. `_check_lora_status`
below checks for that marker after every `_api_method_call`, so a failure
reported this way still raises `WorkerUnavailable` instead of being
recorded as a successful load/unload (treating any return as success was
reviewer-flagged: a take could be marked with a `lora_id` that was never
actually applied, or a stale adapter believed unloaded while still active).
A failure partway through the load/unload/swap sequence leaves the
handler's *actual* adapter state unknown (e.g. `unload_lora` succeeded but
the following `add_lora` failed) -- `_ensure_lora_adapter` responds by
invalidating the whole cached handler (not just the lora bookkeeping), so
`_ensure_loaded` constructs a fresh one for the next job instead of any
later job silently inheriting an indeterminate adapter state.

Two things worth flagging about what `train_lora` deliberately does *not*
replicate from the real pipeline: (1) `DatasetBuilder.scan_directory` only
marks a sample `AudioSample.labeled = True` -- and therefore eligible for
`preprocess_to_tensors`, which silently skips unlabeled samples -- when it
finds a `.caption.txt` sidecar (or `.json`/CSV) caption next to the audio
file; this adapter's `train_lora` signature carries no per-song captions,
only one `name` on the job for the whole style pack, so
every staged source file is given an identical `.caption.txt` sidecar
containing `name` -- reasonable for a *style* pack, whose whole point is one
shared style across its reference songs, but a future job that threads real
per-take captions through would produce richer per-sample labels. (2) the
real `/v1/training/start` route also runs a `RuntimeComponentManager` around
training to offload the VAE/text-encoder/LLM off the GPU first, trading
extra plumbing for headroom on a 12-16 GB card; `train_lora` skips that --
SPEC.md sec 4.4 targets "3090-class" (24 GB), and the studio_ops DiT profile
it trains against is already what `_ensure_loaded` uses for the biggest
generation jobs, so the same amount of VRAM that already works for those
jobs works here too.
"""

from __future__ import annotations

import inspect
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# One GPU occupant: one loaded DiT + one LM, jobs serialize (SPEC.md sec 4.3).
_LOCK = threading.Lock()
# lora_id/lora_adapter_path track what's currently loaded onto `handler` via
# _ensure_lora_adapter (SPEC.md sec 4.4 "LoRA train / load") -- both reset to
# None whenever `handler` itself is swapped/recreated in _ensure_loaded,
# since a freshly constructed handler has no adapters loaded regardless of
# what the previous one had.
_STATE: dict[str, Any] = {
    "dit_profile": None,
    "handler": None,
    "lm": None,
    "lora_id": None,
    "lora_adapter_path": None,
}

DIT_CHECKPOINTS = {
    "iterate": "acestep-v15-turbo",
    "polish": "acestep-v15-sft",
    "quality": "acestep-v15-xl-turbo",
    "studio_ops": "acestep-v15-base",
}
DEFAULT_LM = "acestep-5Hz-lm-1.7B"
# SPEC.md sec 4.2: locked default LM backend. ACE-Step defaults
# LLMHandler.initialize's `backend` to "vllm", which is not a reliable
# native-Windows backend -- must be passed explicitly on every call.
LM_BACKEND = "pt"

# SPEC.md sec 6: weights live under checkpoints/<name>/ (gitignored).
# CHECKPOINTS_ROOT *is* that checkpoints/ directory -- it is NOT the same
# thing as AceStepHandler.initialize_service's `project_root` (see
# _checkpoints_project_root() below and the module docstring above).
# LLMHandler.initialize's `checkpoint_dir`, by contrast, *is* used directly
# as CHECKPOINTS_ROOT with no extra nesting -- only the DiT handler has this
# project_root/checkpoints/ indirection.
CHECKPOINTS_ROOT = Path(os.environ.get("BARD_CHECKPOINTS_DIR", "checkpoints"))
DEVICE = os.environ.get("BARD_DEVICE", "cuda")

# SPEC.md sec 4.1: XL turbo (`quality`) needs CPU offload on a 16 GB card;
# every other profile fits without it.
CPU_OFFLOAD_PROFILES = {"quality"}

# SPEC.md sec 4.1 DiT table: steps / CFG guidance per profile. CFG "no"
# profiles use a guidance_scale of 1.0 (no classifier-free guidance).
GENERATION_STEPS = {
    "iterate": 8,
    "polish": 50,
    "quality": 8,
    "studio_ops": 50,
}
GENERATION_GUIDANCE_SCALE = {
    "iterate": 1.0,
    "polish": 7.5,
    "quality": 1.0,
    "studio_ops": 7.5,
}

TASK_TYPE_BY_ACTION = {
    "generate": "text2music",
    "cover": "cover",
    "repaint": "repaint",
    "extract": "extract",
    "lego": "lego",
    "complete": "complete",
}

# SPEC.md sec 4.4 studio_ops task map: extract/lego/complete pass their
# track selection through GenerationParams' task-specific `instruction`
# field; generate/cover/repaint don't take one.
TRACK_INSTRUCTION_ACTIONS = {"extract", "lego", "complete"}

# SPEC.md sec 4.4: "Style pack | LoRA train / load | 8+ songs".
MIN_LORA_SOURCES = 8

# LoRA training needs a real (non-distilled) fine-tuning target -- the base
# 2B "studio_ops" checkpoint is that; turbo/xl-turbo are distilled few-step
# checkpoints ACE-Step's own trainer.py isn't built to run a full diffusion
# training loop against (see the module docstring's LoRA section).
LORA_BASE_DIT_PROFILE = "studio_ops"

# Mirrors the upstream training route's own request defaults
# (`StartTrainingRequest` in acestep/api/train_api_models.py) -- the numbers
# a user leaves untouched when starting training in the stock UI, and
# presumably what SPEC.md sec 4.4's "~1 hour on 3090-class GPU" throughput
# figure is based on.
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
LORA_LEARNING_RATE = 1e-4
LORA_TRAIN_EPOCHS = 10
LORA_SAVE_EVERY_N_EPOCHS = 5
LORA_GRADIENT_ACCUMULATION_STEPS = 4


class WorkerUnavailable(RuntimeError):
    """ACE-Step is not installed, has no usable GPU, or its Python API no
    longer matches this adapter."""


def _import_acestep():
    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            create_sample,
            generate_music,
        )
        from acestep.llm_inference import LLMHandler
    except ImportError as exc:
        raise WorkerUnavailable(
            "acestep is not installed in this environment. Install ACE-Step 1.5 "
            "and download weights (SPEC.md sec 4 and 13), or set BARD_WORKER=mock "
            "for local dev/tests without a GPU."
        ) from exc
    return AceStepHandler, GenerationParams, GenerationConfig, LLMHandler, generate_music, create_sample


def _import_lora_training():
    """Lazy import for the LoRA training pipeline (see the module docstring's
    LoRA section) -- a separate try/except from `_import_acestep` because
    `acestep.training.*` is real but distinct from the inference-path
    modules that function imports, and a machine with plain acestep
    installed but not its training extras should get a clear message about
    which half is missing."""
    try:
        from acestep.training.configs import LoRAConfig, TrainingConfig
        from acestep.training.dataset_builder import DatasetBuilder
        from acestep.training.trainer import LoRATrainer
    except ImportError as exc:
        raise WorkerUnavailable(
            "acestep's LoRA training pipeline (acestep.training.*) is not installed in this "
            "environment. Install ACE-Step 1.5 with its training dependencies (SPEC.md sec "
            "4.4/13), or set BARD_WORKER=mock for local dev/tests without a GPU."
        ) from exc
    return DatasetBuilder, LoRAConfig, TrainingConfig, LoRATrainer


def _api_call(step: str, fn, *args, **kwargs):
    """Call into acestep and turn a signature mismatch into `WorkerUnavailable`
    instead of an unhandled crash -- this adapter's field/argument names are
    our best mapping onto ACE-Step 1.5's documented API and may need
    updating if upstream drifts further (SPEC.md sec 13)."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, AttributeError) as exc:
        raise WorkerUnavailable(
            f"acestep API mismatch in {step}: {exc}. Update worker/acestep_worker.py to "
            "match the installed acestep version, or set BARD_WORKER=mock."
        ) from exc


def _api_method_call(step: str, obj: Any, method_name: str, *args, **kwargs):
    """Like `_api_call`, but resolves `obj.method_name` inside the guarded
    try too. A plain `_api_call(step, obj.method_name, ...)` would still let
    a missing method raise AttributeError from the *caller's* frame -- the
    attribute lookup happens before `_api_call` is even entered -- so a
    renamed/removed method wouldn't get the clean `WorkerUnavailable` this
    adapter promises everywhere else."""
    try:
        method = getattr(obj, method_name)
        return method(*args, **kwargs)
    except (TypeError, AttributeError) as exc:
        raise WorkerUnavailable(
            f"acestep API mismatch in {step}: {exc}. Update worker/acestep_worker.py to "
            "match the installed acestep version, or set BARD_WORKER=mock."
        ) from exc


def _check_init_result(step: str, result: Any) -> None:
    """Both `AceStepHandler.initialize_service` and `LLMHandler.initialize`
    return `(status_message, success)` -- treat init as failed unless
    `success` is confirmed True, instead of assuming any return value means
    success. Otherwise missing/incompatible weights or another init failure
    gets cached in `_STATE` and reported as a successfully preloaded worker
    (exactly what the reviewer flagged)."""
    try:
        status_message, success = result
    except (TypeError, ValueError):
        raise WorkerUnavailable(
            f"acestep API mismatch in {step}: expected an (status_message, success) "
            f"tuple, got {result!r}."
        ) from None
    if not success:
        raise WorkerUnavailable(f"{step} reported failure: {status_message}")


def _check_lora_status(step: str, result: Any) -> None:
    """`AceStepHandler.add_lora`/`set_active_lora_adapter`/`set_use_lora`/
    `unload_lora` (see the module docstring's LoRA-loading section) don't
    return a `(status_message, success)` tuple like `initialize_service`/
    `initialize` -- but a falsy/exception-free return does not mean
    success either: ACE-Step reports an ordinary operational failure (a
    missing/corrupt adapter path, the base-decoder backup being
    unavailable to restore on unload, ...) as a returned status string
    starting with "❌" instead of raising. Treating any return as success
    (what this adapter previously did) let a failed load/unload get
    recorded in `_STATE` as if it had actually happened -- a take could be
    marked with a `lora_id` that was never really applied, or a stale
    adapter believed unloaded while still active on the decoder."""
    if isinstance(result, str) and result.strip().startswith("❌"):
        raise WorkerUnavailable(f"{step} reported failure: {result}")


def _checkpoints_project_root() -> Path:
    """The `project_root` to pass to `AceStepHandler.initialize_service`
    (SPEC.md sec 6 / module docstring): upstream resolves each DiT checkpoint
    at `<project_root>/checkpoints/<config_path>`, so `project_root` must be
    `CHECKPOINTS_ROOT`'s *parent* for that to land on `CHECKPOINTS_ROOT`
    itself -- which only lines up if `CHECKPOINTS_ROOT`'s own directory name
    is literally `checkpoints` (true for the SPEC-locked default; required of
    any `BARD_CHECKPOINTS_DIR` override too, since ACE-Step's `checkpoints/`
    segment is fixed, not something this adapter can rename around). Raises
    `WorkerUnavailable` instead of silently resolving to the wrong directory
    (exactly the bug the reviewer flagged)."""
    if CHECKPOINTS_ROOT.name != "checkpoints":
        raise WorkerUnavailable(
            f"BARD_CHECKPOINTS_DIR ('{CHECKPOINTS_ROOT}') must be a directory named "
            "'checkpoints': AceStepHandler.initialize_service resolves DiT checkpoints "
            "at <project_root>/checkpoints/<config_path>, and this adapter derives "
            "project_root as CHECKPOINTS_ROOT's parent so that lands on the real "
            "weights directory."
        )
    return CHECKPOINTS_ROOT.parent


def _handler_supports_cpu_offload(AceStepHandler: Any) -> bool:
    try:
        params = inspect.signature(AceStepHandler.initialize_service).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return "offload_to_cpu" in params


def _log_cuda_status() -> str:
    """Detect CUDA and log VRAM (SPEC.md sec 10 point 1). Diagnostic only --
    never raises, so a missing/CPU-only torch still lets the worker start
    and report a clean per-job error instead of refusing to run at all."""
    try:
        import torch
    except ImportError:
        return "CUDA detect: torch is not installed"
    try:
        if not torch.cuda.is_available():
            return "CUDA detect: not available (no GPU or a CPU-only torch build)"
        device_count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return (
            f"CUDA detect: {device_count} device(s), using '{name}'; VRAM "
            f"{free_bytes / (1024 ** 3):.1f} GiB free / {total_bytes / (1024 ** 3):.1f} GiB total"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only, must never block startup
        return f"CUDA detect: error while querying CUDA/VRAM: {exc}"


def get_loaded_dit_profile() -> str | None:
    """Which DiT profile (if any) is currently loaded in this process's
    memory, or None if nothing has loaded successfully yet. Read-only --
    never triggers a load. `worker/run_worker.py` publishes this for
    `/api/health` (SPEC.md sec 8 `dit_loaded`)."""
    return _STATE["dit_profile"]


def initialize_worker() -> tuple[bool, str]:
    """Worker-startup readiness (SPEC.md sec 10 point 1): detect CUDA, log
    VRAM, and preload the default `iterate` DiT + LM (`pt` backend) before
    the worker starts polling for jobs. Returns `(ready, message)`; never
    raises -- a failed startup is reported, not fatal, so the process stays
    up and still reports a clean per-job `WorkerUnavailable` (sec 10 point
    5) instead of refusing to run at all (e.g. while iterating without a
    GPU attached). `worker/run_worker.py` publishes the result so
    `server.jobs` can reflect real startup state, not an optimistic guess."""
    print(_log_cuda_status())
    try:
        with _LOCK:
            _ensure_loaded("iterate")
    except Exception as exc:  # noqa: BLE001 - see below
        # `_api_call`/`_api_method_call` only convert signature mismatches
        # (TypeError/AttributeError) to WorkerUnavailable; ordinary
        # ACE-Step/CUDA failures during a real load -- missing checkpoint
        # files (OSError), a CUDA/driver error or OOM (RuntimeError) -- can
        # still reach here directly. Both must be a reported, non-fatal
        # readiness failure (SPEC.md sec 10 point 5), not an uncaught
        # exception that kills worker.run_worker before it starts polling.
        # `except Exception` still lets KeyboardInterrupt/SystemExit through.
        message = f"worker startup: failed to preload default 'iterate' DiT + LM: {exc}"
        print(message)
        return False, message
    message = (
        f"worker startup: default 'iterate' DiT ({DIT_CHECKPOINTS['iterate']}) + "
        f"LM ({DEFAULT_LM}) loaded and ready"
    )
    print(message)
    return True, message


def supports_dit_profile(dit_profile: str) -> tuple[bool, str | None]:
    """Whether this worker can currently load `dit_profile`. SPEC.md sec 4.1:
    'quality' (XL, 4B) requires CPU offload on a 16 GB GPU; SPEC.md sec 8.1:
    'Reject quality if worker reports it cannot load XL with offload.'

    Only ever called from this worker process (directly, or via
    `worker/run_worker.py` publishing the result to SQLite for
    `server.jobs` to read) -- the FastAPI server must never import acestep
    itself just to answer this (SPEC.md sec 10 point 4)."""
    if dit_profile not in CPU_OFFLOAD_PROFILES:
        return True, None
    try:
        AceStepHandler, _, _, _, _, _ = _import_acestep()
    except WorkerUnavailable as exc:
        return False, str(exc)
    if not _handler_supports_cpu_offload(AceStepHandler):
        return False, (
            f"dit_profile 'quality' ({DIT_CHECKPOINTS['quality']}) requires a "
            "CPU-offload-capable AceStepHandler.initialize_service() on a 16 GB GPU; "
            "the installed acestep does not support offload_to_cpu. Use 'iterate' or "
            "'polish' instead."
        )
    return True, None


def _ensure_loaded(dit_profile: str) -> tuple[Any, Any, Any]:
    """Swap the loaded DiT if needed. Caller holds _LOCK."""
    AceStepHandler, _, _, LLMHandler, _, _ = _import_acestep()

    if _STATE["dit_profile"] != dit_profile or _STATE["handler"] is None:
        # Unload the previous DiT before loading the next; never hold two
        # DiTs at once on a 16 GB card (SPEC.md sec 4.3).
        _STATE["handler"] = None
        # A freshly constructed handler starts with no LoRA adapters loaded
        # -- forget whatever _ensure_lora_adapter previously recorded onto
        # the *old* handler object, or a later job could wrongly believe the
        # new handler already has an adapter loaded and skip loading it.
        _STATE["lora_id"] = None
        _STATE["lora_adapter_path"] = None
        checkpoint = DIT_CHECKPOINTS[dit_profile]
        cpu_offload = dit_profile in CPU_OFFLOAD_PROFILES
        if cpu_offload:
            ok, reason = supports_dit_profile(dit_profile)
            if not ok:
                raise WorkerUnavailable(reason)

        handler = _api_call("AceStepHandler()", AceStepHandler)
        init_kwargs: dict[str, Any] = {
            "project_root": str(_checkpoints_project_root()),
            "config_path": checkpoint,
            "device": DEVICE,
        }
        if cpu_offload:
            init_kwargs["offload_to_cpu"] = True
        result = _api_method_call(
            "AceStepHandler.initialize_service", handler, "initialize_service", **init_kwargs
        )
        # AceStepHandler.initialize_service returns (status_message,
        # success) just like LLMHandler.initialize -- check it instead of
        # assuming any return means success, or missing/incompatible
        # weights get cached in _STATE and reported as a ready worker.
        _check_init_result("AceStepHandler.initialize_service", result)

        _STATE["handler"] = handler
        _STATE["dit_profile"] = dit_profile

    if _STATE["lm"] is None:
        lm = _api_call("LLMHandler()", LLMHandler)
        # LLMHandler is loaded through `initialize`, not `initialize_service`
        # -- a different method *and* different argument names than the DiT
        # handler above (checkpoint_dir/lm_model_path, not project_root/
        # config_path). backend=LM_BACKEND ("pt") is required: ACE-Step
        # otherwise defaults to "vllm", violating SPEC.md sec 4.2's locked
        # native-Windows default.
        result = _api_method_call(
            "LLMHandler.initialize",
            lm,
            "initialize",
            checkpoint_dir=str(CHECKPOINTS_ROOT),
            lm_model_path=DEFAULT_LM,
            backend=LM_BACKEND,
            device=DEVICE,
        )
        # Check the (status_message, success) result instead of assuming
        # any return means success -- see _check_init_result.
        _check_init_result("LLMHandler.initialize", result)
        _STATE["lm"] = lm

    return _STATE["handler"], _STATE["lm"], _STATE["dit_profile"]


def _ensure_lora_adapter(handler: Any, lora_id: str | None, lora_adapter_path: str | None) -> None:
    """Load, swap, or unload the requested LoRA adapter onto `handler`
    (SPEC.md sec 4.4 "LoRA train / load" -- this is the "load" half;
    `train_lora` below is "train"). Caller holds _LOCK and has already
    called `_ensure_loaded`, same ordering as everything else in `run_job`.

    See the module docstring's LoRA-loading section for the real API this
    wraps (`AceStepHandler.add_lora`/`set_active_lora_adapter`/
    `set_use_lora`/`unload_lora`, via `LoraManagerMixin`).
    `server.jobs._resolve_dit_profile` only ever lets a non-None
    `lora_adapter_path` through on a job whose resolved dit_profile is
    `LORA_BASE_DIT_PROFILE` (studio_ops) -- see its docstring -- since a
    LoRA's weight deltas are only valid against that exact base checkpoint;
    `run_job` below re-asserts that invariant rather than trusting the
    caller blindly.

    A no-op if `lora_id` already matches what's currently loaded (including
    both None -- nothing requested, nothing loaded). Otherwise unloads
    whatever is currently loaded first: a later job requesting a *different*
    lora_id (or none at all) must not silently stack a second adapter on top
    of the first, or leave a stale adapter active once nothing needs it.

    Every call is checked with `_check_lora_status` (see its docstring: a
    failure here is reported as a "❌"-prefixed status string, not an
    exception). If any step of the transition fails, the handler's actual
    adapter state is left unknown -- e.g. `unload_lora` may have succeeded
    while the following `add_lora` failed, or vice versa -- so this
    invalidates the *entire* cached handler (not just the lora bookkeeping)
    before re-raising, forcing `_ensure_loaded` to construct a fresh handler
    for the next job rather than any later job inheriting an indeterminate
    adapter state.
    """
    if lora_id == _STATE["lora_id"]:
        return
    try:
        if _STATE["lora_id"] is not None:
            result = _api_method_call("AceStepHandler.unload_lora", handler, "unload_lora")
            _check_lora_status("AceStepHandler.unload_lora", result)
            _STATE["lora_id"] = None
            _STATE["lora_adapter_path"] = None
        if lora_id is not None:
            result = _api_method_call(
                "AceStepHandler.add_lora",
                handler,
                "add_lora",
                lora_path=lora_adapter_path,
                adapter_name=lora_id,
            )
            _check_lora_status("AceStepHandler.add_lora", result)
            result = _api_method_call(
                "AceStepHandler.set_active_lora_adapter",
                handler,
                "set_active_lora_adapter",
                adapter_name=lora_id,
            )
            _check_lora_status("AceStepHandler.set_active_lora_adapter", result)
            result = _api_method_call(
                "AceStepHandler.set_use_lora", handler, "set_use_lora", use_lora=True
            )
            _check_lora_status("AceStepHandler.set_use_lora", result)
            _STATE["lora_id"] = lora_id
            _STATE["lora_adapter_path"] = lora_adapter_path
    except WorkerUnavailable:
        _STATE["handler"] = None
        _STATE["dit_profile"] = None
        _STATE["lora_id"] = None
        _STATE["lora_adapter_path"] = None
        raise


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
        lm_handler=lm,
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
        # (that's Bard's own plan.json field name -- see storage.default_plan).
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
    except Exception:  # noqa: BLE001 - scoring is optional, never fails the take
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
    except Exception:  # noqa: BLE001 - timestamps are optional, never fail the take
        return None


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
            caption=effective_plan.get("caption", ""),
            lyrics=effective_plan.get("lyrics", ""),
            bpm=effective_plan.get("bpm"),
            keyscale=effective_plan.get("keyscale"),
            duration=effective_plan.get("duration_sec"),
            instrumental=effective_plan.get("instrumental", False),
            vocal_language=effective_plan.get("vocal_language"),
            timesignature=effective_plan.get("timesignature"),
            thinking=thinking,
            use_cot_metas=use_cot_metas,
            seed=seed,
            src_audio=job.get("src_audio"),
            audio_cover_strength=job.get("audio_cover_strength"),
            repainting_start=job.get("repainting_start"),
            repainting_end=job.get("repainting_end"),
            # extract/lego/complete's track selection goes through the
            # task-specific `instruction` field, not a `track_name` kwarg
            # (GenerationParams has no such field -- passing it raised
            # TypeError on every real call, converted to WorkerUnavailable).
            instruction=(
                job.get("track_name") if job["action"] in TRACK_INSTRUCTION_ACTIONS else None
            ),
            inference_steps=GENERATION_STEPS[dit_profile],
            guidance_scale=GENERATION_GUIDANCE_SCALE[dit_profile],
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
            # SPEC.md sec 4.3: loudness normalization is "default on, no
            # extra mastering chain in v1" -- requested explicitly rather
            # than relying on ACE-Step's own default staying True upstream.
            enable_normalization=True,
        )
        result = _api_call(
            "generate_music",
            generate_music,
            dit_handler=handler,
            lm_handler=lm,
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
        handler, extra_lookup=_extra, lyrics=lyrics, vocal_language=vocal_language, dit_profile=dit_profile
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
        "created_at": datetime.now(timezone.utc).isoformat(),
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


def train_lora(
    job: dict[str, Any],
    project_id: str,
    lora_id: str,
    lora_dir: Path,
    source_paths: list[Path],
) -> dict:
    """Train a style-pack LoRA from `source_paths` (SPEC.md sec 4.4/12; see
    the module docstring's LoRA section for the real ACE-Step pipeline this
    wraps: `DatasetBuilder.scan_directory` -> `preprocess_to_tensors` ->
    `LoRATrainer.train_from_preprocessed`). Requires at least
    `MIN_LORA_SOURCES` source files -- SPEC.md sec 4.4 is explicit ("8+
    songs") and the real `DatasetBuilder`/`LoRATrainer` enforce no minimum of
    their own, so an under-sized request would otherwise train "successfully"
    on far too little data instead of failing clearly up front. `server.jobs`
    also enforces this at enqueue time (fail fast, before a GPU job is even
    queued); this check is the same invariant enforced again at the point
    that actually matters if this function is ever called directly.

    Call shape matches `worker.mock_worker.train_lora` so `server.jobs`
    `_run_train_lora_job` can dispatch either backend. Writes the staged
    source copies, preprocessed tensors, and final adapter weights under
    `lora_dir` (already allocated and jailed by the caller, same division of
    labor as `run_job`'s `take_dir`). Returns a meta dict for the caller to
    persist via `server.storage.write_lora_meta` -- this function never
    writes `lora_dir/meta.json` itself, mirroring how `run_job` above
    returns `take_meta` instead of writing it directly.

    Runs entirely under `_LOCK`, including the (potentially ~1 hour, SPEC.md
    sec 4.4) training loop itself -- "GPU exclusive" (sec 4.4) means no other
    job may run concurrently, not just that the initial model swap is
    serialized.

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable, the training
    pipeline isn't installed, or its API no longer matches this adapter; a
    plain `RuntimeError` if staging/scanning/preprocessing/training itself
    fails despite the API matching (e.g. ACE-Step found no audio files, or
    training produced no adapter weights). Either way `server.jobs` catches
    it and records the job as `error` without crashing the worker process.
    """
    name = (job.get("name") or "").strip() or "Untitled LoRA"
    if len(source_paths) < MIN_LORA_SOURCES:
        raise WorkerUnavailable(
            f"train_lora requires at least {MIN_LORA_SOURCES} source audio files (SPEC.md sec "
            f"4.4: 'Style pack | LoRA train / load | 8+ songs'), got {len(source_paths)}."
        )

    DatasetBuilder, LoRAConfig, TrainingConfig, LoRATrainer = _import_lora_training()

    lora_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = lora_dir / "dataset"
    tensor_dir = lora_dir / "tensors"
    adapter_dir = lora_dir / "adapter"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Stage each source into a scratch directory DatasetBuilder.scan_directory
    # can walk, with a `.caption.txt` sidecar so ACE-Step marks it `labeled`
    # (see the module docstring: an audio file with no caption source stays
    # unlabeled and preprocess_to_tensors silently skips it). This adapter's
    # signature carries no per-song captions, only one shared `name` -- see
    # the module docstring for why that's a reasonable simplification for a
    # style pack specifically.
    for i, src in enumerate(source_paths):
        dest = dataset_dir / f"{i:03d}_{src.stem}{src.suffix}"
        shutil.copyfile(src, dest)
        dest.with_name(dest.name + ".caption.txt").write_text(name, encoding="utf-8")

    with _LOCK:
        handler, _lm, dit_profile = _ensure_loaded(LORA_BASE_DIT_PROFILE)

        try:
            builder = _api_call("DatasetBuilder", DatasetBuilder)
            try:
                builder.metadata.name = name
                # No per-song lyrics/instrumental info reaches this adapter (see
                # above) -- treat every staged source as instrumental, the same
                # conservative default DatasetMetadata itself ships with.
                builder.metadata.all_instrumental = True
            except AttributeError as exc:
                # Same reasoning as _api_method_call: a plain attribute access
                # would otherwise raise AttributeError straight from this frame
                # instead of the clean WorkerUnavailable every other API-mismatch
                # path in this module promises.
                raise WorkerUnavailable(
                    f"acestep API mismatch in DatasetBuilder.metadata: {exc}. Update "
                    "worker/acestep_worker.py to match the installed acestep version, or set "
                    "BARD_WORKER=mock."
                ) from exc

            samples, scan_status = _api_method_call(
                "DatasetBuilder.scan_directory", builder, "scan_directory", str(dataset_dir)
            )
            if not samples:
                raise RuntimeError(f"ACE-Step found no audio files to train on: {scan_status}")
            labeled_count = _api_method_call(
                "DatasetBuilder.get_labeled_count", builder, "get_labeled_count"
            )
            if labeled_count < MIN_LORA_SOURCES:
                raise RuntimeError(
                    f"ACE-Step only labeled {labeled_count}/{len(samples)} staged source files "
                    f"(needs >= {MIN_LORA_SOURCES}): {scan_status}"
                )

            output_paths, preprocess_status = _api_method_call(
                "DatasetBuilder.preprocess_to_tensors",
                builder,
                "preprocess_to_tensors",
                dit_handler=handler,
                output_dir=str(tensor_dir),
                skip_existing=False,
                progress_callback=None,
            )
            if not output_paths:
                raise RuntimeError(
                    f"ACE-Step preprocessing produced no training tensors: {preprocess_status}"
                )

            lora_config = _api_call(
                "LoRAConfig", LoRAConfig, r=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT
            )
            training_config = _api_call(
                "TrainingConfig",
                TrainingConfig,
                learning_rate=LORA_LEARNING_RATE,
                max_epochs=LORA_TRAIN_EPOCHS,
                save_every_n_epochs=LORA_SAVE_EVERY_N_EPOCHS,
                gradient_accumulation_steps=LORA_GRADIENT_ACCUMULATION_STEPS,
                seed=42,
                output_dir=str(adapter_dir),
            )
            trainer = _api_call(
                "LoRATrainer",
                LoRATrainer,
                dit_handler=handler,
                lora_config=lora_config,
                training_config=training_config,
            )

            train_iter = _api_method_call(
                "LoRATrainer.train_from_preprocessed",
                trainer,
                "train_from_preprocessed",
                str(tensor_dir),
            )
            # train_from_preprocessed is a generator -- the training loop itself
            # only runs while this is iterated; it writes the final adapter to
            # <adapter_dir>/final/ once exhausted (see the module docstring).
            last_step, last_loss, last_status = 0, None, "not started"
            for step, loss, status in train_iter:
                last_step, last_loss, last_status = step, loss, status
        finally:
            # LoRATrainer injects the in-training PEFT adapter directly into
            # `handler`'s decoder as a side effect of training (see the
            # module docstring's LoRA-training section) -- entirely outside
            # add_lora/set_active_lora_adapter/set_use_lora, so _STATE's own
            # lora bookkeeping never observes it and would otherwise still
            # read `lora_id: None` afterward. Invalidate the whole cached
            # handler unconditionally -- whether training above succeeded or
            # raised -- so _ensure_loaded constructs a fresh one next time,
            # instead of a later plain generation no-op'ing past a "nothing
            # loaded" lora_id straight into the just-trained adapter, or a
            # later explicit lora load calling add_lora against a decoder
            # that's already PEFT-wrapped from training.
            _STATE["handler"] = None
            _STATE["dit_profile"] = None
            _STATE["lora_id"] = None
            _STATE["lora_adapter_path"] = None

    final_dir = adapter_dir / "final"
    if not final_dir.exists() or not any(final_dir.iterdir()):
        raise RuntimeError(
            f"ACE-Step training finished (status: {last_status!r}) but wrote no adapter weights "
            f"to {final_dir}"
        )

    return {
        "id": lora_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_take_count": len(source_paths),
        "base_checkpoint": DIT_CHECKPOINTS[LORA_BASE_DIT_PROFILE],
        "dit_profile": dit_profile,
        "final_step": last_step,
        "final_loss": last_loss,
        "status": last_status,
        "error": None,
    }
