"""Real ACE-Step 1.5 worker (SPEC.md sec 4 and sec 10).

This is the production job backend (`server.jobs` selects it whenever
`LYRE_WORKER` is unset or `acestep`). `worker.mock_worker` is the test/local
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
ACE-Step's current API and keep Lyre's HTTP schema stable with an
adapter"): `AceStepHandler`/`LLMHandler` are constructed with no arguments.
`AceStepHandler` is loaded via `initialize_service(project_root=...,
config_path=<checkpoint name>, device=..., offload_to_cpu=...)` -- CPU
offload for `quality` (XL) is `offload_to_cpu`, not `cpu_offload`. Upstream
resolves the DiT checkpoint at `<project_root>/checkpoints/<config_path>`,
*not* at `<project_root>/<config_path>` -- `project_root` and the checkpoint
directory are two different things to ACE-Step, even though Lyre only has
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
language under `language`, which this adapter maps onto Lyre's own
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
on `GenerationParams` (SPEC.md sec 4.3: "default on", always requested
explicitly rather than trusting whatever ACE-Step's own default happens to
be; this field lives on params, not `GenerationConfig`); `generate_music`
takes both plus `dit_handler` (not `handler`)
and `llm_handler`. Its `GenerationResult` reports `success` plus generated
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

This package is the single import surface -- `from worker import
acestep_worker`, then `acestep_worker.<name>` -- split by concern:

- `errors`   -- WorkerUnavailable
- `settings` -- checkpoints, per-profile generation settings, LoRA defaults
- `api`      -- lazy imports and wrapped calls into upstream ACE-Step
- `loading`  -- the GPU occupant: load, swap, unload, attach a LoRA
- `results`  -- reading plan fill-in, quality score, and lyric timestamps back
- `run`      -- run_job
- `lora`     -- train_lora
"""

from __future__ import annotations

from worker.acestep_worker.errors import WorkerUnavailable
from worker.acestep_worker.loading import (
    _checkpoints_project_root,
    _ensure_loaded,
    _ensure_lora_adapter,
    _handler_supports_cpu_offload,
    _log_cuda_status,
    get_loaded_dit_profile,
    initialize_worker,
    supports_dit_profile,
)
from worker.acestep_worker.lora import train_lora
from worker.acestep_worker.run import run_job

# CHECKPOINTS_ROOT is deliberately *not* re-exported here. It is the one
# setting that gets repointed (LYRE_CHECKPOINTS_DIR, and tests), and a second
# binding at package level would silently go stale against settings'. Read it
# as `acestep_worker.settings.CHECKPOINTS_ROOT`.
from worker.acestep_worker.settings import (
    _LOCK,
    _STATE,
    CPU_OFFLOAD_PROFILES,
    DEFAULT_LM,
    DEVICE,
    DIT_CHECKPOINTS,
    GENERATION_GUIDANCE_SCALE,
    GENERATION_STEPS,
    LM_BACKEND,
    LORA_BASE_DIT_PROFILE,
    MIN_LORA_SOURCES,
    TASK_TYPE_BY_ACTION,
    TRACK_INSTRUCTION_ACTIONS,
)

__all__ = [
    "CPU_OFFLOAD_PROFILES",
    "DEFAULT_LM",
    "DEVICE",
    "DIT_CHECKPOINTS",
    "GENERATION_GUIDANCE_SCALE",
    "GENERATION_STEPS",
    "LM_BACKEND",
    "LORA_BASE_DIT_PROFILE",
    "MIN_LORA_SOURCES",
    "TASK_TYPE_BY_ACTION",
    "TRACK_INSTRUCTION_ACTIONS",
    "_LOCK",
    "_STATE",
    "WorkerUnavailable",
    "_checkpoints_project_root",
    "_ensure_loaded",
    "_ensure_lora_adapter",
    "_handler_supports_cpu_offload",
    "_log_cuda_status",
    "get_loaded_dit_profile",
    "initialize_worker",
    "run_job",
    "supports_dit_profile",
    "train_lora",
]
