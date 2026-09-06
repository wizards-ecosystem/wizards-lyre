"""`train_lora`: turn 8+ selected takes into a style pack (SPEC.md sec 4.4).

Wraps ACE-Step's DatasetBuilder -> preprocess_to_tensors -> LoRATrainer
pipeline. Training targets the base (non-distilled) checkpoint --
`settings.LORA_BASE_DIT_PROFILE` -- because the turbo checkpoints are
distilled few-step models the trainer cannot run a full diffusion loop
against.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worker.acestep_worker.api import (
    _api_call,
    _api_method_call,
    _import_lora_training,
)
from worker.acestep_worker.errors import WorkerUnavailable
from worker.acestep_worker.loading import _ensure_loaded
from worker.acestep_worker.settings import (
    _LOCK,
    _STATE,
    DIT_CHECKPOINTS,
    LORA_ALPHA,
    LORA_BASE_DIT_PROFILE,
    LORA_DROPOUT,
    LORA_GRADIENT_ACCUMULATION_STEPS,
    LORA_LEARNING_RATE,
    LORA_RANK,
    LORA_SAVE_EVERY_N_EPOCHS,
    LORA_TRAIN_EPOCHS,
    MIN_LORA_SOURCES,
)


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
        # ACE-Step removes the audio extension before appending the sidecar
        # suffix: `song.wav` -> `song.caption.txt` (not
        # `song.wav.caption.txt`). Keep this exact convention in sync with
        # `acestep.training.dataset_builder_modules.audio_io.load_caption_file`.
        dest.with_suffix(".caption.txt").write_text(name, encoding="utf-8")

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
                    "LYRE_WORKER=mock."
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
                # Zero labeled out of a directory ACE-Step could see and count
                # is a decoder failure, not a data problem: its dataset builder
                # reads audio through torchcodec, which needs FFmpeg's shared
                # libraries present on the system. Generation does not fail the
                # same way -- it falls back to soundfile -- so a machine can
                # generate perfectly well and still label nothing here, and the
                # raw count alone sends people looking at their takes instead
                # of at their system packages.
                hint = ""
                if labeled_count == 0 and samples:
                    hint = (
                        " -- labeling every file failed, which usually means FFmpeg's shared "
                        "libraries are missing (ACE-Step decodes training audio through "
                        "torchcodec). Install FFmpeg (e.g. `sudo apt install ffmpeg`) and "
                        "retry; the worker log will show a libavutil load error if that is it."
                    )
                raise RuntimeError(
                    f"ACE-Step only labeled {labeled_count}/{len(samples)} staged source files "
                    f"(needs >= {MIN_LORA_SOURCES}): {scan_status}{hint}"
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
            # only runs while this is iterated. ACE-Step's trainer writes a
            # PEFT adapter one level below its advertised final directory:
            # <adapter_dir>/final/adapter/. We normalize that upstream layout
            # to Lyre's stable <adapter_dir>/final/ contract below.
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
    upstream_adapter_dir = final_dir / "adapter"
    if not (final_dir / "adapter_config.json").is_file() and upstream_adapter_dir.is_dir():
        # `save_lora_weights()` calls PEFT's save_pretrained() under an
        # additional `adapter/` directory. Keep that ACE-Step implementation
        # detail inside this adapter: every other Lyre layer continues to use
        # <lora_dir>/adapter/final as the stable inference path.
        for artifact in upstream_adapter_dir.iterdir():
            destination = final_dir / artifact.name
            if destination.exists():
                raise RuntimeError(
                    "ACE-Step wrote conflicting final LoRA artifacts at "
                    f"{artifact} and {destination}"
                )
            artifact.replace(destination)
        upstream_adapter_dir.rmdir()

    if not (final_dir / "adapter_config.json").is_file():
        raise RuntimeError(
            f"ACE-Step training finished (status: {last_status!r}) but wrote no loadable PEFT "
            f"adapter_config.json to {final_dir}"
        )

    return {
        "id": lora_id,
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "source_take_count": len(source_paths),
        "base_checkpoint": DIT_CHECKPOINTS[LORA_BASE_DIT_PROFILE],
        "dit_profile": dit_profile,
        "final_step": last_step,
        "final_loss": last_loss,
        "status": last_status,
        "error": None,
    }
