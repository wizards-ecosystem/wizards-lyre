"""Thin shims over upstream ACE-Step's Python API.

Imports are lazy and every call is wrapped, so a missing install, a moved
symbol, or a renamed argument surfaces as `WorkerUnavailable` naming the step
that failed -- not as an opaque ImportError or TypeError from deep inside
generation (SPEC.md sec 13).
"""

from __future__ import annotations

from typing import Any

from worker.acestep_worker.errors import WorkerUnavailable


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
            "and download weights (SPEC.md sec 4 and 13), or set LYRE_WORKER=mock "
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
            "4.4/13), or set LYRE_WORKER=mock for local dev/tests without a GPU."
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
            "match the installed acestep version, or set LYRE_WORKER=mock."
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
            "match the installed acestep version, or set LYRE_WORKER=mock."
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
