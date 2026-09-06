"""Loading, swapping, and unloading the GPU occupant (SPEC.md sec 4.3).

One DiT plus one LM at a time; switching profiles unloads the previous DiT
before loading the next. `_ensure_lora_adapter` layers a trained style pack
onto the loaded handler (SPEC.md sec 4.4).
"""

from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import Any

from worker.acestep_worker import settings
from worker.acestep_worker.api import (
    _api_call,
    _api_method_call,
    _check_init_result,
    _check_lora_status,
    _import_acestep,
)
from worker.acestep_worker.errors import WorkerUnavailable
from worker.acestep_worker.settings import (
    _LOCK,
    _STATE,
    CPU_OFFLOAD_PROFILES,
    DEFAULT_LM,
    DEVICE,
    DIT_CHECKPOINTS,
    LM_BACKEND,
)

# settings.CHECKPOINTS_ROOT is read through the module rather than bound by name: it
# is the one setting tests repoint, and a name binding here would be taken
# at import time and never see the change.


def _checkpoints_project_root() -> Path:
    """The `project_root` to pass to `AceStepHandler.initialize_service`
    (SPEC.md sec 6 / module docstring): upstream resolves each DiT checkpoint
    at `<project_root>/checkpoints/<config_path>`, so `project_root` must be
    `settings.CHECKPOINTS_ROOT`'s *parent* for that to land on `settings.CHECKPOINTS_ROOT`
    itself -- which only lines up if `settings.CHECKPOINTS_ROOT`'s own directory name
    is literally `checkpoints` (true for the SPEC-locked default; required of
    any `LYRE_CHECKPOINTS_DIR` override too, since ACE-Step's `checkpoints/`
    segment is fixed, not something this adapter can rename around). Raises
    `WorkerUnavailable` instead of silently resolving to the wrong directory
    (exactly the bug the reviewer flagged)."""
    if settings.CHECKPOINTS_ROOT.name != "checkpoints":
        raise WorkerUnavailable(
            f"LYRE_CHECKPOINTS_DIR ('{settings.CHECKPOINTS_ROOT}') must be a directory named "
            "'checkpoints': AceStepHandler.initialize_service resolves DiT checkpoints "
            "at <project_root>/checkpoints/<config_path>, and this adapter derives "
            "project_root as settings.CHECKPOINTS_ROOT's parent so that lands on the real "
            "weights directory."
        )
    return settings.CHECKPOINTS_ROOT.parent


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
            f"{free_bytes / (1024**3):.1f} GiB free / {total_bytes / (1024**3):.1f} GiB total"
        )
    except Exception as exc:
        return f"CUDA detect: error while querying CUDA/VRAM: {exc}"


def _release_accelerator_memory() -> None:
    """Collect abandoned model/trainer cycles before constructing a DiT.

    ACE-Step's LoRA trainer wraps the decoder in PEFT and Fabric objects that
    can form reference cycles. Clearing `_STATE["handler"]` makes the old
    handler unreachable, but reference counting alone does not necessarily
    free its CUDA allocations before the next job builds another handler. A
    full collection followed by the allocator cache release keeps the
    one-DiT invariant true in physical VRAM as well as in `_STATE`.

    Torch stays a lazy optional import: ordinary tests and the server process
    must remain GPU-free, and cleanup must never turn an otherwise valid load
    into a failure merely because an accelerator runtime cannot be queried.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass


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
    except Exception as exc:
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
        _release_accelerator_memory()
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
            checkpoint_dir=str(settings.CHECKPOINTS_ROOT),
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
