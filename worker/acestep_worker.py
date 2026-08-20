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
offload for `quality` (XL) is `offload_to_cpu`, not `cpu_offload`;
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
(SPEC.md sec 4.4). `GenerationConfig` carries `batch_size` and
`audio_format="wav"` (ACE-Step defaults to FLAC otherwise, which this
adapter must not silently relabel as `.wav`); `generate_music` takes both
plus `dit_handler` (not `handler`)
and `lm_handler`. Its `GenerationResult` reports `success` plus generated
files in `audios` (dicts) and LM/CoT-filled metadata in `extra_outputs`;
the actual seed used lives nested at `audio["params"]["seed"]`, not
top-level. Every acestep call below is wrapped so a further API drift
raises `WorkerUnavailable` (job `error`, not a crash) instead of an
unhandled exception; `tests/test_acestep_worker_adapter.py` exercises this
whole call contract against fake acestep modules so a mismatch is caught
without needing CUDA installed.
"""

from __future__ import annotations

import inspect
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One GPU occupant: one loaded DiT + one LM, jobs serialize (SPEC.md sec 4.3).
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"dit_profile": None, "handler": None, "lm": None}

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

# SPEC.md sec 6: weights live under checkpoints/<name>/ (gitignored). This is
# `project_root`; `config_path` is just the bare checkpoint name above, not a
# filesystem path under it.
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
        checkpoint = DIT_CHECKPOINTS[dit_profile]
        cpu_offload = dit_profile in CPU_OFFLOAD_PROFILES
        if cpu_offload:
            ok, reason = supports_dit_profile(dit_profile)
            if not ok:
                raise WorkerUnavailable(reason)

        handler = _api_call("AceStepHandler()", AceStepHandler)
        init_kwargs: dict[str, Any] = {
            "project_root": str(CHECKPOINTS_ROOT),
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


def run_job(
    job: dict[str, Any], plan: dict[str, Any], take_id: str, take_dir: Path
) -> tuple[dict, dict | None]:
    """Run one real ACE-Step job. Returns `(take_meta, plan_patch)`;
    `plan_patch` is a delta of the LM-filled fields to persist when this was
    a simple-mode generation, else None (SPEC.md sec 7.2: "Persist the
    filled plan."). `server.jobs` merges it onto the plan that's current on
    disk when the job finishes, not the stale snapshot passed in as `plan`.

    `job["src_audio"]` must already be a resolved, jailed filesystem path (or
    None for `generate`) -- `server.jobs` resolves `source_take_id` /
    `upload_path` before calling this, so the worker never has to reach back
    into project storage itself.

    Raises `WorkerUnavailable` if acestep/CUDA isn't usable or its API no
    longer matches this adapter; `server.jobs` catches that and marks the
    job `error` without crashing the HTTP process.
    """
    _, GenerationParams, GenerationConfig, _, generate_music, create_sample = _import_acestep()
    take_dir.mkdir(parents=True, exist_ok=True)

    simple_mode = _is_simple_mode(job, plan)

    with _LOCK:
        handler, lm, dit_profile = _ensure_loaded(job["dit_profile"])

        effective_plan = _plan_from_query(create_sample, lm, plan) if simple_mode else plan
        # Null/omitted bpm, key, or duration: let ACE-Step's CoT fill them in
        # (SPEC.md sec 7.2, use_cot_metas).
        use_cot_metas = simple_mode or any(
            effective_plan.get(k) is None for k in ("bpm", "keyscale", "duration_sec")
        )

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
            thinking=simple_mode,
            use_cot_metas=use_cot_metas,
            seed=job.get("seed", -1),
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
        "score": None,
        "error": None,
        "repaint": _repaint_meta(job),
        "track_name": job.get("track_name"),
    }
    return meta, plan_patch
