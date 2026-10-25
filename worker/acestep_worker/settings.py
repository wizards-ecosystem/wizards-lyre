"""Checkpoint locations, per-profile generation settings, and LoRA defaults.

Every tunable this adapter has lives here, so the profile table in SPEC.md
sec 4.1 maps onto one file. Readers inside the package go through this module
(`settings.CHECKPOINTS_ROOT`) so a test can repoint it in one place.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

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
# Resolve the default from this source tree, not the caller's working
# directory. That keeps the model store in one portable Lyre checkout even
# if the worker is launched through an absolute module path elsewhere.
CHECKPOINTS_ROOT = Path(
    os.environ.get("LYRE_CHECKPOINTS_DIR", Path(__file__).resolve().parents[2] / "checkpoints")
).resolve()
DEVICE = os.environ.get("LYRE_DEVICE", "cuda")

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

# SPEC.md sec 16.2: how each profile is presented. Stable API ids; these
# describe what the profile actually is, and promise nothing about quality.
PROFILE_INFO = {
    "iterate": {
        "label": "Iterate",
        "summary": "Turbo checkpoint, 8 steps. The fast default for trying ideas.",
    },
    "polish": {
        "label": "Polish",
        "summary": "SFT checkpoint, 50 steps with guidance. Slower; not always better.",
    },
    "quality": {
        "label": "Quality",
        "summary": (
            "XL turbo checkpoint (4B), 8 steps, with CPU offload on a 16 GB card. "
            "Larger and slower to load; needs more system memory."
        ),
    },
    "studio_ops": {
        "label": "Studio",
        "summary": "Base checkpoint, 50 steps. Only for extract, lego, complete, and style packs.",
    },
}

# What a checkpoint directory must hold before it is even worth trying to
# load. Presence is not validation: only a successful load proves the files
# are intact (SPEC.md sec 16.6).
_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")


def checkpoint_status(name: str, root: Path | None = None) -> dict[str, Any]:
    """`present`, `incomplete` (a directory without config or weights), or
    `missing`, from the filesystem alone. Never imports acestep or torch."""
    directory = (root or CHECKPOINTS_ROOT) / name
    if not directory.is_dir():
        return {"name": name, "state": "missing", "path": str(directory)}
    has_config = (directory / "config.json").is_file()
    has_weights = any((directory / weight).is_file() for weight in _WEIGHT_FILES) or any(
        directory.glob("*.safetensors")
    )
    state = "present" if has_config and has_weights else "incomplete"
    return {"name": name, "state": state, "path": str(directory)}


def profile_files(dit_profile: str, root: Path | None = None) -> dict[str, Any]:
    """The worst state among everything a profile needs: its DiT, the 5 Hz
    LM, and the shared VAE."""
    parts = [
        checkpoint_status(DIT_CHECKPOINTS[dit_profile], root),
        checkpoint_status(DEFAULT_LM, root),
        checkpoint_status("vae", root),
    ]
    order = {"missing": 0, "incomplete": 1, "present": 2}
    worst = min(parts, key=lambda part: order[part["state"]])
    return {"state": worst["state"], "parts": parts}


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
