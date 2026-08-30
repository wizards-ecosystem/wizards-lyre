"""Check this adapter's calls against the *installed* ACE-Step, not against
what we believe it looks like.

`tests/test_acestep_worker_adapter.py` pins the adapter's calling convention
against fake `acestep.*` modules, which is what lets the suite run without a
GPU. But a fake encodes a belief, and a wrong belief passes: the adapter called
`create_sample(lm_handler=...)` while upstream spells it `llm_handler`, and
because the fake agreed, every test was green while every real simple-mode
generate raised TypeError.

This module closes that loop. It introspects the real signatures and asserts
the adapter's keywords exist on them. It is skipped entirely when ACE-Step is
not installed, so CI and a GPU-free contributor setup are unaffected -- it runs
on the machine that has the GPU stack, which is the only place the question can
actually be answered.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

acestep_inference = pytest.importorskip(
    "acestep.inference", reason="ACE-Step is not installed; nothing to conform to"
)
acestep_handler = pytest.importorskip("acestep.handler")
acestep_llm = pytest.importorskip("acestep.llm_inference")


def _parameters(func: Any) -> set[str]:
    signature = inspect.signature(func)
    names = set(signature.parameters) - {"self"}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        # **kwargs accepts anything, so nothing can be proven absent.
        pytest.skip(f"{func!r} takes **kwargs; its keywords cannot be checked")
    return names


def _field_names(cls: Any) -> set[str]:
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return _parameters(cls)


# What worker/acestep_worker actually passes. Kept as literals rather than
# introspected from the adapter: the point is to state the contract in one
# readable place and have the installed package confirm it.
EXPECTED_KEYWORDS: dict[str, tuple[Any, set[str]]] = {
    "AceStepHandler.initialize_service": (
        acestep_handler.AceStepHandler.initialize_service,
        {"project_root", "config_path", "device", "offload_to_cpu"},
    ),
    "LLMHandler.initialize": (
        acestep_llm.LLMHandler.initialize,
        {"checkpoint_dir", "lm_model_path", "backend", "device"},
    ),
    "create_sample": (
        acestep_inference.create_sample,
        {"llm_handler", "query", "instrumental", "vocal_language"},
    ),
    "generate_music": (
        acestep_inference.generate_music,
        {"dit_handler", "llm_handler", "params", "config", "save_dir"},
    ),
}

EXPECTED_FIELDS: dict[str, tuple[Any, set[str]]] = {
    "GenerationParams": (
        acestep_inference.GenerationParams,
        {
            "task_type",
            "caption",
            "lyrics",
            "bpm",
            "keyscale",
            "duration",
            "instrumental",
            "vocal_language",
            "timesignature",
            "thinking",
            "use_cot_metas",
            "seed",
            "inference_steps",
            "guidance_scale",
            "instruction",
            "src_audio",
            "audio_cover_strength",
            "repainting_start",
            "repainting_end",
            "enable_normalization",
        },
    ),
    "GenerationConfig": (
        acestep_inference.GenerationConfig,
        {"batch_size", "audio_format", "use_random_seed", "seeds"},
    ),
}

# Fields the adapter deliberately does NOT send, because they do not exist
# upstream and passing them raised TypeError on every real call. If one of
# these ever appears, the adapter can start using it -- but silently
# reintroducing it is the regression this guards.
KNOWN_ABSENT_FROM_GENERATION_PARAMS = {"negative_tags", "track_name"}


@pytest.mark.parametrize("label", sorted(EXPECTED_KEYWORDS))
def test_adapter_keywords_exist_on_the_installed_api(label: str) -> None:
    func, expected = EXPECTED_KEYWORDS[label]
    missing = sorted(expected - _parameters(func))
    assert missing == [], (
        f"worker/acestep_worker passes {missing} to {label}, but the installed ACE-Step does "
        f"not accept them. Its signature is: {inspect.signature(func)}"
    )


@pytest.mark.parametrize("label", sorted(EXPECTED_FIELDS))
def test_adapter_fields_exist_on_the_installed_dataclasses(label: str) -> None:
    cls, expected = EXPECTED_FIELDS[label]
    missing = sorted(expected - _field_names(cls))
    assert missing == [], (
        f"worker/acestep_worker sets {missing} on {label}, which the installed ACE-Step does "
        f"not define."
    )


def test_fields_the_adapter_avoids_are_still_absent() -> None:
    present = sorted(
        KNOWN_ABSENT_FROM_GENERATION_PARAMS & _field_names(acestep_inference.GenerationParams)
    )
    if present:
        pytest.skip(
            f"upstream now defines {present} on GenerationParams; the adapter may start "
            "sending them (see its module docstring)"
        )
