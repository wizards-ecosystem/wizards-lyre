"""Creation services that touch project storage (SPEC.md sec 16.6).

`server.generation` holds the pure rules; this module applies them to a
stored plan, so `/api`, `/v1`, and MCP share one implementation.
"""

from __future__ import annotations

from typing import Any

from server import storage
from server.generation import presets


def preset_on_project(
    project_id: str,
    preset_id: str,
    *,
    apply: bool,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Preview (`apply=False`) or apply a preset to a project's plan.

    The preview writes nothing. Applying writes the proposed plan through
    the ordinary validated, revision-checked plan write, with changed fields
    marked origin `preset`; a plan that moved on since `expected_revision`
    raises `PlanRevisionConflict` (409) and nothing is written."""
    current = storage.load_plan(project_id)
    proposal = presets.apply_preset(current, preset_id)
    plan = proposal["plan"]
    if apply:
        plan = storage.save_plan(project_id, plan, expected_revision=expected_revision)
    return {
        "plan": plan,
        "changes": proposal["changes"],
        "warnings": proposal["warnings"],
        "preset": proposal["preset"],
        "applied": apply,
    }
