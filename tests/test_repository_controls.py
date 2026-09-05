"""GitHub contribution controls should fail closed when repository files drift."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
ACTION_REFERENCE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)@([^\s#]+)", re.MULTILINE)
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def test_workflows_use_minimal_tokens_and_immutable_actions() -> None:
    problems: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if "\npermissions:\n" not in text:
            problems.append(f"{workflow.name}: missing top-level permissions")
        if "pull_request_target:" in text:
            problems.append(f"{workflow.name}: pull_request_target is prohibited")
        if "timeout-minutes:" not in text:
            problems.append(f"{workflow.name}: jobs need a timeout")
        for action, revision in ACTION_REFERENCE.findall(text):
            if action.startswith("./"):
                continue
            if FULL_SHA.fullmatch(revision) is None:
                problems.append(f"{workflow.name}: {action}@{revision} is not pinned")

    assert problems == []


def test_checkout_does_not_persist_workflow_credentials() -> None:
    problems: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        lines = workflow.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "uses: actions/checkout@" not in line:
                continue
            block = "\n".join(lines[index + 1 : index + 5])
            if "persist-credentials: false" not in block:
                problems.append(f"{workflow.name}:{index + 1}")

    assert problems == []


def test_repository_has_review_and_security_entry_points() -> None:
    assert (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").endswith("* @limbwizard\n")
    for relative in (
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "SECURITY.md",
        "SUPPORT.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/dependabot.yml",
    ):
        assert (ROOT / relative).is_file(), relative
