"""Release-facing documentation should not ship with broken local links."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".cache",
    ".data",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".tools",
    ".venv",
    "checkpoints",
    "node_modules",
    "output",
    "projects",
    "vendor",
}

INLINE_LINK = re.compile(r"!?\[[^]]*]\(([^)\s]+)")
REFERENCE_LINK = re.compile(r"^\[[^]]+]:\s+(\S+)", re.MULTILINE)
HTML_LINK = re.compile(r"(?:href|src)=\"([^\"]+)\"")


def _markdown_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        # These two templates are moved to the archive root before their
        # links are resolved; test_release_bundle.py verifies that form.
        and path.name not in {"BUNDLE_README.md", "BUNDLE_NOTICES.md"}
    ]


def test_every_local_documentation_link_resolves() -> None:
    broken: list[str] = []
    for document in _markdown_files():
        text = document.read_text(encoding="utf-8")
        targets = INLINE_LINK.findall(text) + REFERENCE_LINK.findall(text) + HTML_LINK.findall(text)
        for raw_target in targets:
            target = raw_target.strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target} (outside repo)")
                continue
            if not resolved.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert broken == []


def test_readme_uses_the_checked_in_accessible_hero() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    hero = (ROOT / "docs" / "assets" / "lyre-hero.svg").read_text(encoding="utf-8")

    assert 'src="docs/assets/lyre-hero.svg"' in readme
    assert "<title" in hero
    assert "<desc" in hero


def test_every_compiled_web_dependency_has_a_versioned_license_notice() -> None:
    lock = json.loads((ROOT / "web" / "package-lock.json").read_text(encoding="utf-8"))
    notices = (ROOT / "docs" / "WEB_THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")

    runtime_packages = {
        path.rsplit("node_modules/", 1)[-1]: record["version"]
        for path, record in lock["packages"].items()
        if path and not record.get("dev", False)
    }
    assert runtime_packages
    for package, version in runtime_packages.items():
        assert f"| {package} | {version} |" in notices
