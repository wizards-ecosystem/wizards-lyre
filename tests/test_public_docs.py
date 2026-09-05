"""Release-facing documentation should not ship with broken local links."""

from __future__ import annotations

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
