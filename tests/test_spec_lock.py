"""Bootstrap tests: lock SPEC.md and forbid cloud music clients. No GPU."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORTS = (
    "google.genai",
    "google.generativeai",
    "elevenlabs",
    "stability_sdk",
    "suno",
    "udio",
)

CODE_GLOBS = ("**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.mjs")
SKIP_PARTS = {".venv", "node_modules", ".git"}


def _iter_source() -> list[Path]:
    files: list[Path] = []
    for glob in CODE_GLOBS:
        for path in ROOT.glob(glob):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def test_spec_exists_and_forbids_cloud_engines() -> None:
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    assert "ACE-Step 1.5" in spec
    assert "127.0.0.1" in spec
    assert "Lyria" in spec
    assert "no adapters" in spec.lower() or "No Lyria" in spec or "no Lyria" in spec


def test_source_does_not_import_forbidden_engines() -> None:
    pattern = re.compile(
        r"^\s*(?:from|import)\s+("
        + "|".join(re.escape(name) for name in FORBIDDEN_IMPORTS)
        + r")\b",
        re.MULTILINE,
    )
    hits: list[str] = []
    for path in _iter_source():
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []
