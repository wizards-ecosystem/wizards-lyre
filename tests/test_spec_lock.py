"""Bootstrap tests: lock SPEC.md and forbid cloud music clients. No GPU."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exact "import X" / "from X import ..." module names. Kept narrow (dotted
# module identifiers only) so this list can safely contain words that also
# appear as prose in SPEC.md or as plain keywords in FORBIDDEN_KEYWORDS below
# without the two checks colliding with each other.
FORBIDDEN_IMPORTS = (
    "google.genai",
    "google.generativeai",
    "elevenlabs",
    "stability_sdk",
    "suno",
    "udio",
    "gradio",
    "magenta",
)

# Looser "does this file mention a forbidden client/engine at all" scan.
# Matched with letter-boundaries (not \b) so underscore-joined identifiers
# like "udio_client" are still caught while legitimate tokens that merely
# contain the keyword as a sub-string of a larger word (e.g. "studio_ops",
# "gradient") are left alone.
FORBIDDEN_KEYWORDS = (
    "elevenlabs",
    "stabilityai",
    "stability_sdk",
    "suno",
    "udio",
    "lyria",
    "magenta",
    "levo",
    "songgeneration",
    "yue",
    "gemini",
    "gradio",
    "genai",
    "generativeai",
)

# Literal public bind hosts. 127.0.0.1 (see SPEC.md "Bind: 127.0.0.1 only")
# is deliberately not in this list.
FORBIDDEN_BIND_HOSTS = ("0.0.0.0",)

CODE_GLOBS = ("**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.mjs")
SKIP_PARTS = {".venv", "node_modules", ".git"}


def _iter_source(*, skip_tests: bool = False) -> list[Path]:
    files: list[Path] = []
    for glob in CODE_GLOBS:
        for path in ROOT.glob(glob):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            if skip_tests and "tests" in path.parts:
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


def test_source_does_not_reference_forbidden_clients() -> None:
    """Broader scan of application source (excludes tests/, since test files
    legitimately hold these names as literals to assert their absence
    elsewhere). Catches non-import references too, e.g. gradio app mounting
    (`mount_gradio_app`, `gr.Blocks(...).launch()`), REST clients built
    around a forbidden vendor's API, or a stray `import suno_client`.
    """
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(name) for name in FORBIDDEN_KEYWORDS) + r")(?![A-Za-z])",
        re.IGNORECASE,
    )
    hits: list[str] = []
    for path in _iter_source(skip_tests=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            hits.append(f"{path.relative_to(ROOT)}: {match.group(1)!r}")
    assert hits == []


def test_source_does_not_bind_public_host() -> None:
    """Bard server entrypoints must not default to a public bind host.
    127.0.0.1 is fine; 0.0.0.0 (all interfaces) is not.
    """
    pattern = re.compile("|".join(re.escape(host) for host in FORBIDDEN_BIND_HOSTS))
    hits: list[str] = []
    for path in _iter_source(skip_tests=True):
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []
