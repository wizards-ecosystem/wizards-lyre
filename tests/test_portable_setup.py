"""Portable launcher regression checks. These must stay GPU- and network-free."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "lyre"


def test_portable_launcher_is_executable_and_reports_repo_local_paths() -> None:
    assert LAUNCHER.is_file()
    assert LAUNCHER.stat().st_mode & 0o111

    syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run([str(LAUNCHER), "paths"], capture_output=True, text=True, check=True)
    paths = dict(line.split(":", 1) for line in result.stdout.splitlines() if ":" in line)

    assert Path(paths["root"].strip()) == ROOT
    assert Path(paths["venv"].strip()) == ROOT / ".venv"
    assert Path(paths["ACE-Step"].strip()) == ROOT / "vendor" / "ACE-Step-1.5"
    assert Path(paths["models"].strip()) == ROOT / "checkpoints"
    assert Path(paths["projects"].strip()) == ROOT / "projects"
    assert Path(paths["output"].strip()) == ROOT / "output"
    assert Path(paths["caches"].strip()) == ROOT / ".cache"
    assert Path(paths["config"].strip()) == ROOT / ".config"
    assert Path(paths["temporary"].strip()) == ROOT / ".tmp"


def test_portable_setup_pins_ace_step_and_downloads_all_lyre_profiles() -> None:
    revision = (ROOT / "ACE_STEP_REVISION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", revision)

    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'ACESTEP_CHECKPOINTS_DIR="$ROOT/checkpoints"' in source
    assert 'LYRE_CHECKPOINTS_DIR="$ROOT/checkpoints"' in source
    for model in (
        "acestep-v15-sft",
        "acestep-v15-base",
        "acestep-v15-xl-turbo",
    ):
        assert f"--model {model} --skip-main" in source


def test_no_legacy_project_name_survives() -> None:
    """Lyre's runtime contract is LYRE_* and its database is lyre.db.

    The project carried an earlier name before its first release; nothing in
    this repository should still say it. A half-finished rename is easy to
    miss by eye and leaves a silently ignored setting behind, so scan every
    tracked file.

    Matched with letter boundaries (not \\b) so `LYRE_`-style underscore joins
    and `.db` suffixes are still caught, while an ordinary word that merely
    contains the letters is not. Generated lock files are skipped: their
    content hashes are effectively random text and are not ours to rename.
    vendor/ is upstream ACE-Step, likewise not ours.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()

    pattern = re.compile(r"(?<![A-Za-z])bard(?![A-Za-z])", re.IGNORECASE)
    skip_suffixes = ("uv.lock", "package-lock.json")

    hits: list[str] = []
    for name in tracked:
        path = ROOT / name
        if (
            name.startswith("vendor/")
            or name.endswith(skip_suffixes)
            or name == "tests/test_portable_setup.py"
            or not path.is_file()
        ):
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            hits.append(name)
    assert hits == []
