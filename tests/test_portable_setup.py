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


def test_portable_setup_migrates_the_pre_rename_database() -> None:
    """The 0.1.0 rename moved projects/bard.db to projects/lyre.db. An
    existing checkout must keep its library, so the launcher renames the
    legacy file (plus its SQLite sidecars and worker lock) once, and never
    over an existing lyre.db.
    """
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "migrate_legacy_database" in source
    assert 'legacy="$ROOT/projects/bard.db"' in source
    assert 'current="$ROOT/projects/lyre.db"' in source
    assert 'if [[ ! -f "$legacy" || -e "$current" ]]; then' in source
    for suffix in ("-wal", "-shm", ".worker.lock"):
        assert suffix in source


def test_no_pre_rename_environment_variables_survive() -> None:
    """Lyre's runtime contract is LYRE_* (SPEC.md sec 5); the pre-0.1.0
    BARD_* names were dropped outright, with no fallback. A half-finished
    rename is easy to miss by eye and produces a silently ignored setting,
    so fail loudly on any surviving token in tracked source.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()

    # vendor/ is upstream ACE-Step, not Lyre's to rename. The launcher owns
    # the one-time bard.db migration and this module tests it, so both
    # legitimately name the old file. Prose docs may recount the history.
    exempt = {
        "scripts/lyre",
        "tests/test_portable_setup.py",
        "SPEC.md",
        "README.md",
        "CHANGELOG.md",
    }

    hits: list[str] = []
    for name in tracked:
        path = ROOT / name
        if name.startswith("vendor/") or name in exempt or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("BARD_", "bard.db"):
            if token in text:
                hits.append(f"{name}: {token!r}")
    assert hits == []
