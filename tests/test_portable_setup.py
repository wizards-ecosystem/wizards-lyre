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

    # What this guard is for: catching code, config, or the launcher still
    # *reading* a pre-rename name. Two categories are exempt because naming
    # the old variables there is correct, not a leftover:
    #
    #   - Markdown, as a class. The changelog, the configuration reference,
    #     and the contributor guide all have to name BARD_* to explain the
    #     rename and the database migration to anyone upgrading. Listing
    #     individual files here instead just rots the moment a doc is added.
    #   - The launcher and this module, which implement and test the one-time
    #     projects/bard.db -> projects/lyre.db migration.
    #
    # vendor/ is upstream ACE-Step and not Lyre's to rename.
    exempt = {"scripts/lyre", "tests/test_portable_setup.py"}

    hits: list[str] = []
    for name in tracked:
        path = ROOT / name
        if (
            name.startswith("vendor/")
            or name in exempt
            or name.endswith(".md")
            or not path.is_file()
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in ("BARD_", "bard.db"):
            if token in text:
                hits.append(f"{name}: {token!r}")
    assert hits == []
