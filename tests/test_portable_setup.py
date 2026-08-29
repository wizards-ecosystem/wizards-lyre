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
    assert 'BARD_CHECKPOINTS_DIR="$ROOT/checkpoints"' in source
    for model in (
        "acestep-v15-sft",
        "acestep-v15-base",
        "acestep-v15-xl-turbo",
    ):
        assert f"--model {model} --skip-main" in source
