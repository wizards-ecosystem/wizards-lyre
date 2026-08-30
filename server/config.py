"""Runtime paths and bind settings. No CUDA here. See SPEC.md sec 5, 6, 8."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PORT = 8421
HOST = "127.0.0.1"


def lyre_port() -> int:
    return int(os.environ.get("LYRE_PORT", DEFAULT_PORT))


def projects_dir() -> Path:
    raw = os.environ.get("LYRE_PROJECTS_DIR")
    path = Path(raw) if raw else REPO_ROOT / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def output_dir() -> Path:
    raw = os.environ.get("LYRE_OUTPUT_DIR")
    path = Path(raw) if raw else REPO_ROOT / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def db_path() -> Path:
    raw = os.environ.get("LYRE_DB_PATH")
    if raw:
        path = Path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    return projects_dir() / "lyre.db"
