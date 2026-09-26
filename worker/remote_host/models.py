"""Fetch ACE-Step weights on the host the first time a profile needs them.

A fresh pod has none. The core set (the `iterate` DiT, VAE, text encoder and
planner LM) is fetched at startup; the other profiles are fetched the first
time a job asks for one, so a session that only ever generates never pays for
the base or XL checkpoints. This mirrors `./scripts/lyre models-*` locally and
uses the same upstream downloader.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

from worker.remote_host import runtime

# profile -> the checkpoint directory it needs beyond the core set. `iterate`
# is the core set itself (acestep-download with no --model).
PROFILE_CHECKPOINTS = {
    "iterate": "acestep-v15-turbo",
    "polish": "acestep-v15-sft",
    "studio_ops": "acestep-v15-base",
    "quality": "acestep-v15-xl-turbo",
}
CORE_LM = "acestep-5Hz-lm-1.7B"

_LOCK = threading.Lock()


def _present(name: str, root: Path) -> bool:
    directory = root / name
    return (directory / "model.safetensors").is_file() or (
        directory / "model.safetensors.index.json"
    ).is_file()


def _core_present(root: Path) -> bool:
    return _present(PROFILE_CHECKPOINTS["iterate"], root) and (root / CORE_LM).is_dir()


def _downloader() -> str:
    found = shutil.which("acestep-download")
    if found is None:
        raise RuntimeError(
            "acestep-download is not installed on this host; the Remote GPU image "
            "includes it, so a host started by hand needs ACE-Step installed first"
        )
    return found


def _run(args: list[str]) -> None:
    # Output goes to the host's own log (the pod console), which is where an
    # operator watches a first download.
    result = subprocess.run(args, check=False, env=os.environ.copy())
    if result.returncode != 0:
        command = " ".join(args[1:])
        raise RuntimeError(f"model download failed ({command}): exit {result.returncode}")


def ensure_profile(profile: str, root: Path | None = None) -> None:
    """Make sure the weights `profile` needs are on disk. Serialized, so two
    callers never race one download."""
    root = root or runtime.CHECKPOINTS
    if profile not in PROFILE_CHECKPOINTS:
        raise ValueError(f"unknown dit_profile '{profile}'")
    with _LOCK:
        if not _core_present(root):
            _run([_downloader(), "--dir", str(root)])
        checkpoint = PROFILE_CHECKPOINTS[profile]
        if not _present(checkpoint, root):
            _run([_downloader(), "--dir", str(root), "--model", checkpoint, "--skip-main"])
