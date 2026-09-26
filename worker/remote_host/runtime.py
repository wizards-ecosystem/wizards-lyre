"""Where a Remote GPU host keeps its files, and how it is told who may call it.

Import this before anything that reads cache or checkpoint locations at import
time (`worker.acestep_worker.settings`, torch, huggingface_hub). It points all
of them under one root so a pod can keep its weights on a volume that outlives
the container, and so nothing lands in a rented machine's home directory.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_PORT = 8000
# Loopback unless told otherwise. The container image sets LYRE_REMOTE_HOST so
# the provider's proxy can reach it; a host run by hand stays private.
DEFAULT_HOST = "127.0.0.1"

_ROOT_OVERRIDE = os.environ.get("LYRE_REMOTE_ROOT", "").strip()
ROOT = (
    Path(_ROOT_OVERRIDE).expanduser().resolve()
    if _ROOT_OVERRIDE
    else Path.cwd() / ".lyre-remote-gpu"
)
# ACE-Step resolves DiT weights as <parent>/checkpoints/<name>, so the
# directory must be named `checkpoints` (see docs/CONFIGURATION.md).
CHECKPOINTS = ROOT / "checkpoints"
JOBS = ROOT / "jobs"
CACHE = ROOT / "cache"
TMP = ROOT / "tmp"

# Secrets that are obviously not secrets. The host refuses to start with one:
# the proxy URL in front of it is public, and this is all that guards the GPU.
_PLACEHOLDER_SECRETS = {
    "",
    "changeme",
    "change-me",
    "secret",
    "password",
    "lyre",
    "test",
    "example",
}
MIN_SECRET_LENGTH = 24


def prepare() -> None:
    """Create the layout and redirect every cache under ROOT. Idempotent."""
    for directory in (CHECKPOINTS, JOBS, CACHE, TMP):
        directory.mkdir(parents=True, exist_ok=True)
    defaults = {
        "LYRE_CHECKPOINTS_DIR": str(CHECKPOINTS),
        "ACESTEP_CHECKPOINTS_DIR": str(CHECKPOINTS),
        "ACESTEP_PROJECT_ROOT": str(ROOT),
        "HF_HOME": str(CACHE / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(CACHE / "huggingface" / "hub"),
        "MODELSCOPE_CACHE": str(CACHE / "modelscope"),
        "TORCH_HOME": str(CACHE / "torch"),
        "TRITON_CACHE_DIR": str(CACHE / "triton"),
        "XDG_CACHE_HOME": str(CACHE),
        "TMPDIR": str(TMP),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def shared_secret() -> str:
    """The secret every request must carry, or a ValueError saying why not."""
    secret = os.environ.get("LYRE_REMOTE_SECRET", "").strip()
    if secret.lower() in _PLACEHOLDER_SECRETS or len(secret) < MIN_SECRET_LENGTH:
        raise ValueError(
            "LYRE_REMOTE_SECRET must be set to a random value of at least "
            f"{MIN_SECRET_LENGTH} characters (try `openssl rand -hex 32`). The address "
            "in front of this host is public; the secret is what keeps others off the GPU."
        )
    return secret


def host() -> str:
    return os.environ.get("LYRE_REMOTE_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST


def port() -> int:
    return int(os.environ.get("LYRE_REMOTE_PORT", DEFAULT_PORT))


def backend_name() -> str:
    """`acestep` in production. `mock` renders silence with no GPU, which is
    how the protocol is tested and how a deployment can be smoke-tested cheaply."""
    return os.environ.get("LYRE_REMOTE_BACKEND", "acestep").strip() or "acestep"


def free_disk_gb(path: Path = ROOT) -> float:
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return 0.0
