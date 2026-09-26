"""The Remote GPU host: Lyre's ACE-Step worker behind an authenticated HTTP API.

Runs on hardware you rent or own (see docs/remote-gpu.md), never beside the
local server. It wraps the same `worker.acestep_worker` adapter the local
worker uses, so a take rendered remotely went through exactly the code a local
one would have. The local side of the connection is `worker.remote_backend`.

Importing this package must stay cheap and GPU-free: the local backend reads
`build_id()` from it to tell whether a deployed host runs older code than the
checkout talking to it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_WORKER_DIR = Path(__file__).resolve().parent.parent

# Everything that decides how a remote take is rendered. The host's own
# protocol code and the ACE-Step adapter it wraps both count; the local-only
# modules (run_worker, mock_worker, remote_backend) do not.
_BUILD_SOURCES = ("remote_host", "acestep_worker")


def build_id() -> str:
    """Fingerprint of the sources a deployed host runs.

    `/health` reports it and the local app compares it with its own, so a
    stale image becomes a visible warning instead of behaviour that quietly
    contradicts the checkout. Hashes source files in a fixed order and nothing
    else, so it never depends on configuration and can never carry a secret.
    """
    digest = hashlib.sha256()
    for package in _BUILD_SOURCES:
        for path in sorted((_WORKER_DIR / package).glob("*.py")):
            digest.update(f"{package}/{path.name}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
