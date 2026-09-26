"""Start a Remote GPU host: `python -m worker.remote_host`.

Refuses to start without a real LYRE_REMOTE_SECRET. See docs/remote-gpu.md.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from worker.remote_host import build_id, runtime


def _load_backend() -> ModuleType:
    name = runtime.backend_name()
    if name not in ("acestep", "mock"):
        raise ValueError(f"unknown LYRE_REMOTE_BACKEND '{name}' (use acestep or mock)")
    return importlib.import_module(f"worker.{name}_worker")


def main() -> None:
    # Before anything reads cache or checkpoint locations at import time.
    runtime.prepare()
    try:
        secret = runtime.shared_secret()
    except ValueError as exc:
        print(f"The Wizard's Lyre remote host: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    import uvicorn

    from worker.remote_host import models
    from worker.remote_host.app import Host, create_app

    download = models.ensure_profile if runtime.backend_name() == "acestep" else None
    host = Host(
        _load_backend,
        jobs_dir=runtime.JOBS,
        download=download,
        free_disk_gb=runtime.free_disk_gb,
    )
    host.start()
    print(
        f"The Wizard's Lyre remote host {build_id()} ({runtime.backend_name()}) "
        f"listening on {runtime.host()}:{runtime.port()}; models under {runtime.ROOT}"
    )
    uvicorn.run(create_app(host, secret), host=runtime.host(), port=runtime.port())


if __name__ == "__main__":
    main()
