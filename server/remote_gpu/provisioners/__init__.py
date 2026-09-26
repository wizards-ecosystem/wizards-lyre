"""Provisioners: how Remote GPU hardware comes to exist.

Adding one is a module implementing `base.Provisioner` plus a row in
`_REGISTRY`. The rest of Lyre only ever learns a base URL and a secret.

The default is `manual`, and an unknown or unloadable name falls back to it
rather than failing: a provisioner is an optional convenience, and no
misconfiguration of one may stop a local-first app from starting.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from server.remote_gpu.provisioners.base import Provisioner
from server.remote_gpu.provisioners.manual import ManualProvisioner

logger = logging.getLogger("lyre.remote_gpu")


def _runpod() -> Provisioner:
    from server.remote_gpu.provisioners.runpod import RunpodProvisioner

    return RunpodProvisioner()


_REGISTRY: dict[str, Callable[[], Provisioner]] = {
    "manual": ManualProvisioner,
    "runpod": _runpod,
}
_CACHE: dict[str, Provisioner] = {}


def available() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_provisioner() -> Provisioner:
    name = os.environ.get("LYRE_REMOTE_GPU_PROVISIONER", "manual").strip().lower() or "manual"
    if name not in _REGISTRY:
        logger.warning("unknown LYRE_REMOTE_GPU_PROVISIONER %r; using manual", name)
        name = "manual"
    if name not in _CACHE:
        try:
            _CACHE[name] = _REGISTRY[name]()
        except Exception as exc:  # a broken provisioner must not stop the app
            logger.warning("provisioner %s could not be loaded (%s); using manual", name, exc)
            return ManualProvisioner()
    return _CACHE[name]


def stop_on_exit() -> bool:
    raw = os.environ.get("LYRE_REMOTE_GPU_STOP_ON_EXIT", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}
