"""The default: you provide the hardware, and the app never spends anything.

Not a stub. It is the shipped behaviour, and the reason the lane is safe to have
at all: an install that never sets LYRE_REMOTE_GPU_PROVISIONER cannot create a
billable resource, whatever else is misconfigured.
"""

from __future__ import annotations

from typing import Any

from server.remote_gpu.provisioners.base import ProvisionerError, Session


class ManualProvisioner:
    id = "manual"
    label = "Self-hosted"

    def configured(self) -> bool:
        return True

    def quote(self) -> dict[str, Any]:
        return {}

    def start(self) -> Session:
        raise ProvisionerError(
            "This install connects to a Remote GPU you run yourself. Start the host and "
            "enter its address and secret, or set LYRE_REMOTE_GPU_PROVISIONER=runpod to let "
            "Lyre rent one. See docs/remote-gpu.md."
        )

    def status(self) -> Session | None:
        return None

    def adopt(self) -> Session | None:
        return None

    def stop(self) -> None:
        return None
