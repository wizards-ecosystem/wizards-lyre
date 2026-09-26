"""What a provisioner is: something that can make a Remote GPU host exist.

Distinct from `server.remote_gpu.client`, which talks to a host that already
exists. The default provisioner is `manual`: you run the host yourself and the
app never touches your money.

**Every other implementation spends money**, which shapes the interface:

- `start()` is only ever called from an explicit user action. Nothing may
  provision on a timer, on launch, or because a job is queued.
- a started session is recorded before `start()` returns, so a crash between
  "created" and "recorded" cannot orphan a billing resource.
- `adopt()` exists for the same reason: on startup the app asks what it already
  owns, because the previous process may have died holding something.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProvisionerError(RuntimeError):
    """A provisioner could not do what was asked. The message reaches the user,
    so it says what to do about it and never contains a key or secret."""


class AlreadyRunning(ProvisionerError):
    """A second start while one rental is up. Refused rather than billed twice."""


# A coarse state machine rather than a percentage: an image pull, a model
# download and a model load take very different times, and a percentage across
# them would lie.
STARTING = "starting"
READY = "ready"
STOPPING = "stopping"
ERROR = "error"


@dataclass(frozen=True)
class Session:
    """One running rental, as the app understands it."""

    id: str
    provisioner: str
    state: str = STARTING
    base_url: str = ""
    detail: str = ""
    started_at: float = 0.0
    hourly_usd: float | None = None
    gpu: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def elapsed_sec(self, now: float | None = None) -> float:
        if not self.started_at:
            return 0.0
        return max(0.0, (now if now is not None else time.time()) - self.started_at)

    def cost_estimate_usd(self, now: float | None = None) -> float | None:
        """Rough compute spend so far; None when the rate is unknown, never a
        guess. The provider bills from its own clock, and storage is extra."""
        if self.hourly_usd is None:
            return None
        return round(self.hourly_usd * self.elapsed_sec(now) / 3600.0, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provisioner": self.provisioner,
            "state": self.state,
            "base_url": self.base_url,
            "detail": self.detail,
            "started_at": self.started_at,
            "hourly_usd": self.hourly_usd,
            "gpu": self.gpu,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=str(data.get("id", "")),
            provisioner=str(data.get("provisioner", "")),
            state=str(data.get("state", STARTING)),
            base_url=str(data.get("base_url", "")),
            detail=str(data.get("detail", "")),
            started_at=float(data.get("started_at") or 0.0),
            hourly_usd=(float(data["hourly_usd"]) if data.get("hourly_usd") is not None else None),
            gpu=str(data.get("gpu", "")),
            extra=dict(data.get("extra") or {}),
        )


class Provisioner(Protocol):
    """Small on purpose: a new provider should be an afternoon, not a project."""

    id: str
    label: str

    def configured(self) -> bool:
        """Whether there is enough configuration to try. No network calls."""
        ...

    def quote(self) -> dict[str, Any]:
        """What a start would rent, and its advertised rate, before anything is
        created, so the UI can show the price before the button is pressed."""
        ...

    def start(self) -> Session:
        """Create hardware and return as soon as it has an id. Never blocks
        until ready; the caller polls `status()` to show progress."""
        ...

    def status(self) -> Session | None:
        """The session this provisioner owns, refreshed. None when none is."""
        ...

    def stop(self) -> None:
        """Release the hardware. Safe to call when nothing is running, because
        shutdown calls it without checking first."""
        ...

    def adopt(self) -> Session | None:
        """Re-attach to a session a previous process left behind."""
        ...
