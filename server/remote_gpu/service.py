"""What the HTTP routes do with the Remote GPU lane.

One invariant: **nothing here rents hardware without a user asking.** There is
no timer, no start-on-launch, and no "provision because a job is queued".

Status joins two sources on purpose. The provisioner knows whether hardware
exists; the host's `/health` knows whether ACE-Step on it can take work. "Ready"
to a user means both, and reporting either alone is how a UI claims a GPU is
usable while its models are still downloading.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from server.remote_gpu import client, connection, provisioners
from server.remote_gpu.provisioners.base import READY, STARTING, ProvisionerError, Session
from worker.remote_host import build_id

logger = logging.getLogger("lyre.remote_gpu")


class Conflict(ProvisionerError):
    """The request makes sense, just not in the current state."""


def _host(conn: connection.Connection | None) -> dict[str, Any] | None:
    if conn is None:
        return None
    health = client.probe(conn)
    if not health.get("connected"):
        return {"connected": False, "ready": False, "error": health.get("error")}
    build = health.get("build")
    return {
        "connected": True,
        "ready": bool(health.get("ready")),
        "message": health.get("message"),
        "gpu": health.get("gpu"),
        "build": build,
        "stale_build": build != build_id(),
        "loaded_dit_profile": health.get("loaded_dit_profile"),
        "disk_free_gb": health.get("disk_free_gb"),
        "error": None,
    }


def _session(session: Session | None, host: dict[str, Any] | None) -> dict[str, Any] | None:
    if session is None:
        return None
    state, detail = session.state, session.detail
    if state == READY and not (host and host.get("ready")):
        # The pod is up but ACE-Step on it is not: first boot downloads models.
        state = STARTING
        detail = (host or {}).get("message") or "pod is up; waiting for the host to answer"
    now = time.time()
    return {
        "id": session.id,
        "state": state,
        "detail": detail,
        "gpu": session.gpu,
        "elapsed_sec": round(session.elapsed_sec(now), 1),
        "hourly_usd": session.hourly_usd,
        "cost_estimate_usd": session.cost_estimate_usd(now),
    }


def status(session: Session | None = None, *, refresh: bool = True) -> dict[str, Any]:
    provisioner = provisioners.get_provisioner()
    if refresh:
        session = provisioner.status()
    conn = connection.get_connection()
    host = _host(conn)
    can_provision = provisioner.id != "manual" and provisioner.configured()
    return {
        "provisioner": provisioner.id,
        "label": provisioner.label,
        "can_provision": can_provision,
        "stop_on_exit": provisioners.stop_on_exit(),
        "quote": provisioner.quote() if can_provision and session is None else None,
        "connection": {"base_url": conn.base_url} if conn is not None else None,
        "session": _session(session, host),
        "host": host,
    }


def _running() -> Session | None:
    return provisioners.get_provisioner().status()


def connect(base_url: str, secret: str) -> dict[str, Any]:
    if _running() is not None:
        raise Conflict("A rented Remote GPU is running; stop it before connecting another host.")
    connection.set_connection(base_url, secret)
    client.forget_probe()
    return status()


def disconnect() -> dict[str, Any]:
    if _running() is not None:
        raise Conflict("A rented Remote GPU is running; stop it to disconnect.")
    connection.clear_connection()
    client.forget_probe()
    return status()


def start() -> dict[str, Any]:
    """Rent hardware. Billable, and only ever from an explicit user action."""
    session = provisioners.get_provisioner().start()
    client.forget_probe()
    return status(session, refresh=False)


def stop() -> dict[str, Any]:
    provisioners.get_provisioner().stop()
    client.forget_probe()
    return status()


def on_startup() -> None:
    """Say loudly if a previous run left a rental up. Never starts one."""
    try:
        session = provisioners.get_provisioner().adopt()
    except ProvisionerError as exc:
        logger.warning("could not check for a running Remote GPU: %s", exc)
        return
    if session is None:
        return
    cost = session.cost_estimate_usd()
    spend = f", about ${cost:.2f} so far" if cost is not None else ""
    logger.warning(
        "a Remote GPU from an earlier run is still up: %s pod %s, running %.0f min%s. "
        "Stop it from the Remote GPU panel if you are not using it.",
        session.provisioner,
        session.id,
        session.elapsed_sec() / 60,
        spend,
    )


def on_shutdown() -> None:
    """Terminate a rental on a clean exit, when configured to (the default). A
    crash runs no shutdown code, which is why on_startup reports survivors."""
    if not provisioners.stop_on_exit():
        return
    provisioner = provisioners.get_provisioner()
    try:
        if provisioner.adopt() is not None:
            provisioner.stop()
            logger.warning("terminated the Remote GPU on shutdown")
    except ProvisionerError as exc:
        logger.error("could not terminate the Remote GPU on shutdown: %s", exc)
