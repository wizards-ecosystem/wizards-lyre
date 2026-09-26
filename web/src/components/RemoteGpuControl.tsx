import { FormEvent, useEffect, useState } from "react";
import { api, RemoteGpuStatus } from "../api";
import { HEALTH_POLL_INTERVAL_MS } from "../constants";
import { ConfirmationRequest } from "../types";
import { ConfirmationDialog } from "./ConfirmationDialog";

// The opt-in Remote GPU (docs/remote-gpu.md): ACE-Step on hardware the user
// rents or runs elsewhere. Self-contained, like useHealth: it polls its own
// route, and an install that never configures the lane just shows "Remote GPU".
//
// Money rule: the only thing that rents hardware is the Start button, and it
// asks first, showing the rate. Nothing here starts a GPU on its own.

function errorDetail(err: unknown): string {
  const text = String(err instanceof Error ? err.message : err);
  const json = text.slice(text.indexOf("{"));
  try {
    const parsed = JSON.parse(json) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    // not a FastAPI error body; show it as-is
  }
  return text;
}

function money(usd: number | null | undefined): string | null {
  return usd == null ? null : `$${usd.toFixed(2)}`;
}

function duration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

function summary(status: RemoteGpuStatus | null): { label: string; tone: string } {
  if (!status) return { label: "Remote GPU", tone: "off" };
  const { session, host, connection } = status;
  if (session) {
    const cost = money(session.cost_estimate_usd);
    const state = session.state === "ready" ? "GPU ready" : `GPU ${session.state}`;
    return { label: cost ? `${state} · ${cost}` : state, tone: session.state };
  }
  if (connection) {
    if (host?.ready) return { label: "Remote GPU ready", tone: "ready" };
    return host?.connected
      ? { label: "Remote GPU starting", tone: "starting" }
      : { label: "Remote GPU offline", tone: "error" };
  }
  return { label: "Remote GPU", tone: "off" };
}

export function RemoteGpuControl() {
  const [status, setStatus] = useState<RemoteGpuStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseUrl, setBaseUrl] = useState("");
  const [secret, setSecret] = useState("");
  const [confirmation, setConfirmation] = useState<ConfirmationRequest | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const result = await api.remoteGpu();
        if (!cancelled) setStatus(result);
      } catch {
        // Server offline: the health badge already says so.
      }
    }
    poll();
    const interval = setInterval(poll, HEALTH_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  async function act(action: () => Promise<RemoteGpuStatus>) {
    setBusy(true);
    setError(null);
    try {
      setStatus(await action());
    } catch (err) {
      setError(errorDetail(err));
    } finally {
      setBusy(false);
    }
  }

  function confirmStart() {
    const quote = status?.quote;
    const rate = money(quote?.hourly_usd);
    const what = quote ? `one ${quote.gpu} pod` : "one GPU pod";
    const price = rate ? ` at about ${rate}/hr` : "";
    setConfirmation({
      title: `Rent a GPU on ${status?.label ?? "the provider"}?`,
      message:
        `This starts ${what}${price}, billed to your account until you stop it. ` +
        (status?.stop_on_exit
          ? "Quitting Lyre cleanly also stops it; a crash does not."
          : "It keeps running after Lyre exits until you stop it."),
      confirmLabel: "Start GPU",
      resolve: (accepted) => {
        setConfirmation(null);
        if (accepted) void act(api.startRemoteGpu);
      },
    });
  }

  function connect(event: FormEvent) {
    event.preventDefault();
    void act(() => api.connectRemoteGpu(baseUrl, secret)).then(() => setSecret(""));
  }

  const { label, tone } = summary(status);
  const session = status?.session ?? null;
  const host = status?.host ?? null;

  return (
    <details className="shortcut-help remote-gpu">
      <summary className={`remote-gpu-summary ${tone}`}>{label}</summary>
      <div className="shortcut-popover remote-gpu-popover" aria-live="polite">
        <strong className="remote-gpu-title">Remote GPU</strong>

        {session ? (
          <div className="remote-gpu-session">
            <span>
              {session.gpu || "GPU"} · <em>{session.state}</em>
            </span>
            {session.detail && <small>{session.detail}</small>}
            <small>
              {duration(session.elapsed_sec)}
              {money(session.hourly_usd) && ` at ${money(session.hourly_usd)}/hr`}
              {money(session.cost_estimate_usd) &&
                ` · about ${money(session.cost_estimate_usd)} so far`}
            </small>
            <button type="button" disabled={busy} onClick={() => void act(api.stopRemoteGpu)}>
              Stop GPU
            </button>
          </div>
        ) : status?.can_provision ? (
          <div className="remote-gpu-quote">
            <span>
              {status.quote?.gpu ?? "GPU"} on {status.label}
            </span>
            <small>
              {money(status.quote?.hourly_usd)
                ? `${money(status.quote?.hourly_usd)}/hr, ${status.quote?.cloud.toLowerCase()} cloud`
                : "rate unavailable"}
              {status.quote?.availability && ` · stock ${status.quote.availability.toLowerCase()}`}
            </small>
            <button type="button" disabled={busy} onClick={confirmStart}>
              Start GPU
            </button>
          </div>
        ) : null}

        {status?.connection && !session && (
          <div className="remote-gpu-connection">
            <small>Connected to {status.connection.base_url}</small>
            <button type="button" disabled={busy} onClick={() => void act(api.disconnectRemoteGpu)}>
              Disconnect
            </button>
          </div>
        )}

        {host && (
          <small className={`remote-gpu-host ${host.ready ? "ready" : "waiting"}`}>
            {host.ready
              ? `Host ready: ${host.gpu ?? "GPU"}${host.loaded_dit_profile ? `, ${host.loaded_dit_profile} loaded` : ""}`
              : host.connected
                ? `Host starting: ${host.message ?? "loading ACE-Step"}`
                : `Host unreachable${host.error ? `: ${host.error}` : ""}`}
          </small>
        )}
        {host?.stale_build && (
          <small className="remote-gpu-warning">
            The host runs an older build than this checkout.
          </small>
        )}

        {!session && !status?.connection && (
          <form className="remote-gpu-form" onSubmit={connect}>
            <small>Connect a host you run:</small>
            <input
              aria-label="Remote GPU address"
              placeholder="https://..."
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
            <input
              aria-label="Remote GPU secret"
              type="password"
              placeholder="shared secret"
              autoComplete="off"
              value={secret}
              onChange={(event) => setSecret(event.target.value)}
            />
            <button type="submit" disabled={busy || !baseUrl || !secret}>
              Connect
            </button>
          </form>
        )}

        {error && (
          <small className="remote-gpu-error" role="alert">
            {error}
          </small>
        )}
        <small className="remote-gpu-help">
          Renders need the worker started with <code>LYRE_WORKER=remote</code>. See
          docs/remote-gpu.md.
        </small>
      </div>
      <ConfirmationDialog
        request={confirmation}
        onDecision={(accepted) => confirmation?.resolve(accepted)}
      />
    </details>
  );
}
