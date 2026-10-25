import { useEffect, useState } from "react";
import { api, Capabilities, Health } from "../api";
import { HEALTH_POLL_INTERVAL_MS } from "../constants";

/**
 * Polls `GET /api/health` and `GET /api/capabilities` on a fixed interval.
 * Capabilities (SPEC.md sec 16.6) are best-effort: a server that does not
 * answer them is still reported online by health.
 *
 * Owns no other studio state, so it is a standalone hook: the server (and
 * worker) may not be running at all, and this reports that as `error` rather
 * than raising it into the app's generic error banner.
 */
export function useHealth(): {
  health: Health | null;
  capabilities: Capabilities | null;
  error: string | null;
} {
  const [health, setHealth] = useState<Health | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Server (and worker) may not be running at all -- a fetch failure here
  // is a normal, expected state (shown as "offline"), not something to
  // surface via the generic errorMsg banner.
  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const result = await api.health();
        if (!cancelled) {
          setHealth(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null);
          setError(String(err));
        }
        return;
      }
      try {
        const report = await api.capabilities();
        if (!cancelled) setCapabilities(report);
      } catch {
        if (!cancelled) setCapabilities(null);
      }
    }
    poll();
    const interval = setInterval(poll, HEALTH_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { health, capabilities, error };
}
