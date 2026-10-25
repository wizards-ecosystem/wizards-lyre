import { useCallback, useEffect, useRef } from "react";

/**
 * One playback owner for the whole studio (SPEC.md sec 16.6): whichever
 * source starts playing -- the waveform, a take row, the library preview, a
 * compare slot -- pauses every other one, so two tracks never play by
 * accident.
 *
 * Media elements in the document are found on each `play` (a capture
 * listener sees every element's play event). Elements outside the document,
 * such as the waveform library's own media element, are registered.
 */
export function usePlaybackOwner(): { register: (el: HTMLMediaElement | null) => void } {
  const detached = useRef<Set<HTMLMediaElement>>(new Set());
  const watchRef = useRef<((el: HTMLMediaElement) => void) | null>(null);

  useEffect(() => {
    function onPlay(event: Event) {
      const started = event.target;
      if (!(started instanceof HTMLMediaElement)) return;
      const others = new Set<HTMLMediaElement>(detached.current);
      document.querySelectorAll("audio, video").forEach((el) => others.add(el as HTMLMediaElement));
      for (const el of others) {
        if (el !== started && !el.paused) el.pause();
      }
    }
    document.addEventListener("play", onPlay, true);
    const registered = detached.current;
    // A detached element's play event does not reach the document.
    const listeners = new Map<HTMLMediaElement, EventListener>();
    function watch(el: HTMLMediaElement) {
      if (listeners.has(el)) return;
      const listener: EventListener = (event) => onPlay(event);
      el.addEventListener("play", listener);
      listeners.set(el, listener);
    }
    registered.forEach(watch);
    watchRef.current = watch;
    return () => {
      document.removeEventListener("play", onPlay, true);
      listeners.forEach((listener, el) => el.removeEventListener("play", listener));
      watchRef.current = null;
    };
  }, []);

  const register = useCallback((el: HTMLMediaElement | null) => {
    if (!el || el.isConnected) return;
    detached.current.add(el);
    watchRef.current?.(el);
  }, []);

  return { register };
}

/** Clamp a playhead carried from one take to another of a different length
 *  (A/B switching). Stays a moment short of the end so the target does not
 *  finish the instant it starts. */
export function clampPlayhead(time: number, duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(0, Math.min(time, Math.max(duration - 0.05, 0)));
}
