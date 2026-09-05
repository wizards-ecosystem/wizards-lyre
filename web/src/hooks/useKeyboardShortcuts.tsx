import { useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import { ProjectDetail } from "../api";

/**
 * Studio keyboard shortcuts (SPEC.md sec 12 Phase 5).
 *
 * Kept as a hook rather than inline in App so the text-entry guard -- the
 * rule that decides which shortcuts are safe to fire while the user is
 * typing in the caption/lyrics/query fields -- lives in one readable place.
 * These are the only place the shortcuts are documented outside web/README.
 */
export function useKeyboardShortcuts({
  activeId,
  detail,
  selectedTakeId,
  busy,
  audioRefs,
  wavesurferRef,
  flushPendingPlanSave,
  generate,
  toggleWaveformPlayback,
  setSelectedTakeId,
  setErrorMsg,
}: {
  activeId: string | null;
  detail: ProjectDetail | null;
  selectedTakeId: string | null;
  busy: boolean;
  audioRefs: React.MutableRefObject<Record<string, HTMLAudioElement | null>>;
  wavesurferRef: React.MutableRefObject<WaveSurfer | null>;
  flushPendingPlanSave: () => Promise<void>;
  generate: () => void;
  toggleWaveformPlayback: () => void;
  setSelectedTakeId: (id: string) => void;
  setErrorMsg: (message: string) => void;
}): void {
  // Keyboard shortcuts (SPEC.md sec 12 Phase 5). Gated on a project being
  // open (there's nothing to act on otherwise). Save is exempt from the
  // text-entry guard below -- it's the one shortcut users need most while
  // actually typing in the caption/lyrics/query fields, and Ctrl/Cmd+S is
  // never a literal character those fields would otherwise receive.
  // Generate / play-pause / prev-next-take *are* guarded, since "g" and
  // Space are ordinary characters those same fields need to accept normally.
  useEffect(() => {
    function isTextEntryFocused(): boolean {
      const tag = document.activeElement?.tagName;
      return tag === "INPUT" || tag === "TEXTAREA";
    }

    function onKeyDown(event: KeyboardEvent) {
      if (!activeId || !detail) return;

      const key = event.key;

      if ((key === "s" || key === "S") && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        flushPendingPlanSave().catch((err) => setErrorMsg(String(err)));
        return;
      }

      if (isTextEntryFocused()) return;

      if (key === "g" || key === "G") {
        if (!busy) generate();
        return;
      }

      if (key === " " || event.code === "Space") {
        event.preventDefault();
        const take = detail.takes.find((t) => t.id === selectedTakeId);
        if (!take || take.error) return;
        if (typeof wavesurferRef.current?.playPause === "function") {
          toggleWaveformPlayback();
          return;
        }
        // Test/legacy fallback for a waveform adapter without playPause.
        const audio = audioRefs.current[take.id];
        if (!audio) return;
        if (audio.paused) audio.play();
        else audio.pause();
        return;
      }

      if (key === "ArrowDown" || key === "ArrowUp") {
        if (detail.takes.length === 0) return;
        const currentIndex = detail.takes.findIndex((t) => t.id === selectedTakeId);
        // Newest-first order (server already sorts it that way) -- Down
        // moves toward older takes, Up toward newer ones. Clamp at the ends
        // rather than wrap so repeated presses can't silently loop back
        // around onto a take the user already stepped past.
        let nextIndex: number;
        if (currentIndex === -1) {
          nextIndex = 0;
        } else if (key === "ArrowDown") {
          nextIndex = Math.min(currentIndex + 1, detail.takes.length - 1);
        } else {
          nextIndex = Math.max(currentIndex - 1, 0);
        }
        event.preventDefault();
        setSelectedTakeId(detail.takes[nextIndex].id);
        return;
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // The handler is re-created whenever the state it reads changes. The
    // callbacks and refs omitted here are stable for a given render and adding
    // them would re-bind the window listener on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, detail, selectedTakeId, busy]);
}
