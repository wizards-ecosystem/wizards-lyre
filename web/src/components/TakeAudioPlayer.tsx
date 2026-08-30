import { useCallback, useState } from "react";
import { api } from "../api";
import { formatClock } from "../lib/format";
import { Icon } from "./Icon";
import { LoudnessMeter } from "./LoudnessMeter";

// Wraps a take's <audio> element together with its LoudnessMeter. A
// dedicated component (rather than inlining hooks into the takes .map)
// keeps the ref callback identity stable across unrelated re-renders of the
// list, so the underlying DOM node -- and the AudioContext tied to it --
// isn't torn down and rebuilt every time App re-renders.
export function TakeAudioPlayer({
  projectId,
  takeId,
  registerRef,
  onSelect,
}: {
  projectId: string;
  takeId: string;
  registerRef: (id: string, el: HTMLAudioElement | null) => void;
  onSelect: () => void;
}) {
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const setRef = useCallback(
    (el: HTMLAudioElement | null) => {
      registerRef(takeId, el);
      setAudioEl(el);
    },
    [takeId, registerRef],
  );

  return (
    <span className="take-audio-player">
      <button
        type="button"
        className="transport-button transport-button-small"
        aria-label={`${playing ? "Pause" : "Play"} take ${takeId}`}
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
          if (!audioEl) return;
          if (audioEl.paused) audioEl.play().catch(() => {});
          else audioEl.pause();
        }}
      >
        <Icon name={playing ? "pause" : "play"} />
      </button>
      <audio
        src={api.takeAudioUrl(projectId, takeId)}
        ref={setRef}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        onDurationChange={(event) => setDuration(event.currentTarget.duration || 0)}
      />
      <span className="transport-time" aria-label={`${formatClock(currentTime)} elapsed`}>
        {formatClock(currentTime)}
      </span>
      <input
        className="take-scrubber"
        type="range"
        min={0}
        max={Math.max(duration, 0.01)}
        step={0.1}
        value={Math.min(currentTime, Math.max(duration, 0.01))}
        aria-label={`Seek take ${takeId}`}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => {
          event.stopPropagation();
          if (!audioEl) return;
          const next = Number(event.currentTarget.value);
          audioEl.currentTime = next;
          setCurrentTime(next);
        }}
      />
      <LoudnessMeter audioEl={audioEl} />
    </span>
  );
}
