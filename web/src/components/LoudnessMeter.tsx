import { useEffect, useRef, useState } from "react";

// Small live peak meter for a take's <audio> element, built directly on the
// Web Audio API. Self-contained on purpose (SPEC.md sec 12 Phase 6): it owns
// its own AudioContext/AnalyserNode and never threads Web Audio state
// through App's state -- this is a leaf UI feature with no interaction with
// jobs, plans, or take selection.
export function LoudnessMeter({ audioEl }: { audioEl: HTMLAudioElement | null }) {
  const [peak, setPeak] = useState(0);
  const graphRef = useRef<{
    ctx: AudioContext;
    analyser: AnalyserNode;
    data: Uint8Array<ArrayBuffer>;
  } | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!audioEl) return;
    // Nested function declarations below close over `audioEl`, and TS can't
    // prove it's still non-null by the time they run -- capture the
    // narrowed value once, up front.
    const el = audioEl;

    function stopLoop() {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setPeak(0);
    }

    function tick() {
      const graph = graphRef.current;
      if (!graph) return;
      graph.analyser.getByteTimeDomainData(graph.data);
      let max = 0;
      for (let i = 0; i < graph.data.length; i++) {
        const sample = Math.abs(graph.data[i] - 128) / 128;
        if (sample > max) max = sample;
      }
      setPeak(max);
      rafRef.current = requestAnimationFrame(tick);
    }

    function handlePlay() {
      // A given <audio> element can only ever be handed to one
      // MediaElementAudioSourceNode for its lifetime (the browser throws on
      // a second attempt), so the graph is built lazily here, once, on
      // first play -- not eagerly for every take up front.
      if (!graphRef.current) {
        const AudioContextCtor =
          window.AudioContext ??
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const ctx = new AudioContextCtor();
        const source = ctx.createMediaElementSource(el);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        // Passive tap only: source -> analyser -> destination. Must still
        // forward to destination (otherwise playback goes silent) and must
        // not insert any gain/compression/filter node of its own -- SPEC.md
        // sec 4.3 rules out an extra mastering chain in v1.
        source.connect(analyser);
        analyser.connect(ctx.destination);
        graphRef.current = { ctx, analyser, data: new Uint8Array(analyser.fftSize) };
      }
      if (graphRef.current.ctx.state === "suspended") {
        graphRef.current.ctx.resume();
      }
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(tick);
      }
    }

    el.addEventListener("play", handlePlay);
    el.addEventListener("pause", stopLoop);
    el.addEventListener("ended", stopLoop);

    return () => {
      el.removeEventListener("play", handlePlay);
      el.removeEventListener("pause", stopLoop);
      el.removeEventListener("ended", stopLoop);
      stopLoop();
      // Tear the graph down whenever the underlying element changes (or
      // this meter unmounts) rather than leaving a stray AudioContext
      // running for a take that's no longer on screen.
      if (graphRef.current) {
        graphRef.current.ctx.close().catch(() => {});
        graphRef.current = null;
      }
    };
  }, [audioEl]);

  const pct = Math.round(peak * 100);
  return (
    <span className="loudness-meter" aria-hidden="true" title="live peak level">
      <span className="loudness-meter-fill" style={{ width: `${pct}%` }} />
    </span>
  );
}
