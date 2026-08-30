import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/plugins/regions";
import { api, ProjectDetail } from "../api";
import { REPAINT_REGION_ID, SECTION_REGION_ID_PREFIX } from "../constants";

export interface Region {
  start: number;
  end: number;
}

/**
 * Owns the wavesurfer.js instance for the selected take (SPEC.md sec 9.2).
 *
 * Two kinds of region share one RegionsPlugin: the single ad-hoc repaint
 * selection the user drags, and the plan's persisted section labels. Keeping
 * both in one hook is what lets each side tell its own regions apart -- a new
 * drag must not clear the saved labels, and redrawing the labels must not
 * clear the selection.
 */
export function useWaveform({
  selectedTakeId,
  detail,
  activeIdRef,
  audioRefs,
  setErrorMsg,
}: {
  selectedTakeId: string | null;
  detail: ProjectDetail | null;
  activeIdRef: React.MutableRefObject<string | null>;
  /** The per-take <audio> elements, paused when waveform playback starts so
   *  two sources never play at once. */
  audioRefs: React.MutableRefObject<Record<string, HTMLAudioElement | null>>;
  setErrorMsg: (message: string) => void;
}) {
  const [region, setRegion] = useState<Region | null>(null);
  const [waveformPlaying, setWaveformPlaying] = useState(false);
  const [waveformCurrentTime, setWaveformCurrentTime] = useState(0);
  const [waveformDuration, setWaveformDuration] = useState(0);
  const [waveformMediaEl, setWaveformMediaEl] = useState<HTMLAudioElement | null>(null);

  const waveformContainerRef = useRef<HTMLDivElement | null>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);

  // A newly selected take has no region yet -- drop whatever was drawn for
  // the previous one instead of showing a stale start/end that no longer
  // corresponds to anything on screen.
  useEffect(() => {
    // Deliberate: a newly selected take has no region, so a stale start/end
    // must not survive the switch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setRegion(null);
  }, [selectedTakeId]);

  // to null synchronously on every project switch (the effect just above),
  // so by the time this effect fires for a non-null selectedTakeId, the
  // project has already settled.
  useEffect(() => {
    const projectId = activeIdRef.current;
    if (!selectedTakeId || !projectId || !waveformContainerRef.current) return;
    const take = detail?.takes.find((t) => t.id === selectedTakeId);
    if (!take || take.error) return;

    const regions = RegionsPlugin.create();
    const wavesurfer = WaveSurfer.create({
      container: waveformContainerRef.current,
      url: api.takeAudioUrl(projectId, selectedTakeId),
      waveColor: "#565d5a",
      progressColor: "#e6b85c",
      cursorColor: "#f2f0e8",
      height: 164,
      normalize: true,
      plugins: [regions],
    });
    wavesurferRef.current = wavesurfer;
    regionsPluginRef.current = regions;
    // Deliberate: transport state is reset as the new WaveSurfer instance is
    // constructed, so the UI never shows the previous take's position.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setWaveformPlaying(false);
    setWaveformCurrentTime(0);
    setWaveformDuration(0);
    setWaveformMediaEl(
      typeof wavesurfer.getMediaElement === "function"
        ? (wavesurfer.getMediaElement() as HTMLAudioElement)
        : null,
    );

    wavesurfer.on("ready", (duration) => setWaveformDuration(duration));
    wavesurfer.on("timeupdate", (time) => setWaveformCurrentTime(time));
    wavesurfer.on("play", () => setWaveformPlaying(true));
    wavesurfer.on("pause", () => setWaveformPlaying(false));
    wavesurfer.on("finish", () => setWaveformPlaying(false));

    // A missing/corrupt/unsupported take would otherwise just leave a blank
    // waveform with no indication anything went wrong.
    wavesurfer.on("error", (err) => setErrorMsg(`waveform load failed: ${err.message}`));

    // Only one repaint selection at a time (SPEC.md: drag a region ->
    // repaint) -- a new drag replaces the previous selection instead of
    // accumulating. Persisted section labels (SECTION_REGION_ID_PREFIX) live
    // on this same plugin instance and must survive that cleanup, so this
    // only ever removes regions sharing the selection's fixed id. The
    // iteration copies the list first because remove() mutates it in place.
    regions.on("region-created", (created) => {
      if (created.id.startsWith(SECTION_REGION_ID_PREFIX)) return;
      for (const existing of regions.getRegions().slice()) {
        if (existing !== created && existing.id === created.id) existing.remove();
      }
      setRegion({ start: created.start, end: created.end });
    });
    regions.on("region-updated", (updated) => {
      // Section labels are drag/resize-disabled, but guard anyway: only the
      // repaint selection feeds the region state used by Repaint/Lego.
      if (updated.id !== REPAINT_REGION_ID) return;
      setRegion({ start: updated.start, end: updated.end });
    });
    regions.enableDragSelection({ id: REPAINT_REGION_ID });

    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
      regionsPluginRef.current = null;
      setWaveformMediaEl(null);
      setWaveformPlaying(false);
    };
    // The instance is remounted per selected take only. detail/activeIdRef are
    // read at mount; remounting on every detail change would restart playback.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTakeId]);

  // Renders plan.sections as labeled, non-editable regions on the waveform
  // (SPEC.md sec 7.2: sections are "region labels on the waveform") and
  // keeps them in sync with the Plan pane's section list -- editing, adding,
  // or deleting a section there redraws its label here. addRegion is safe to
  // call before the audio finishes decoding: the plugin defers positioning
  // until ready. Declared after the mount effect above so that on a take
  // switch effects run in order and the fresh RegionsPlugin already exists
  // when labels are (re)drawn.
  useEffect(() => {
    const regions = regionsPluginRef.current;
    if (!regions) return;
    // Copy before iterating: remove() mutates the plugin's region list.
    for (const existing of regions.getRegions().slice()) {
      if (existing.id.startsWith(SECTION_REGION_ID_PREFIX)) existing.remove();
    }
    for (const [index, section] of (detail?.plan.sections ?? []).entries()) {
      regions.addRegion({
        id: `${SECTION_REGION_ID_PREFIX}${index}`,
        start: section.start_sec,
        // A section whose end was typed before its start in the Plan pane
        // would otherwise render with zero/negative width and vanish.
        end: Math.max(section.end_sec, section.start_sec),
        content: section.name,
        // Accent tint so labels are visually distinct from the repaint
        // selection's default gray; drag/resize off -- these are labels,
        // edited via the Plan pane, not on the waveform.
        color: "rgba(230, 184, 92, 0.16)",
        drag: false,
        resize: false,
      });
    }
  }, [detail?.plan.sections, selectedTakeId]);

  function clearRegion() {
    // Remove only the repaint selection -- persisted section labels share
    // this RegionsPlugin instance and must survive clearing it.
    for (const existing of regionsPluginRef.current?.getRegions().slice() ?? []) {
      if (existing.id === REPAINT_REGION_ID) existing.remove();
    }
    setRegion(null);
  }

  function toggleWaveformPlayback(): void {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer) return;
    for (const audio of Object.values(audioRefs.current)) {
      audio?.pause();
    }
    if (typeof wavesurfer.playPause !== "function") return;
    wavesurfer.playPause().catch((err) => setErrorMsg(`playback failed: ${String(err)}`));
  }

  /** Seek to an absolute position in seconds, moving the cursor with it. */
  function seekWaveform(seconds: number): void {
    if (waveformDuration > 0) wavesurferRef.current?.seekTo(seconds / waveformDuration);
    setWaveformCurrentTime(seconds);
  }

  return {
    region,
    setRegion,
    clearRegion,
    waveformPlaying,
    waveformCurrentTime,
    waveformDuration,
    waveformMediaEl,
    waveformContainerRef,
    wavesurferRef,
    toggleWaveformPlayback,
    seekWaveform,
  };
}
