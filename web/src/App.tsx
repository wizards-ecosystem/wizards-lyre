import { DragEvent, useCallback, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/plugins/regions";
import { api, Health, Job, Lora, Plan, ProjectDetail, ProjectSummary, Section } from "./api";

const HEALTH_POLL_INTERVAL_MS = 5000;
const JOB_POLL_INTERVAL_MS = 1000;
// Real ACE-Step generation can take a while (SPEC.md sec 5: it queues
// behind a dedicated worker process); give it a generous ceiling before
// giving up on polling rather than declaring failure too early.
const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000;
// train_lora runs the whole training loop under the worker's GPU-exclusive
// lock and can take roughly an hour (SPEC.md sec 4.4) -- far longer than the
// ceiling above that's tuned for ordinary generate/cover/repaint jobs.
const LORA_TRAIN_POLL_TIMEOUT_MS = 90 * 60 * 1000;
// Cadence for the recovery poll that watches a train_lora job discovered via
// GET /api/jobs (i.e. one that outlived a page refresh, or is running in
// another tab). Training itself is hour-scale, so this only needs to be
// prompt enough that completion shows up shortly after it happens.
const LORA_TRAIN_RECOVERY_POLL_MS = 3000;

// SPEC.md sec 4.4 "Style pack | LoRA train / load | 8+ songs" -- mirrors
// server.jobs.MIN_LORA_SOURCE_TAKES so the Train button can disable itself
// before even attempting a request the server would reject.
const MIN_LORA_SOURCE_TAKES = 8;

// Coalesce rapid keystrokes into one PUT instead of firing one per
// keystroke (which can complete out of order and let an older request
// overwrite a newer edit on disk).
const PLAN_SAVE_DEBOUNCE_MS = 500;

// Identities for the two kinds of waveform regions (SPEC.md sec 9.2), which
// share one RegionsPlugin instance: the single ad-hoc repaint selection and
// the persisted plan section labels. The fixed id scheme is what lets each
// side tell its own regions apart -- creating a repaint selection must not
// clear the saved section labels, and re-rendering the labels must not touch
// the selection.
const REPAINT_REGION_ID = "repaint-selection";
const SECTION_REGION_ID_PREFIX = "section-label-";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// A job only finishes async, via server.jobs' queued -> running -> done|error
// lifecycle (SPEC.md sec 5) -- the enqueue response is just the initial
// `queued` row, so the caller has to keep polling /api/jobs/{id} itself.
async function pollJob(
  jobId: string,
  onUpdate?: (job: Job) => void,
  timeoutMs: number = JOB_POLL_TIMEOUT_MS,
): Promise<Job> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await api.getJob(jobId);
    onUpdate?.(job);
    if (job.status === "done" || job.status === "error") {
      return job;
    }
    if (Date.now() > deadline) {
      throw new Error(`job ${jobId} is still ${job.status} after ${timeoutMs / 1000}s`);
    }
    await sleep(JOB_POLL_INTERVAL_MS);
  }
}

// Small live peak meter for a take's <audio> element, built directly on the
// Web Audio API. Self-contained on purpose (SPEC.md sec 12 Phase 6): it owns
// its own AudioContext/AnalyserNode and never threads Web Audio state
// through App's state -- this is a leaf UI feature with no interaction with
// jobs, plans, or take selection.
function LoudnessMeter({ audioEl }: { audioEl: HTMLAudioElement | null }) {
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

// Wraps a take's <audio> element together with its LoudnessMeter. A
// dedicated component (rather than inlining hooks into the takes .map)
// keeps the ref callback identity stable across unrelated re-renders of the
// list, so the underlying DOM node -- and the AudioContext tied to it --
// isn't torn down and rebuilt every time App re-renders.
function TakeAudioPlayer({
  projectId,
  takeId,
  registerRef,
}: {
  projectId: string;
  takeId: string;
  registerRef: (id: string, el: HTMLAudioElement | null) => void;
}) {
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);

  const setRef = useCallback(
    (el: HTMLAudioElement | null) => {
      registerRef(takeId, el);
      setAudioEl(el);
    },
    [takeId, registerRef],
  );

  return (
    <span className="take-audio-player">
      <audio controls src={api.takeAudioUrl(projectId, takeId)} ref={setRef} />
      <LoudnessMeter audioEl={audioEl} />
    </span>
  );
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [librarySearch, setLibrarySearch] = useState("");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [newQuery, setNewQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyStatus, setBusyStatus] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedTakeId, setSelectedTakeId] = useState<string | null>(null);
  const [compareTakeId, setCompareTakeId] = useState<string | null>(null);
  // A dropped local file (SPEC.md sec 12 Phase 6) is an alternative
  // cover/repaint source to a selected take -- server.jobs
  // _resolve_source_audio only ever accepts one or the other. uploadedSourceName
  // is just for display (the server discards the client's original filename).
  const [uploadedSourcePath, setUploadedSourcePath] = useState<string | null>(null);
  const [uploadedSourceName, setUploadedSourceName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [region, setRegion] = useState<{ start: number; end: number } | null>(null);
  const [coverStrength, setCoverStrength] = useState(0.7);
  const [trackName, setTrackName] = useState("");
  const [includeStems, setIncludeStems] = useState(true);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [loras, setLoras] = useState<Lora[]>([]);
  // Multi-select of takes to train a style pack from (SPEC.md sec 4.4 "8+
  // songs") -- deliberately separate from selectedTakeId/compareTakeId,
  // which are single-select take-list concepts used for a different purpose
  // (waveform/cover/repaint source, A/B compare).
  const [loraSourceIds, setLoraSourceIds] = useState<Set<string>>(new Set());
  const [loraName, setLoraName] = useState("");
  // The style pack (if any) to apply to the next generate/cover/repaint
  // (SPEC.md sec 4.4 "LoRA train / load" -- the load half). Separate from
  // loraSourceIds above, which only feeds *training* a new pack.
  const [selectedLoraId, setSelectedLoraId] = useState<string | null>(null);
  // This project's train_lora jobs that are still queued/running, recovered
  // from GET /api/jobs on project load and kept current by a poll below.
  // This is what makes a long training survive a page refresh: the busy
  // state of the session that started it is gone, but the job row in the
  // queue is not, so the Style Packs pane can keep showing its status and
  // refresh the pack list the moment it finishes (no second reload needed).
  const [trainingJobs, setTrainingJobs] = useState<Job[]>([]);

  // Inline rename of the open project from the workspace heading (SPEC.md
  // sec 9). Non-null while the heading's edit input is open, holding the
  // in-progress draft; null shows the plain heading. Deliberately NOT part
  // of the debounced plan-save machinery above: a rename commits as one
  // PATCH /api/projects/{id} on Enter/blur, never one request per
  // keystroke.
  const [titleDraft, setTitleDraft] = useState<string | null>(null);
  // Guards Enter and blur against racing each other into two PATCHes for
  // the same commit (Enter fires it; the input can still blur while the
  // request is in flight).
  const titleSavingRef = useRef(false);

  // Plan saves are debounced and serialized: at most one PUT /plan in
  // flight at a time, always carrying the latest edit. Without this, one
  // PUT per keystroke can complete out of order and let an older request
  // clobber a newer edit on disk even though the UI already shows it.
  //
  // There is only one pending-save slot, shared across projects (not one
  // per project ID), so switching the active project must flush whatever
  // is pending *before* the switch -- otherwise an edit to the project
  // being left can be silently discarded by the next project's edits
  // reusing the same slot/timer.
  //
  // saveChainRef is a promise chain used purely to serialize save
  // *execution order* -- it always resolves, even after a failed save, so
  // one failure doesn't permanently break every save after it.
  // lastSaveOutcomeRef instead reflects the *real* outcome of the most
  // recently started save; flushPendingPlanSave() awaits that (not the
  // chain) so a failed save is never silently treated as success --
  // generate() must not enqueue a job against a plan that failed to save.
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingSaveRef = useRef<{ projectId: string; plan: Plan } | null>(null);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());
  const lastSaveOutcomeRef = useRef<Promise<void>>(Promise.resolve());

  // Take notes debounce, analogous to the plan save mechanism above but
  // keyed by take id (several takes' notes fields can each have their own
  // edit in flight/pending at once, unlike the single shared plan slot).
  const takeSaveTimeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout> | null>>({});
  const pendingTakeNotesRef = useRef<Record<string, string>>({});

  // Mirrors activeId, but updated synchronously (the instant a switch is
  // committed, not after the next render) so refreshDetail can tell whether
  // its response is still for the project actually on screen. Selecting A
  // then B rapidly fires two overlapping GET /api/projects/{id} requests
  // that can resolve in either order; without this check, A's slower
  // response can land after B's and overwrite `detail` with A's data while
  // activeId (and the rest of the UI) is already B -- so a subsequent edit
  // would be saved under B's project id but built from A's plan.
  const activeIdRef = useRef<string | null>(null);

  const waveformContainerRef = useRef<HTMLDivElement | null>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);

  // Keyed by take id rather than a single ref, since every take in the list
  // renders its own <audio> (and a future compare panel could show two at
  // once) -- looking one up via document.querySelector would be ambiguous
  // and non-React-idiomatic.
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({});
  const registerAudioRef = useCallback((id: string, el: HTMLAudioElement | null) => {
    audioRefs.current[id] = el;
  }, []);

  // Reviewer-flagged: canceling a pending take-notes debounce timer (on
  // unmount, or on the page actually closing/reloading) used to just drop
  // the buffered edit on the floor. `pagehide` (fires reliably on
  // close/reload/navigate, including into bfcache -- unlike `beforeunload`,
  // which some browsers skip) and `beforeunload` (kept as a belt-and-braces
  // fallback for engines that don't fire `pagehide` in every case) both
  // flush with `keepalive: true` so the browser completes the request even
  // as the page is torn down instead of aborting it mid-flight.
  useEffect(() => {
    function flushTakeNotesOnUnload() {
      flushAllPendingTakeNotes({ keepalive: true }).catch(() => {});
    }
    window.addEventListener("pagehide", flushTakeNotesOnUnload);
    window.addEventListener("beforeunload", flushTakeNotesOnUnload);
    return () => {
      window.removeEventListener("pagehide", flushTakeNotesOnUnload);
      window.removeEventListener("beforeunload", flushTakeNotesOnUnload);
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      flushTakeNotesOnUnload();
    };
  }, []);

  function enqueueSave(): Promise<void> {
    const runSave = saveChainRef.current.then(async () => {
      const pending = pendingSaveRef.current;
      if (!pending) return;
      // Take ownership of this pending value before the request starts (not
      // after) so a newer edit made while the request is in flight lands in
      // a fresh slot instead of being clobbered when this save resolves.
      pendingSaveRef.current = null;
      try {
        await api.savePlan(pending.projectId, pending.plan);
      } catch (err) {
        // Put the failed edit back so a later flush can retry it -- but
        // only if nothing newer has already claimed the slot, otherwise
        // this would overwrite (and lose) that newer edit.
        if (pendingSaveRef.current === null) {
          pendingSaveRef.current = pending;
        }
        throw err;
      }
    });
    saveChainRef.current = runSave.catch(() => {});
    lastSaveOutcomeRef.current = runSave;
    return runSave;
  }

  // Cancels any pending debounce and waits for the latest edit (plus
  // anything already in flight) to finish saving. generate() and
  // switchActiveProject() must call this first: without it, a job can
  // start (or a project switch can happen) within the debounce window and
  // read/discard the plan from before the user's last edit. Throws if the
  // save actually failed, so callers can abort instead of proceeding
  // against a stale on-disk plan.
  async function flushPendingPlanSave(): Promise<void> {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    if (pendingSaveRef.current) {
      await enqueueSave();
    } else {
      await lastSaveOutcomeRef.current;
    }
  }

  // Flushes any pending/in-flight save for the *current* project before
  // switching to a different one (SPEC.md: a shared save slot must not
  // silently drop an edit when the user switches projects mid-debounce). If
  // that flush fails, the switch must not proceed: activeId must stay put
  // so the failed edit is still visible/retryable instead of being swapped
  // out from under the user or silently discarded when the next project's
  // edits reuse the same pending-save slot. Take notes are keyed by take id
  // rather than project id, but the same reasoning applies -- a pending
  // note for a take in the project being left must not be silently dropped
  // (reviewer-flagged) just because the debounce window hasn't elapsed yet.
  async function switchActiveProject(id: string): Promise<void> {
    try {
      await flushPendingPlanSave();
      await flushAllPendingTakeNotes();
    } catch (err) {
      setErrorMsg(String(err));
      return;
    }
    activeIdRef.current = id;
    setActiveId(id);
  }

  async function refreshProjects() {
    setProjects(await api.listProjects());
  }

  async function toggleFavorite(p: ProjectSummary): Promise<void> {
    try {
      await api.patchProject(p.id, { favorite: !p.favorite });
      await refreshProjects();
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  // Renames the open project from the workspace heading: one PATCH
  // /api/projects/{id} on commit (Enter or blur), nothing per keystroke.
  // On success the server-normalized project from the response replaces
  // detail.project (covers e.g. a whitespace-only draft coming back as
  // 'Untitled', matching storage.create_project), and the library list is
  // refreshed so the sidebar title matches too. On failure the error is
  // surfaced in the banner and the input stays open with the typed value
  // intact, so the edit is never lost.
  async function commitProjectTitle(): Promise<void> {
    if (!activeId || !detail || titleDraft === null || titleSavingRef.current) return;
    const projectId = activeId;
    const draft = titleDraft;
    if (draft === detail.project.title) {
      // Opened the editor but changed nothing -- close it, skip the PATCH.
      setTitleDraft(null);
      return;
    }
    titleSavingRef.current = true;
    try {
      const updated = await api.patchProject(projectId, { title: draft });
      // Stale-response guard (same reasoning as activeIdRef above): if the
      // user switched projects while the PATCH was in flight, keep the new
      // project's detail instead of clobbering it with the old one's.
      setDetail((prev) =>
        prev && prev.project.id === projectId ? { ...prev, project: updated } : prev,
      );
      setTitleDraft(null);
      await refreshProjects();
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      titleSavingRef.current = false;
    }
  }

  // SPEC.md sec 9.1 "Delete (confirm)". window.confirm is the explicit
  // confirmation gate -- deletion is irreversible (server.storage.delete_project
  // rmtrees the project dir), so there's no undo to fall back on.
  async function deleteProject(p: ProjectSummary): Promise<void> {
    if (!window.confirm(`Delete "${p.title}"? This permanently removes its takes and cannot be undone.`)) {
      return;
    }
    try {
      await api.deleteProject(p.id);
    } catch (err) {
      setErrorMsg(String(err));
      return;
    }
    if (activeIdRef.current === p.id) {
      // The project (and anything a debounced save would target) is gone --
      // drop any save/notes work still pending for it rather than let it
      // fire later against a now-404 project id, then clear the open-project
      // state the same way the [activeId] effect does for `activeId === null`.
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = null;
      }
      pendingSaveRef.current = null;
      for (const timeout of Object.values(takeSaveTimeoutsRef.current)) {
        if (timeout) clearTimeout(timeout);
      }
      takeSaveTimeoutsRef.current = {};
      pendingTakeNotesRef.current = {};
      activeIdRef.current = null;
      setActiveId(null);
    }
    await refreshProjects();
  }

  function updateTakeLocal(takeId: string, patch: { favorite?: boolean; notes?: string }): void {
    setDetail((prev) =>
      prev
        ? { ...prev, takes: prev.takes.map((t) => (t.id === takeId ? { ...t, ...patch } : t)) }
        : prev,
    );
  }

  async function toggleTakeFavorite(take: { id: string; favorite: boolean }): Promise<void> {
    if (!activeId) return;
    const favorite = !take.favorite;
    updateTakeLocal(take.id, { favorite });
    try {
      await api.patchTake(activeId, take.id, { favorite });
    } catch (err) {
      updateTakeLocal(take.id, { favorite: take.favorite });
      setErrorMsg(String(err));
    }
  }

  // Sends whatever note edit is pending for `takeId` right now, canceling
  // its debounce timer first -- the single path every take-notes save
  // actually goes through, whether triggered by the debounce firing, a
  // blur, a project switch, or the page unloading. Re-queues the edit on
  // failure (mirroring enqueueSave's failure handling above) so a later
  // flush can retry it, but only if nothing newer has already claimed the
  // slot. `opts.keepalive` is passed straight through to `api.patchTake`
  // for the pagehide/beforeunload case, where a plain fetch would otherwise
  // be aborted mid-flight by the navigation.
  async function flushTakeNotes(takeId: string, opts?: { keepalive?: boolean }): Promise<void> {
    const timeout = takeSaveTimeoutsRef.current[takeId];
    if (timeout) {
      clearTimeout(timeout);
      takeSaveTimeoutsRef.current[takeId] = null;
    }
    if (!(takeId in pendingTakeNotesRef.current)) return;
    const projectId = activeIdRef.current;
    const notes = pendingTakeNotesRef.current[takeId];
    delete pendingTakeNotesRef.current[takeId];
    if (!projectId) return;
    try {
      await api.patchTake(projectId, takeId, { notes }, opts);
    } catch (err) {
      if (!(takeId in pendingTakeNotesRef.current)) {
        pendingTakeNotesRef.current[takeId] = notes;
      }
      throw err;
    }
  }

  // Flushes every take's pending note edit (not just one) -- used before a
  // project switch, and from the pagehide/beforeunload handler below, since
  // either can happen while more than one take's textarea has an unsaved
  // edit in flight.
  async function flushAllPendingTakeNotes(opts?: { keepalive?: boolean }): Promise<void> {
    const takeIds = Object.keys(pendingTakeNotesRef.current);
    await Promise.all(takeIds.map((takeId) => flushTakeNotes(takeId, opts)));
  }

  function saveTakeNotes(takeId: string, notes: string): void {
    if (!activeId) return;
    updateTakeLocal(takeId, { notes });

    pendingTakeNotesRef.current[takeId] = notes;
    const existing = takeSaveTimeoutsRef.current[takeId];
    if (existing) clearTimeout(existing);
    takeSaveTimeoutsRef.current[takeId] = setTimeout(() => {
      flushTakeNotes(takeId).catch((err) => setErrorMsg(String(err)));
    }, PLAN_SAVE_DEBOUNCE_MS);
  }

  async function setActiveTake(takeId: string): Promise<void> {
    if (!activeId) return;
    try {
      await api.setActiveTake(activeId, takeId);
      await refreshDetail(activeId);
      await refreshProjects();
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  // Swapping A/B also changes which take Cover/Repaint/Extract/Lego/Complete
  // act on next, since those all key off selectedTakeId -- that's the
  // "instant swap" SPEC.md sec 12 calls for, not a separate hidden state.
  function swapCompare(): void {
    setSelectedTakeId(compareTakeId);
    setCompareTakeId(selectedTakeId);
  }

  async function refreshDetail(id: string) {
    const data = await api.getProject(id);
    // Discard a response that's no longer for the active project (see
    // activeIdRef above) -- covers both a rapid A->B switch racing this
    // fetch and generate()'s post-job refresh completing after the user
    // has since switched away from the project the job ran in.
    if (activeIdRef.current !== id) return;
    setDetail(data);
  }

  async function refreshLoras(id: string) {
    const data = await api.listLoras(id);
    // Same stale-response guard as refreshDetail above.
    if (activeIdRef.current !== id) return;
    setLoras(data);
  }

  // Rediscovers this project's still-active (queued/running) train_lora job
  // from the shared job queue (SPEC.md sec 8 GET /api/jobs) -- called on
  // project load so a refresh mid-training restores visible progress, and
  // again after a training finishes/fails so the pane matches the queue.
  // active: true returns the project's complete queued/running worklist
  // (no recency truncation server-side), so an older running training can
  // never be pushed out of the result by newer jobs and lost to the
  // recovery.
  async function refreshTrainingJobs(id: string): Promise<void> {
    const jobs = await api.listJobs({ projectId: id, action: "train_lora", active: true });
    // Same stale-response guard as refreshDetail above. The client-side
    // filter is belt-and-braces on top of the server's active filter.
    if (activeIdRef.current !== id) return;
    setTrainingJobs(jobs.filter((j) => j.status === "queued" || j.status === "running"));
  }

  useEffect(() => {
    refreshProjects().catch((err) => setErrorMsg(String(err)));
  }, []);

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
          setHealthError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null);
          setHealthError(String(err));
        }
      }
    }
    poll();
    const interval = setInterval(poll, HEALTH_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    setSelectedTakeId(null);
    setCompareTakeId(null);
    setUploadedSourcePath(null);
    setUploadedSourceName(null);
    setLoraSourceIds(new Set());
    setLoraName("");
    setSelectedLoraId(null);
    setTrainingJobs([]);
    // Drop any in-progress rename from the project being left -- the draft
    // belongs to that project's title and must not appear on the next one.
    setTitleDraft(null);
    if (activeId) {
      refreshDetail(activeId).catch((err) => setErrorMsg(String(err)));
      refreshLoras(activeId).catch((err) => setErrorMsg(String(err)));
      // Recover a style-pack training that is still queued/running for this
      // project (started before a refresh, or in another tab) so the Style
      // Packs pane keeps showing its status.
      refreshTrainingJobs(activeId).catch((err) => setErrorMsg(String(err)));
    } else {
      setDetail(null);
      setLoras([]);
    }
  }, [activeId]);

  // Recovery poll for the train_lora jobs above: while any are active,
  // re-check the job queue and, the moment one completes or fails, refresh
  // the pack list so the selector/pack entries update without another page
  // reload. A failed training also surfaces its job error in the banner --
  // a failure before the pack directory is allocated (e.g. the worker
  // crashing on claim) never produces a pack entry, so this is the only
  // place that error becomes visible. The session that started the training
  // has its own pollJob running too; double-observing the same job is
  // harmless (both just re-list an idempotent queue) and is exactly what
  // makes this survive a refresh, which kills that original poller.
  const trainingJobIds = trainingJobs.map((j) => j.id).join(",");
  useEffect(() => {
    if (!activeId || trainingJobIds === "") return;
    const projectId = activeId;
    const watchedIds = trainingJobIds.split(",");
    let cancelled = false;

    async function tick() {
      let jobs: Job[];
      try {
        // active: true returns the complete queued/running worklist (no
        // recency truncation server-side), so a watched training can never
        // drop out of this list (and out of the pane/poll) just because
        // newer jobs piled up behind it while it runs.
        jobs = await api.listJobs({ projectId, action: "train_lora", active: true });
      } catch {
        return; // transient (server restarting...) -- retry on the next tick
      }
      if (cancelled || activeIdRef.current !== projectId) return;
      const activeIds = new Set(jobs.map((j) => j.id));
      // A watched id that fell out of the active set has finished (done or
      // error). Fetch those by id -- the active-filtered list excludes them
      // by definition, and a recency-limited unfiltered list can drop them
      // too, which would silently lose the finish event and its error.
      const finishedIds = watchedIds.filter((id) => !activeIds.has(id));
      setTrainingJobs(jobs);
      if (finishedIds.length === 0) return;
      const finished: Job[] = [];
      for (const id of finishedIds) {
        try {
          finished.push(await api.getJob(id));
        } catch {
          // No longer fetchable (e.g. the job row vanished with a project
          // deletion) -- nothing left to surface for it.
        }
      }
      if (finished.length === 0 || cancelled || activeIdRef.current !== projectId) return;
      await refreshLoras(projectId);
      // The await above can outlast a project switch -- never surface this
      // project's training failure on top of another project's workspace.
      if (cancelled || activeIdRef.current !== projectId) return;
      for (const job of finished) {
        if (job.status === "error") {
          // The pack entry itself (refreshLoras above) shows up in the
          // list whenever the worker allocated one before failing; this
          // banner is the actionable message either way.
          setErrorMsg(`style pack training failed: ${job.error ?? "unknown error"}`);
        }
      }
    }

    const interval = setInterval(tick, LORA_TRAIN_RECOVERY_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [activeId, trainingJobIds]);

  // A newly selected take has no region yet -- drop whatever was drawn for
  // the previous one instead of showing a stale start/end that no longer
  // corresponds to anything on screen.
  useEffect(() => {
    setRegion(null);
  }, [selectedTakeId]);

  // An uploaded file and a selected take are alternative cover/repaint
  // sources (server.jobs._resolve_source_audio accepts exactly one) --
  // picking a take deselects whatever was dropped. The reverse (dropping a
  // file deselects the take) happens directly in the drop handler.
  useEffect(() => {
    if (selectedTakeId) {
      setUploadedSourcePath(null);
      setUploadedSourceName(null);
    }
  }, [selectedTakeId]);

  // A take picked as A (selectedTakeId) can also already be set as B
  // (compareTakeId) -- e.g. clicking its row while it's mid-comparison, or
  // following a parent-take link onto it. Clear the comparison rather than
  // showing two identical players and a no-op swap.
  useEffect(() => {
    if (compareTakeId && compareTakeId === selectedTakeId) {
      setCompareTakeId(null);
    }
  }, [selectedTakeId, compareTakeId]);

  // Mounts a fresh WaveSurfer instance pointed at the selected take's audio
  // and tears it down on every change (SPEC.md sec 9.2) -- WaveSurfer owns
  // its own <canvas>/audio element inside the container div, so leaving a
  // stale instance running while a new one is created would leak audio
  // decoding work and duplicate canvases. activeIdRef (not activeId) is used
  // for the take URL because it's already kept in sync with the project
  // actually on screen (see its own comment above); selectedTakeId is reset
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
      waveColor: "#7c8cff",
      progressColor: "#4fd67a",
      cursorColor: "#e4e6ec",
      height: 96,
      plugins: [regions],
    });
    wavesurferRef.current = wavesurfer;
    regionsPluginRef.current = regions;

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
    };
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
        color: "rgba(124, 140, 255, 0.15)",
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

  function clearUploadedSource() {
    setUploadedSourcePath(null);
    setUploadedSourceName(null);
    setUploadError(null);
  }

  async function handleDropAudio(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!activeId) return;
    const projectId = activeId;
    const file = event.dataTransfer.files[0];
    if (!file) return;
    setUploadError(null);
    try {
      const { upload_path } = await api.uploadAudio(projectId, file);
      // Discard a response that's no longer for the active project (same
      // activeIdRef race guard as refreshDetail above) -- otherwise a slow
      // upload that resolves after the user has switched projects would
      // populate uploadedSourcePath (and thus a subsequent Cover/Repaint
      // job's upload_path) against the wrong project.
      if (activeIdRef.current !== projectId) return;
      // An uploaded file and a selected take are alternative sources --
      // picking this one deselects whatever take was selected.
      setSelectedTakeId(null);
      setUploadedSourcePath(upload_path);
      setUploadedSourceName(file.name);
    } catch (err) {
      if (activeIdRef.current !== projectId) return;
      setUploadError(String(err));
    }
  }

  async function createProject() {
    setErrorMsg(null);
    try {
      const project = await api.createProject(newTitle || "Untitled", newQuery);
      setNewTitle("");
      setNewQuery("");
      await refreshProjects();
      await switchActiveProject(project.id);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  function savePlanField<K extends keyof Plan>(key: K, value: Plan[K]) {
    if (!activeId || !detail) return;
    const plan = { ...detail.plan, [key]: value };
    setDetail({ ...detail, plan });

    pendingSaveRef.current = { projectId: activeId, plan };
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = setTimeout(() => {
      saveTimeoutRef.current = null;
      // Fire-and-forget from here (nothing is awaiting this save yet), but
      // still surface a failure -- flushPendingPlanSave() picks up the
      // real outcome via lastSaveOutcomeRef if something awaits it later.
      enqueueSave().catch((err) => setErrorMsg(String(err)));
    }, PLAN_SAVE_DEBOUNCE_MS);
  }

  // Song-structure sections (SPEC.md sec 7.2) live entirely in plan.json and
  // are round-tripped verbatim by the backend, so every mutation below just
  // rewrites plan.sections and funnels through savePlanField -- the exact
  // same debounced/serialized PUT /plan path as every other plan field, no
  // separate save mechanism.
  function updateSection(index: number, patch: Partial<Section>): void {
    if (!detail) return;
    const sections = detail.plan.sections.map((section, i) =>
      i === index ? { ...section, ...patch } : section,
    );
    savePlanField("sections", sections);
  }

  function addSection(section?: Section): void {
    if (!detail) return;
    const blank: Section = section ?? { name: "", start_sec: 0, end_sec: 0, lyrics: "" };
    savePlanField("sections", [...detail.plan.sections, blank]);
  }

  function removeSection(index: number): void {
    if (!detail) return;
    savePlanField(
      "sections",
      detail.plan.sections.filter((_, i) => i !== index),
    );
  }

  // Turns the single ad-hoc repaint region (drag-select on the waveform) into
  // a persisted named section, reusing the existing region interaction rather
  // than a second region-drawing mechanism (SPEC.md sec 7.2 / 9.2).
  function addSectionFromRegion(): void {
    if (!region) return;
    addSection({ name: "", start_sec: region.start, end_sec: region.end, lyrics: "" });
  }

  async function generate() {
    if (!activeId) return;
    // Same base-model-swap gate as extract()/lego()/complete() (SPEC.md sec
    // 4.3) -- a lora's weights only load against the studio_ops base
    // checkpoint, so selecting one forces the same swap those already gate
    // behind confirmation. Only shown when a lora is actually selected; a
    // plain generate never prompts.
    if (
      selectedLoraId &&
      !window.confirm(
        "Generate swaps the loaded model to the studio_ops base model to use this style pack (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // The plan can still be mid-debounce (or an earlier save still in
      // flight) when Generate is clicked -- without this, the job can read
      // the plan from before the user's last edit (e.g. the query/caption
      // they just typed).
      await flushPendingPlanSave();
      const queued = await api.generate(activeId, selectedLoraId);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "generate job failed");
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  async function cover() {
    if (!activeId || (!selectedTakeId && !uploadedSourcePath)) return;
    // Same base-model-swap gate as generate() above.
    if (
      selectedLoraId &&
      !window.confirm(
        "Cover swaps the loaded model to the studio_ops base model to use this style pack (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // Same race as generate(): the plan can still be mid-debounce (or an
      // earlier save still in flight) when Cover is clicked -- the worker
      // reads the on-disk plan, so a stale caption/lyrics edit would
      // otherwise silently leak into the cover job.
      await flushPendingPlanSave();
      const source = selectedTakeId
        ? { takeId: selectedTakeId }
        : { uploadPath: uploadedSourcePath! };
      const queued = await api.cover(activeId, source, coverStrength, selectedLoraId);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "cover job failed");
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  async function repaint() {
    // A region is only drawable on the selected take's waveform -- an
    // uploaded file has no waveform to drag a region on, so it repaints the
    // whole file (repainting_start/end default to the same 0/-1 "full
    // track" the job body itself defaults to).
    if (!activeId) return;
    if (selectedTakeId) {
      if (!region) return;
    } else if (!uploadedSourcePath) {
      return;
    }
    // Same base-model-swap gate as generate() above.
    if (
      selectedLoraId &&
      !window.confirm(
        "Repaint swaps the loaded model to the studio_ops base model to use this style pack (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // Same race as generate()/cover(): flush any in-flight plan edit before
      // the worker reads plan.json off disk.
      await flushPendingPlanSave();
      const source = selectedTakeId
        ? { takeId: selectedTakeId }
        : { uploadPath: uploadedSourcePath! };
      const start = selectedTakeId ? region!.start : 0;
      const end = selectedTakeId ? region!.end : -1;
      const queued = await api.repaint(activeId, source, start, end, selectedLoraId);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "repaint job failed");
      } else if (selectedTakeId) {
        clearRegion();
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  async function extract() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    // SPEC.md sec 4.3: one GPU occupant -- swapping between the iterate and
    // studio_ops base models unloads/reloads the DiT, so this must be a
    // deliberate, confirmed action, not a side effect of a stray click.
    if (
      !window.confirm(
        "Extract swaps the loaded model to the studio_ops base model (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // Same race as cover()/repaint(): flush any in-flight plan edit before
      // the worker reads plan.json off disk.
      await flushPendingPlanSave();
      const queued = await api.extract(activeId, selectedTakeId, trackName);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "extract job failed");
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  async function lego() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    // Same base-model-swap gate as extract() (SPEC.md sec 4.3).
    if (
      !window.confirm(
        "Lego swaps the loaded model to the studio_ops base model (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // Same race as cover()/repaint()/extract(): flush any in-flight plan
      // edit before the worker reads plan.json (and its caption) off disk.
      await flushPendingPlanSave();
      const queued = await api.lego(activeId, selectedTakeId, trackName, region);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "lego job failed");
      } else {
        clearRegion();
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  async function complete() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    // Same base-model-swap gate as extract() (SPEC.md sec 4.3).
    if (
      !window.confirm(
        "Complete swaps the loaded model to the studio_ops base model (slower, SPEC sec 4.3). Continue?"
      )
    ) {
      return;
    }
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      await flushPendingPlanSave();
      const queued = await api.complete(activeId, selectedTakeId, trackName);
      const job = await pollJob(queued.id, (update) => setBusyStatus(update.status));
      if (job.status === "error") {
        setErrorMsg(job.error ?? "complete job failed");
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
    }
  }

  function toggleLoraSource(takeId: string): void {
    setLoraSourceIds((prev) => {
      const next = new Set(prev);
      if (next.has(takeId)) next.delete(takeId);
      else next.add(takeId);
      return next;
    });
  }

  async function trainLora() {
    if (!activeId || loraSourceIds.size < MIN_LORA_SOURCE_TAKES || !loraName.trim()) return;
    // Captured once, up front -- training can run for up to
    // LORA_TRAIN_POLL_TIMEOUT_MS (~90 minutes), and nothing stops the user
    // from switching to a different project while it's in flight. Every
    // mutation below that touches project-specific state (the error banner,
    // the source-take selection, the name field, the mid-poll status text)
    // must check activeIdRef against this before writing, or a stale poll
    // tick / late completion for project A can clobber project B's
    // in-progress selections or surface A's failure in B's UI (reviewer-
    // flagged). busy itself is the exception: it's the single cross-app
    // "a job is in flight" lock (SPEC.md sec 4.3 one GPU occupant), not
    // project-specific data, so it's still cleared unconditionally in
    // `finally` -- guarding that clear would leave every other project
    // permanently disabled if the user isn't back on this one the moment
    // training finishes.
    const projectId = activeId;
    const sourceIds = Array.from(loraSourceIds);
    const name = loraName.trim();
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      const queued = await api.trainLora(projectId, sourceIds, name);
      // Show the new training in the Style Packs pane right away (same
      // in-flight entry a refresh would recover) and arm the recovery poll
      // for it, so the pane's view of training is identical whether or not
      // the page is reloaded mid-training.
      refreshTrainingJobs(projectId).catch(() => {});
      // Training runs the ACE-Step training loop under the worker's
      // GPU-exclusive lock and can take roughly an hour (SPEC.md sec 4.4) --
      // far past the default job-poll ceiling tuned for generate/cover/etc.
      const job = await pollJob(
        queued.id,
        (update) => {
          if (activeIdRef.current === projectId) setBusyStatus(update.status);
        },
        LORA_TRAIN_POLL_TIMEOUT_MS,
      );
      if (activeIdRef.current === projectId) {
        if (job.status === "error") {
          setErrorMsg(job.error ?? "train_lora job failed");
        } else {
          setLoraSourceIds(new Set());
          setLoraName("");
        }
      }
      await refreshLoras(projectId);
    } catch (err) {
      if (activeIdRef.current === projectId) setErrorMsg(String(err));
    } finally {
      setBusy(false);
      setBusyStatus(null);
      // Re-sync the recovered-training view with the queue. On success this
      // just clears any entry; on a poll-timeout (the catch above) the job
      // is still running server-side, and this is what hands it over to the
      // recovery poll so it keeps being tracked after `busy` is released.
      refreshTrainingJobs(projectId).catch(() => {});
    }
  }

  // While an extract/lego/complete job is busy, the base model may still be
  // mid-swap (SPEC.md sec 4.3) -- the already-running 5s health poll keeps
  // health.dit_loaded current, so this just reads that instead of polling
  // again. This is only accurate because server.jobs.run_claimed_job
  // publishes worker_status as soon as the worker backend confirms the swap
  // itself is done (right before generation starts), not just once the
  // whole job finishes -- otherwise dit_loaded would stay on the pre-swap
  // profile for the entire job and this would say "loading base model…"
  // throughout the actual extraction too. Falls back to the normal
  // "<verb>ing… (status)" text once the worker reports studio_ops loaded.
  const studioOpsLoading = busy && health?.dit_loaded !== "studio_ops";

  // The pack the load-half selection (SPEC.md sec 4.4) currently points at,
  // resolved against the project's pack list so the indicator next to
  // Generate/Cover/Repaint can show its name. Null when none is selected --
  // or when the id no longer resolves (a pack dir removed out of band), in
  // which case the <select> visually falls back to "None" too.
  const selectedLora = selectedLoraId
    ? loras.find((l) => l.id === selectedLoraId) ?? null
    : null;

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
  }, [activeId, detail, selectedTakeId, busy]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Wizard's Bard</h1>
        <div className={`health ${health?.ok ? "ok" : "down"}`} title={healthError ?? undefined}>
          <span className="dot" />
          {health ? `${health.gpu}${health.dit_loaded ? ` · ${health.dit_loaded}` : ""}` : "server offline"}
        </div>
      </header>

      <div className="body">
        <aside className="library">
          <h2>Library</h2>
          <div className="new-project">
            <input
              placeholder="title"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
            />
            <input
              placeholder="simple query (optional)"
              value={newQuery}
              onChange={(e) => setNewQuery(e.target.value)}
            />
            <button onClick={createProject}>New project</button>
          </div>
          <input
            className="library-search"
            placeholder="Search projects"
            value={librarySearch}
            onChange={(e) => setLibrarySearch(e.target.value)}
          />
          <ul className="project-list">
            {projects
              .filter((p) => p.title.toLowerCase().includes(librarySearch.toLowerCase()))
              .sort((a, b) => Number(b.favorite) - Number(a.favorite))
              .map((p) => (
                <li key={p.id} className={p.id === activeId ? "active" : ""}>
                  <div className="project-row" onClick={() => switchActiveProject(p.id)}>
                    <span className="project-title">{p.title}</span>
                    <span className="project-updated">
                      {p.updated_at ? new Date(p.updated_at).toLocaleString() : ""}
                    </span>
                  </div>
                  <button
                    className={`favorite-btn ${p.favorite ? "favorited" : ""}`}
                    title={p.favorite ? "Unfavorite" : "Favorite"}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleFavorite(p);
                    }}
                  >
                    {p.favorite ? "★" : "☆"}
                  </button>
                  <button className="open-btn" onClick={() => switchActiveProject(p.id)}>
                    Open
                  </button>
                  <button
                    className="delete-btn"
                    title="Delete project"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteProject(p);
                    }}
                  >
                    ✕
                  </button>
                </li>
              ))}
          </ul>
        </aside>

        <main className="workspace">
          {errorMsg && <div className="error">{errorMsg}</div>}
          {!detail && <p className="hint">Select or create a project.</p>}
          {detail && (
            <>
              {titleDraft === null ? (
                <h2 className="workspace-title">
                  <button
                    type="button"
                    className="rename-title"
                    title="Rename project"
                    onClick={() => setTitleDraft(detail.project.title)}
                  >
                    {detail.project.title}
                    <span className="rename-title-icon" aria-hidden="true">
                      ✎
                    </span>
                  </button>
                </h2>
              ) : (
                <h2 className="workspace-title">
                  <input
                    className="rename-title-input"
                    aria-label="Project title"
                    value={titleDraft}
                    autoFocus
                    onChange={(e) => setTitleDraft(e.target.value)}
                    onFocus={(e) => e.currentTarget.select()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        commitProjectTitle();
                      } else if (e.key === "Escape") {
                        // Cancel: drop the draft, keep the saved title.
                        setTitleDraft(null);
                      }
                    }}
                    onBlur={() => commitProjectTitle()}
                  />
                </h2>
              )}

              <div className="panes">
                <section className="pane plan">
                  <h3>Plan</h3>
                  <label>
                    Simple query
                    <input
                      value={detail.plan.query}
                      onChange={(e) => savePlanField("query", e.target.value)}
                    />
                  </label>
                  <label>
                    Caption
                    <input
                      value={detail.plan.caption}
                      onChange={(e) => savePlanField("caption", e.target.value)}
                    />
                  </label>
                  <label>
                    Lyrics
                    <textarea
                      value={detail.plan.lyrics}
                      onChange={(e) => savePlanField("lyrics", e.target.value)}
                    />
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={detail.plan.instrumental}
                      onChange={(e) => savePlanField("instrumental", e.target.checked)}
                    />
                    Instrumental
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={detail.plan.caption_rewrite}
                      onChange={(e) => savePlanField("caption_rewrite", e.target.checked)}
                    />
                    Allow caption rewrite (Custom mode LM thinking)
                  </label>
                  <div className="plan-grid">
                    <label>
                      BPM
                      <input
                        type="number"
                        value={detail.plan.bpm ?? ""}
                        onChange={(e) =>
                          savePlanField("bpm", e.target.value === "" ? null : Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      Key
                      <input
                        value={detail.plan.keyscale ?? ""}
                        onChange={(e) =>
                          savePlanField("keyscale", e.target.value === "" ? null : e.target.value)
                        }
                      />
                    </label>
                    <label>
                      Time signature
                      <input
                        value={detail.plan.timesignature}
                        onChange={(e) => savePlanField("timesignature", e.target.value)}
                      />
                    </label>
                    <label>
                      Duration (sec)
                      <input
                        type="number"
                        value={detail.plan.duration_sec}
                        onChange={(e) => savePlanField("duration_sec", Number(e.target.value))}
                      />
                    </label>
                    <label>
                      Language
                      <input
                        value={detail.plan.vocal_language}
                        onChange={(e) => savePlanField("vocal_language", e.target.value)}
                      />
                    </label>
                  </div>

                  <div className="plan-sections">
                    <div className="plan-sections-header">
                      <span className="plan-sections-title">Sections</span>
                      <button type="button" onClick={() => addSection()}>
                        Add section
                      </button>
                    </div>
                    {detail.plan.sections.length === 0 && (
                      <p className="hint">
                        No sections. Add one here, or drag a waveform region and use “Add
                        section from region”.
                      </p>
                    )}
                    <ul className="section-list">
                      {detail.plan.sections.map((section, index) => (
                        <li key={index} className="section-row">
                          <input
                            className="section-name"
                            placeholder="name"
                            value={section.name}
                            onChange={(e) => updateSection(index, { name: e.target.value })}
                          />
                          <input
                            className="section-time"
                            type="number"
                            min={0}
                            step={0.1}
                            title="start (sec)"
                            value={section.start_sec}
                            onChange={(e) =>
                              updateSection(index, { start_sec: Number(e.target.value) })
                            }
                          />
                          <span className="section-sep">–</span>
                          <input
                            className="section-time"
                            type="number"
                            min={0}
                            step={0.1}
                            title="end (sec)"
                            value={section.end_sec}
                            onChange={(e) =>
                              updateSection(index, { end_sec: Number(e.target.value) })
                            }
                          />
                          <input
                            className="section-lyrics"
                            placeholder="lyrics snippet"
                            value={section.lyrics}
                            onChange={(e) => updateSection(index, { lyrics: e.target.value })}
                          />
                          <button type="button" onClick={() => removeSection(index)}>
                            Delete
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </section>

                <section className="pane takes">
                  <h3>Takes</h3>
                  {detail.takes.length === 0 && <p className="hint">No takes yet.</p>}
                  <ul>
                    {detail.takes.map((take) => (
                      <li
                        key={take.id}
                        className={[
                          take.id === selectedTakeId ? "selected" : "",
                          take.id === detail.project.active_take_id ? "active-take" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        onClick={() => setSelectedTakeId(take.id)}
                      >
                        <div className="take-meta">
                          <input
                            type="checkbox"
                            className="lora-source-checkbox"
                            title="Include in style pack training source"
                            checked={loraSourceIds.has(take.id)}
                            onClick={(e) => e.stopPropagation()}
                            onChange={() => toggleLoraSource(take.id)}
                          />
                          <span>{take.task_type}</span>
                          <span>seed {take.seed}</span>
                          <span>
                            {take.duration_sec != null ? `${take.duration_sec.toFixed(1)}s` : "—"}
                          </span>
                          <span>score {take.score ?? "—"}</span>
                          {take.id === detail.project.active_take_id && (
                            <span className="active-take-badge">active</span>
                          )}
                          {/* LoRA provenance (SPEC.md sec 4.4): which style
                              pack this take was generated with, so a styled
                              result can be reproduced -- resolved to the
                              pack's name, falling back to the stable pack id
                              when the meta no longer resolves. Clicking a
                              still-loadable pack selects it for the next
                              Generate/Cover/Repaint. */}
                          {take.lora_id &&
                            (() => {
                              const pack = loras.find((l) => l.id === take.lora_id);
                              if (pack && !pack.error) {
                                return (
                                  <button
                                    type="button"
                                    className="take-lora-badge"
                                    title={`Generated with style pack "${pack.name}" — click to select it for the next Generate/Cover/Repaint`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setSelectedLoraId(pack.id);
                                    }}
                                  >
                                    style: {pack.name}
                                  </button>
                                );
                              }
                              return (
                                <span
                                  className="take-lora-badge"
                                  title={
                                    pack
                                      ? `Generated with style pack "${pack.name}" (${take.lora_id}) — it failed training and cannot be loaded`
                                      : `Generated with style pack ${take.lora_id} (not found in this project)`
                                  }
                                >
                                  style: {pack ? pack.name : take.lora_id.slice(0, 8)}
                                </span>
                              );
                            })()}
                          <button
                            type="button"
                            className={`favorite-btn ${take.favorite ? "favorited" : ""}`}
                            title={take.favorite ? "Unfavorite" : "Favorite"}
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleTakeFavorite(take);
                            }}
                          >
                            {take.favorite ? "★" : "☆"}
                          </button>
                        </div>
                        <textarea
                          className="take-notes"
                          placeholder="Notes..."
                          value={take.notes}
                          onClick={(e) => e.stopPropagation()}
                          onChange={(e) => saveTakeNotes(take.id, e.target.value)}
                          onBlur={() => flushTakeNotes(take.id).catch((err) => setErrorMsg(String(err)))}
                        />
                        <div className="take-actions">
                          <button
                            type="button"
                            disabled={take.id === detail.project.active_take_id || !!take.error}
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveTake(take.id);
                            }}
                          >
                            Set active
                          </button>
                          <button
                            type="button"
                            disabled={!!take.error || take.id === selectedTakeId}
                            title={
                              take.id === selectedTakeId
                                ? "Already selected as A -- pick a different take to compare"
                                : undefined
                            }
                            onClick={(e) => {
                              e.stopPropagation();
                              setCompareTakeId((prev) => (prev === take.id ? null : take.id));
                            }}
                          >
                            {take.id === compareTakeId ? "Comparing" : "Compare"}
                          </button>
                          {take.parent_take_id &&
                            (detail.takes.find((t) => t.id === take.parent_take_id) ? (
                              <button
                                type="button"
                                className="parent-take-link"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedTakeId(take.parent_take_id);
                                }}
                              >
                                from {take.parent_take_id.slice(0, 8)}
                              </button>
                            ) : (
                              <span className="parent-take-id">
                                from {take.parent_take_id.slice(0, 8)}
                              </span>
                            ))}
                        </div>
                        {take.error ? (
                          // A take whose generation failed has no audio file
                          // (SPEC.md sec 10 point 5) -- show the error instead
                          // of an <audio> that would just 404.
                          <span className="take-error">failed: {take.error}</span>
                        ) : (
                          <>
                            <TakeAudioPlayer
                              projectId={detail.project.id}
                              takeId={take.id}
                              registerRef={registerAudioRef}
                            />
                            <a
                              href={api.takeAudioUrl(detail.project.id, take.id)}
                              download={`${take.id}.wav`}
                            >
                              download
                            </a>
                            {take.has_lrc && (
                              <a
                                href={api.takeLrcUrl(detail.project.id, take.id)}
                                download={`${take.id}.lrc`}
                              >
                                lyrics (.lrc)
                              </a>
                            )}
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                  <div className="export-panel">
                    <label className="include-stems">
                      <input
                        type="checkbox"
                        checked={includeStems}
                        onChange={(e) => setIncludeStems(e.target.checked)}
                      />
                      Include stems (extract / lego takes)
                    </label>
                    <a
                      className="export-link"
                      href={`${api.exportUrl(detail.project.id)}?include_stems=${includeStems}`}
                      download={`${detail.project.title}-export.zip`}
                    >
                      Export project (.zip)
                    </a>
                  </div>
                </section>

                <section className="pane style-packs">
                  <h3>Style packs</h3>
                  {loras.length === 0 && trainingJobs.length === 0 && (
                    <p className="hint">No style packs trained yet.</p>
                  )}
                  <ul className="lora-list">
                    {/* Trainings recovered from the job queue (GET /api/jobs)
                        -- still queued/running, so no pack meta exists yet
                        (it is written only when training finishes). Shown
                        here, not just as this session's busy state, so a
                        refresh mid-training restores visible progress. */}
                    {trainingJobs.map((job) => (
                      <li key={job.id} className="lora-training">
                        <span className="lora-name">Training style pack…</span>
                        <span className="lora-status">
                          {job.status === "queued"
                            ? "queued — waiting for the GPU"
                            : "running"}
                        </span>
                      </li>
                    ))}
                    {loras.map((lora) => (
                      <li key={lora.id} className={lora.error ? "lora-error" : ""}>
                        <span className="lora-name">{lora.name}</span>
                        <span className="lora-status">{lora.error ? `error: ${lora.error}` : lora.status ?? "—"}</span>
                        <span className="lora-loss">
                          {lora.final_loss != null ? `loss ${lora.final_loss.toFixed(4)}` : ""}
                        </span>
                      </li>
                    ))}
                  </ul>
                  <div className="lora-train-panel">
                    <p className="hint">
                      Check {MIN_LORA_SOURCE_TAKES}+ takes above, name the pack, then train.
                      Selected: {loraSourceIds.size}/{MIN_LORA_SOURCE_TAKES}
                    </p>
                    <label>
                      Style pack name
                      <input
                        placeholder="my-style"
                        value={loraName}
                        onChange={(e) => setLoraName(e.target.value)}
                      />
                    </label>
                    <button
                      onClick={trainLora}
                      disabled={
                        busy ||
                        trainingJobs.length > 0 ||
                        loraSourceIds.size < MIN_LORA_SOURCE_TAKES ||
                        !loraName.trim()
                      }
                      title={
                        trainingJobs.length > 0
                          ? "A style pack is already training for this project"
                          : loraSourceIds.size < MIN_LORA_SOURCE_TAKES
                            ? `Select at least ${MIN_LORA_SOURCE_TAKES} takes first`
                            : !loraName.trim()
                              ? "Enter a style pack name first"
                              : undefined
                      }
                    >
                      {busy
                        ? `Training… (${busyStatus ?? "queued"})`
                        : trainingJobs.length > 0
                          ? `Training… (${trainingJobs[0].status})`
                          : "Train style pack"}
                    </button>
                  </div>
                </section>

                {compareTakeId &&
                  (() => {
                    const compareTake = detail.takes.find((t) => t.id === compareTakeId);
                    if (!compareTake || compareTake.error) return null;
                    const selectedTake = selectedTakeId
                      ? detail.takes.find((t) => t.id === selectedTakeId)
                      : undefined;
                    return (
                    <section className="pane compare">
                      <h3>Compare</h3>
                      <div className="compare-panel">
                        <div className="compare-slot">
                          <span className="compare-label">A: selected</span>
                          {selectedTake && !selectedTake.error ? (
                            <audio controls src={api.takeAudioUrl(detail.project.id, selectedTakeId!)} />
                          ) : (
                            <p className="hint">Select a take to fill A too.</p>
                          )}
                        </div>
                        <button
                          type="button"
                          className="swap-compare"
                          disabled={!selectedTakeId}
                          onClick={swapCompare}
                        >
                          Swap A/B
                        </button>
                        <div className="compare-slot">
                          <span className="compare-label">B: comparing</span>
                          <audio controls src={api.takeAudioUrl(detail.project.id, compareTakeId)} />
                        </div>
                        <button
                          type="button"
                          className="close-compare"
                          onClick={() => setCompareTakeId(null)}
                        >
                          Close compare
                        </button>
                      </div>
                    </section>
                    );
                  })()}

                <section className="pane waveform">
                  <h3>Waveform</h3>
                  <div
                    className="upload-dropzone"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDropAudio}
                  >
                    {uploadedSourcePath ? (
                      <span className="upload-dropzone-file">
                        Source: {uploadedSourceName ?? uploadedSourcePath}
                        <button type="button" onClick={clearUploadedSource}>
                          Clear
                        </button>
                      </span>
                    ) : (
                      <p className="hint">
                        Drag a local WAV/MP3 here to use it as a Cover/Repaint source
                      </p>
                    )}
                    {uploadError && <p className="upload-error">{uploadError}</p>}
                  </div>
                  <div className="waveform-canvas">
                    {selectedTakeId &&
                    !detail.takes.find((t) => t.id === selectedTakeId)?.error ? (
                      <div ref={waveformContainerRef} className="waveform-wavesurfer" />
                    ) : (
                      <p className="hint">no active take</p>
                    )}
                  </div>
                  <div className="region-actions">
                    <button
                      type="button"
                      onClick={addSectionFromRegion}
                      disabled={!region}
                      title={
                        region
                          ? "Append this region as a named section in the Plan"
                          : "Drag a region on the waveform first"
                      }
                    >
                      Add section from region
                    </button>
                  </div>
                  {region && (
                    <div className="region-info">
                      <span>
                        Region: {region.start.toFixed(1)}s – {region.end.toFixed(1)}s
                      </span>
                      <button onClick={clearRegion}>Clear region</button>
                    </div>
                  )}
                  <div className="waveform-actions">
                    <label className="lora-select">
                      Style pack
                      <select
                        value={selectedLoraId ?? ""}
                        onChange={(e) => setSelectedLoraId(e.target.value || null)}
                        disabled={busy}
                        title="Applies to Generate/Cover/Repaint below (SPEC.md sec 4.4)"
                      >
                        <option value="">None</option>
                        {loras
                          .filter((lora) => !lora.error)
                          .map((lora) => (
                            <option key={lora.id} value={lora.id}>
                              {lora.name}
                            </option>
                          ))}
                      </select>
                    </label>
                    {/* Makes the load-half choice impossible to miss next to
                        the buttons it affects: the plain <select> above can
                        read as just another form field. */}
                    {selectedLora && (
                      <span
                        className="lora-active-badge"
                        title={`Generate/Cover/Repaint will run against the studio_ops base model with style pack "${selectedLora.name}" (SPEC.md sec 4.4)`}
                      >
                        Style pack: {selectedLora.name}
                      </span>
                    )}
                    <button
                      onClick={generate}
                      disabled={busy}
                      title="Shortcuts: g generate · space play/pause · ↑/↓ prev/next take · ctrl/cmd+s save plan"
                    >
                      {busy ? `Generating… (${busyStatus ?? "queued"})` : "Generate"}
                    </button>
                    <label className="cover-strength">
                      Strength
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={coverStrength}
                        onChange={(e) => setCoverStrength(Number(e.target.value))}
                      />
                    </label>
                    <button
                      onClick={cover}
                      disabled={
                        busy ||
                        (!selectedTakeId && !uploadedSourcePath) ||
                        !!detail.takes.find((t) => t.id === selectedTakeId)?.error
                      }
                      title={
                        selectedTakeId || uploadedSourcePath
                          ? undefined
                          : "Select a take or drop a file first"
                      }
                    >
                      {busy ? `Covering… (${busyStatus ?? "queued"})` : "Cover"}
                    </button>
                    <button
                      onClick={repaint}
                      disabled={
                        busy ||
                        (!selectedTakeId && !uploadedSourcePath) ||
                        (!!selectedTakeId && !region) ||
                        !!detail.takes.find((t) => t.id === selectedTakeId)?.error
                      }
                      title={
                        !selectedTakeId && !uploadedSourcePath
                          ? "Select a take or drop a file first"
                          : selectedTakeId && !region
                            ? "Drag a region on the waveform first"
                            : undefined
                      }
                    >
                      {busy ? `Repainting… (${busyStatus ?? "queued"})` : "Repaint"}
                    </button>
                    <label className="track-name">
                      Track name / classes
                      <input
                        placeholder="vocals, drums, bass..."
                        value={trackName}
                        onChange={(e) => setTrackName(e.target.value)}
                      />
                    </label>
                    <button
                      onClick={extract}
                      disabled={
                        busy ||
                        !selectedTakeId ||
                        !trackName.trim() ||
                        !!detail.takes.find((t) => t.id === selectedTakeId)?.error
                      }
                      title={
                        !selectedTakeId
                          ? "Select a take first"
                          : !trackName.trim()
                            ? "Enter a track name first"
                            : undefined
                      }
                    >
                      {busy
                        ? studioOpsLoading
                          ? "loading base model…"
                          : `Extracting… (${busyStatus ?? "queued"})`
                        : "Extract"}
                    </button>
                    <button
                      onClick={lego}
                      disabled={
                        busy ||
                        !selectedTakeId ||
                        !trackName.trim() ||
                        !!detail.takes.find((t) => t.id === selectedTakeId)?.error
                      }
                      title={
                        !selectedTakeId
                          ? "Select a take first"
                          : !trackName.trim()
                            ? "Enter a track name first"
                            : undefined
                      }
                    >
                      {busy
                        ? studioOpsLoading
                          ? "loading base model…"
                          : `Adding track… (${busyStatus ?? "queued"})`
                        : "Lego"}
                    </button>
                    <button
                      onClick={complete}
                      disabled={
                        busy ||
                        !selectedTakeId ||
                        !trackName.trim() ||
                        !!detail.takes.find((t) => t.id === selectedTakeId)?.error
                      }
                      title={
                        !selectedTakeId
                          ? "Select a take first"
                          : !trackName.trim()
                            ? "Enter a track name / classes first"
                            : undefined
                      }
                    >
                      {busy
                        ? studioOpsLoading
                          ? "loading base model…"
                          : `Completing… (${busyStatus ?? "queued"})`
                        : "Complete"}
                    </button>
                  </div>
                </section>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
