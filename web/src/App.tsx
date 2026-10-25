import {
  DragEvent,
  KeyboardEvent as ReactKeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  api,
  Job,
  JobRequest,
  Lora,
  Plan,
  PresetResult,
  ProfileCapability,
  ProjectDetail,
  ProjectSummary,
  Section,
} from "./api";
import { ConfirmationDialog } from "./components/ConfirmationDialog";
import { Icon } from "./components/Icon";
import { FieldOrigin } from "./components/FieldOrigin";
import { LibraryPane } from "./components/LibraryPane";
import { StylePackPanel } from "./components/StylePackPanel";
import { LoudnessMeter } from "./components/LoudnessMeter";
import { TakeAudioPlayer } from "./components/TakeAudioPlayer";
import {
  AppShell,
  OperationDock,
  PlanInspector,
  StudioPlayer,
  StudioStage,
  TakesRail,
} from "./components/Workbench";
import { DIT_PROFILE_OPTIONS, MIN_LORA_SOURCE_TAKES, STRUCTURE_TAGS } from "./constants";
import { generateIntent, hasInstrumentalConflict, INTENT_COPY, originOf } from "./lib/creation";
import { formatClock, parseSeed } from "./lib/format";
import { jobBodies } from "./lib/jobBodies";
import { describeJob, errorSummary, submitJob } from "./lib/jobs";
import { useProjectJobs } from "./hooks/useProjectJobs";
import { useHealth } from "./hooks/useHealth";
import { RemoteGpuControl } from "./components/RemoteGpuControl";
import {
  clearLocalDraft,
  LocalDraft,
  readLocalDraft,
  usePlanAutosave,
} from "./hooks/usePlanAutosave";
import { useTakeNotes } from "./hooks/useTakeNotes";
import { useWaveform } from "./hooks/useWaveform";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { ConfirmationRequest, InspectorTab, OperationGroup } from "./types";

const READINESS_LABEL: Record<string, string> = {
  ready: "Ready",
  busy: "Busy",
  loading: "Loading model",
  stopped: "Worker stopped",
  unavailable: "Unavailable",
};

function profileTitle(profile: string, report: ProfileCapability | undefined): string {
  if (!report) return profile;
  const files =
    report.files === "remote"
      ? "models on the worker host"
      : report.validated
        ? "has rendered on this machine"
        : report.files === "present"
          ? "files present, not yet validated by a render"
          : `files ${report.files}`;
  const reason =
    report.supported === false && report.reason ? ` Unavailable: ${report.reason}` : "";
  return `${report.label}: ${report.summary} (${files}).${reason}`;
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [librarySearch, setLibrarySearch] = useState("");
  // Jukebox-style inline preview from the library list: one shared <audio>
  // element so starting a second project's preview stops the first rather
  // than letting previews stack up concurrently.
  const [previewProjectId, setPreviewProjectId] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [newQuery, setNewQuery] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  // True only while a job body is on its way to the server. Whether a job is
  // queued or running comes from the queue itself (useProjectJobs below).
  const [submitting, setSubmitting] = useState(false);
  // A local draft newer than nothing on the server but older than the saved
  // plan: offered back to the user, never applied silently (sec 16.6).
  const [draftOffer, setDraftOffer] = useState<LocalDraft | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedTakeId, setSelectedTakeId] = useState<string | null>(null);
  const [compareTakeId, setCompareTakeId] = useState<string | null>(null);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("takes");
  const [operationGroup, setOperationGroup] = useState<OperationGroup>("create");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [planDetailsOpen, setPlanDetailsOpen] = useState(true);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  // Enter is followed by blur as the input closes. This synchronous guard
  // prevents both events from racing identical PATCH requests.
  const titleSavingRef = useRef(false);
  const [confirmation, setConfirmation] = useState<ConfirmationRequest | null>(null);
  // A dropped local file (SPEC.md sec 12 Phase 6) is an alternative
  // cover/repaint source to a selected take -- server.jobs
  // _resolve_source_audio only ever accepts one or the other. uploadedSourceName
  // is just for display (the server discards the client's original filename).
  const [uploadedSourcePath, setUploadedSourcePath] = useState<string | null>(null);
  const [uploadedSourceName, setUploadedSourceName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [coverStrength, setCoverStrength] = useState(0.7);
  // Fixed seed for the next Generate/Cover/Repaint (SPEC.md sec 7.3). Kept
  // as the raw input string so an empty field stays empty on screen; empty
  // or -1 means the worker picks a seed and records the actual one used.
  const [seedInput, setSeedInput] = useState("");
  const [trackName, setTrackName] = useState("");
  const [includeStems, setIncludeStems] = useState(true);
  const { health, capabilities, error: healthError } = useHealth();
  // A preset preview awaiting Apply or Cancel (SPEC.md sec 16.6). Nothing is
  // written until the user applies it.
  const [presetPreview, setPresetPreview] = useState<PresetResult | null>(null);
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
  // Jobs this page submitted. Only their completion selects the new take for
  // listening or clears the inputs that produced it; a job another tab or an
  // agent started finishes without moving this page's selection.
  const startedHereRef = useRef<Set<string>>(new Set());

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
  const {
    saveState,
    retrying: planSaveRetrying,
    setSaveState,
    savePlanField,
    savePlan: savePlanDraft,
    flushPendingPlanSave,
    retryNow: retryPlanSave,
    resolveConflict: resolvePlanConflict,
    hasPendingEdit: hasPendingPlanEdit,
    flushOnUnload: flushPlanOnUnload,
    setServerRevision,
    reset: resetPlanAutosave,
  } = usePlanAutosave({
    getContext: () => (activeId && detail ? { projectId: activeId, plan: detail.plan } : null),
    onPlanChange: (plan) => setDetail((current) => (current ? { ...current, plan } : current)),
    onError: (message) => setErrorMsg(message),
    // The server derives origins, mode, and revision on every write (sec
    // 16.3). Take those from the saved plan; keep the text on screen.
    onSaved: (projectId, saved) =>
      setDetail((current) =>
        current?.project.id === projectId
          ? {
              ...current,
              plan: {
                ...current.plan,
                creation_mode: saved.creation_mode,
                field_origins: saved.field_origins,
                revision: saved.revision,
                brief: saved.brief,
              },
            }
          : current,
      ),
  });

  const {
    saveTakeNotes,
    flushTakeNotes,
    flushAllPendingTakeNotes,
    reset: resetTakeNotes,
  } = useTakeNotes({
    getProjectId: () => activeIdRef.current,
    onNotesChange: (takeId, notes) => updateTakeLocal(takeId, { notes }),
    onError: (message) => setErrorMsg(message),
  });

  // Mirrors activeId, but updated synchronously (the instant a switch is
  // committed, not after the next render) so refreshDetail can tell whether
  // its response is still for the project actually on screen. Selecting A
  // then B rapidly fires two overlapping GET /api/projects/{id} requests
  // that can resolve in either order; without this check, A's slower
  // response can land after B's and overwrite `detail` with A's data while
  // activeId (and the rest of the UI) is already B -- so a subsequent edit
  // would be saved under B's project id but built from A's plan.
  const activeIdRef = useRef<string | null>(null);

  const lyricsTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const confirmationFocusRef = useRef<HTMLElement | null>(null);

  const requestConfirmation = useCallback(
    (request: Omit<ConfirmationRequest, "resolve">): Promise<boolean> => {
      confirmationFocusRef.current = document.activeElement as HTMLElement | null;
      return new Promise((resolve) => setConfirmation({ ...request, resolve }));
    },
    [],
  );

  const resolveConfirmation = useCallback((accepted: boolean) => {
    setConfirmation((current) => {
      current?.resolve(accepted);
      return null;
    });
    requestAnimationFrame(() => confirmationFocusRef.current?.focus());
  }, []);

  // Keyed by take id rather than a single ref, since every take in the list
  // renders its own <audio> (and a future compare panel could show two at
  // once) -- looking one up via document.querySelector would be ambiguous
  // and non-React-idiomatic.
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({});
  const registerAudioRef = useCallback((id: string, el: HTMLAudioElement | null) => {
    audioRefs.current[id] = el;
  }, []);

  const {
    region,
    clearRegion,
    waveformPlaying,
    waveformCurrentTime,
    waveformDuration,
    waveformMediaEl,
    waveformContainerRef,
    wavesurferRef,
    toggleWaveformPlayback,
    seekWaveform,
  } = useWaveform({ selectedTakeId, detail, activeIdRef, audioRefs, setErrorMsg });

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
      // The last debounced plan edit too (SPEC.md sec 16.6); its local draft
      // stays until a save confirms it, in case the browser drops this one.
      flushPlanOnUnload();
    }
    window.addEventListener("pagehide", flushTakeNotesOnUnload);
    window.addEventListener("beforeunload", flushTakeNotesOnUnload);
    return () => {
      window.removeEventListener("pagehide", flushTakeNotesOnUnload);
      window.removeEventListener("beforeunload", flushTakeNotesOnUnload);
      resetPlanAutosave();
      flushTakeNotesOnUnload();
    };
    // Mount-only by design: this registers the unload listeners once. Adding
    // flushAllPendingTakeNotes (redefined every render) would tear down and
    // re-register them on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  // SPEC.md sec 9.1 "Play last take inline (optional)": preview a project's
  // active take straight from the library list, without opening it. A single
  // shared <audio> element makes this jukebox-style -- playing one project's
  // preview replaces whatever was previously playing.
  function togglePreview(p: ProjectSummary): void {
    const audio = previewAudioRef.current;
    if (!audio || !p.active_take_id) return;
    if (previewProjectId === p.id) {
      audio.pause();
      setPreviewProjectId(null);
      return;
    }
    audio.src = api.takeAudioUrl(p.id, p.active_take_id);
    audio.play();
    setPreviewProjectId(p.id);
  }

  // SPEC.md sec 4.1: the DiT profile is a project-level default, persisted
  // via PATCH /api/projects/{id} -- jobs never carry their own dit_profile
  // from this UI (server.jobs._resolve_dit_profile falls back to the
  // project's persisted value, and coerces lora-attached jobs to studio_ops
  // on its own). Optimistic update with revert on failure, same shape as the
  // take-favorite toggle below.
  async function setDitProfile(profile: string): Promise<void> {
    if (!activeId || !detail) return;
    const projectId = activeId;
    const previous = detail.project.dit_profile;
    if (previous === profile) return;
    setDetail((prev) =>
      prev ? { ...prev, project: { ...prev.project, dit_profile: profile } } : prev,
    );
    try {
      await api.patchProject(projectId, { dit_profile: profile });
    } catch (err) {
      // Revert only if the user is still looking at this project -- a
      // switch that already happened means `detail` holds the other
      // project's data and must not be clobbered with this one's old value.
      if (activeIdRef.current === projectId) {
        setDetail((prev) =>
          prev ? { ...prev, project: { ...prev.project, dit_profile: previous } } : prev,
        );
      }
      setErrorMsg(String(err));
    }
  }

  // SPEC.md sec 9.1 "Delete (confirm)". The workbench owns the confirmation
  // surface so the irreversible action remains keyboard accessible and
  // visually consistent with the rest of the local tool.
  async function deleteProject(p: ProjectSummary): Promise<void> {
    const accepted = await requestConfirmation({
      title: `Delete “${p.title}”?`,
      message:
        "This permanently removes the project, every take, and its local files. This cannot be undone.",
      confirmLabel: "Delete project",
      destructive: true,
    });
    if (!accepted) {
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
      resetPlanAutosave();
      resetTakeNotes();
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
    // fetch and a job-completion refresh landing after the user has since
    // switched away from the project the job ran in.
    if (activeIdRef.current !== id) return;
    const openingProject = detail?.project.id !== id;
    // An edit still waiting to save is newer than the server's plan: keep
    // it on screen rather than replacing it with the older copy (sec 16.6).
    const keepLocalPlan = !openingProject && hasPendingPlanEdit();
    if (!keepLocalPlan) setServerRevision(id, data.plan.revision);
    setDetail((current) =>
      keepLocalPlan && current?.project.id === id ? { ...data, plan: current.plan } : data,
    );
    if (openingProject) offerLocalDraft(id, data.plan);
    setSelectedTakeId((current) => {
      if (current && data.takes.some((take) => take.id === current)) return current;
      return data.project.active_take_id ?? data.takes[0]?.id ?? null;
    });
    if (openingProject) {
      setTitleDraft(data.project.title);
      setPlanDetailsOpen(Boolean(data.plan.caption || data.plan.lyrics));
      setSaveState("idle");
    }
  }

  // SPEC.md sec 16.6: an unsent edit survives a reload. When the server plan
  // is still the revision the edit started from, the edit is simply resumed
  // (nothing newer can be overwritten). When the server moved on, the edit
  // is offered, never applied silently.
  function offerLocalDraft(id: string, serverPlan: Plan): void {
    const draft = readLocalDraft(id);
    setDraftOffer(null);
    if (!draft) return;
    if (JSON.stringify(draft.plan) === JSON.stringify(serverPlan)) {
      clearLocalDraft(id);
      return;
    }
    if (draft.baseRevision != null && draft.baseRevision === (serverPlan.revision ?? null)) {
      savePlanDraft(draft.plan, id);
      return;
    }
    setDraftOffer(draft);
  }

  function restoreDraftOffer(): void {
    if (!draftOffer || draftOffer.projectId !== activeIdRef.current) return;
    // The user chose this edit over the newer saved plan, so it is sent
    // against the current revision.
    savePlanDraft(draftOffer.plan, draftOffer.projectId);
    setDraftOffer(null);
  }

  function discardDraftOffer(): void {
    if (draftOffer) clearLocalDraft(draftOffer.projectId);
    setDraftOffer(null);
  }

  async function refreshLoras(id: string) {
    const data = await api.listLoras(id);
    // Same stale-response guard as refreshDetail above.
    if (activeIdRef.current !== id) return;
    setLoras(data);
  }

  useEffect(() => {
    // The setState is inside an async rejection handler, not the effect body,
    // so it cannot cascade renders; this is the initial project-list load.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshProjects().catch((err) => setErrorMsg(String(err)));
  }, []);

  useEffect(() => {
    // Deliberate: switching projects must clear the previous project's
    // selection, comparison, upload, and style-pack state in one synchronous
    // pass, before any of it can be rendered against the new project.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelectedTakeId(null);
    setCompareTakeId(null);
    setUploadedSourcePath(null);
    setUploadedSourceName(null);
    setLoraSourceIds(new Set());
    setLoraName("");
    setSelectedLoraId(null);
    setDraftOffer(null);
    setPresetPreview(null);
    setEditingTitle(false);
    setTitleDraft("");
    if (activeId) {
      refreshDetail(activeId).catch((err) => setErrorMsg(String(err)));
      refreshLoras(activeId).catch((err) => setErrorMsg(String(err)));
    } else {
      setDetail(null);
      setLoras([]);
    }
    // Keyed on activeId alone. The refresh* helpers are redefined every render;
    // listing them would refetch the whole project on every state change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  // An uploaded file and a selected take are alternative cover/repaint
  // sources (server.jobs._resolve_source_audio accepts exactly one) --
  // picking a take deselects whatever was dropped. The reverse (dropping a
  // file deselects the take) happens directly in the drop handler.
  useEffect(() => {
    if (selectedTakeId) {
      // Deliberate: a take and an uploaded file are mutually exclusive sources,
      // so selecting one must clear the other.
      // eslint-disable-next-line react-hooks/set-state-in-effect
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
      // Deliberate: A and B must never be the same take, so picking A as B's
      // current value clears the comparison.
      // eslint-disable-next-line react-hooks/set-state-in-effect
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
  function clearUploadedSource() {
    setUploadedSourcePath(null);
    setUploadedSourceName(null);
    setUploadError(null);
  }

  async function uploadSourceFile(file: File) {
    if (!activeId) return;
    const projectId = activeId;
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

  async function handleDropAudio(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) await uploadSourceFile(file);
  }

  async function createProject() {
    setErrorMsg(null);
    try {
      const project = await api.createProject(newTitle || "Untitled", newQuery);
      setNewTitle("");
      setNewQuery("");
      setCreatingProject(false);
      await refreshProjects();
      await switchActiveProject(project.id);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function commitProjectTitle(): Promise<void> {
    if (!activeId || !detail || titleSavingRef.current) return;
    const projectId = activeId;
    const nextTitle = titleDraft.trim() || "Untitled";
    if (nextTitle === detail.project.title) {
      setEditingTitle(false);
      setTitleDraft(detail.project.title);
      return;
    }
    titleSavingRef.current = true;
    try {
      const project = await api.patchProject(projectId, { title: nextTitle });
      setDetail((current) =>
        current?.project.id === projectId
          ? { ...current, project: { ...current.project, title: project.title } }
          : current,
      );
      if (activeIdRef.current === projectId) {
        setTitleDraft(project.title);
        setEditingTitle(false);
      }
      await refreshProjects();
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      titleSavingRef.current = false;
    }
  }

  // Inserts a structure tag (SPEC.md sec 4.4/9.2) at the lyrics textarea's
  // current cursor position, replacing any selection -- or appends it on a
  // new line when the textarea isn't focused/has no tracked selection.
  // Routes through savePlanField exactly like typing the tag by hand, so
  // there's no second save mechanism for tag-button edits.
  function insertLyricsTag(tag: string): void {
    if (!detail) return;
    const current = detail.plan.lyrics ?? "";
    const el = lyricsTextareaRef.current;
    const hasSelection = el && document.activeElement === el;
    const start = hasSelection ? el.selectionStart : current.length;
    const end = hasSelection ? el.selectionEnd : current.length;
    const needsLeadingNewline = start > 0 && current[start - 1] !== "\n";
    const insertion = `${needsLeadingNewline ? "\n" : ""}${tag}\n`;
    const next = current.slice(0, start) + insertion + current.slice(end);
    savePlanField("lyrics", next);

    const cursor = start + insertion.length;
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(cursor, cursor);
    });
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

  async function previewPreset(presetId: string): Promise<void> {
    if (!activeId || !presetId) {
      setPresetPreview(null);
      return;
    }
    const projectId = activeId;
    try {
      // The preview is computed from the saved plan, so save first.
      await flushPendingPlanSave();
      const preview = await api.presetOnProject(projectId, presetId, { apply: false });
      if (activeIdRef.current === projectId) setPresetPreview(preview);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function applyPresetPreview(): Promise<void> {
    if (!activeId || !presetPreview) return;
    const projectId = activeId;
    try {
      await flushPendingPlanSave();
      const result = await api.presetOnProject(projectId, presetPreview.preset.id, {
        apply: true,
        expectedRevision: detail?.plan.revision ?? null,
      });
      if (activeIdRef.current !== projectId) return;
      setServerRevision(projectId, result.plan.revision);
      setDetail((current) =>
        current?.project.id === projectId ? { ...current, plan: result.plan } : current,
      );
      setPresetPreview(null);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  // Submitting, observing, and collecting are separate (SPEC.md sec 16.5):
  // startJob flushes the plan and queues the job with a submission identity,
  // useProjectJobs observes it from the queue (surviving reloads and project
  // switches), and handleJobFinished collects the result.
  const { activeJobs, lastFinished, track } = useProjectJobs({
    projectId: activeId,
    onFinished: handleJobFinished,
  });
  const busy = submitting || activeJobs.length > 0;
  const currentJob = activeJobs[activeJobs.length - 1] ?? lastFinished;
  const activeJobAction = currentJob
    ? currentJob.action === "train_lora"
      ? "style pack training"
      : currentJob.action
    : null;
  const trainingJobs = activeJobs.filter((job) => job.action === "train_lora");

  async function startJob(body: JobRequest): Promise<void> {
    if (!activeId) return;
    const projectId = activeId;
    setSubmitting(true);
    setErrorMsg(null);
    try {
      // The plan can still be mid-debounce (or an earlier save still in
      // flight) when an action is clicked. The job snapshots the plan when
      // it is queued, so it must be the plan the user last saw. A save
      // that failed stops the job instead of queuing stale content.
      await flushPendingPlanSave();
      const job = await submitJob(projectId, body);
      startedHereRef.current.add(job.id);
      if (activeIdRef.current === projectId) track(job);
    } catch (err) {
      if (activeIdRef.current === projectId) setErrorMsg(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  function handleJobFinished(job: Job): void {
    const projectId = job.project_id;
    // useProjectJobs only reports jobs of the project it observes, but the
    // refreshes below are async: every write checks the project again.
    if (activeIdRef.current !== projectId) return;
    const startedHere = startedHereRef.current.delete(job.id);
    if (job.status === "error") {
      setErrorMsg(
        job.action === "train_lora" && !startedHere
          ? `style pack training failed: ${job.error ?? "unknown error"}`
          : (job.error ?? `${job.action} job failed`),
      );
    }
    if (job.action === "train_lora") {
      if (job.status === "done" && startedHere) {
        setLoraSourceIds(new Set());
        setLoraName("");
      }
      refreshLoras(projectId).catch((err) => setErrorMsg(String(err)));
      return;
    }
    refreshDetail(projectId)
      .then(() => {
        if (activeIdRef.current !== projectId || !startedHere) return;
        if (job.status === "done" && job.take_id) {
          // Selected for listening, not kept: the active take only changes
          // when the user says so (SPEC.md sec 16.5).
          setSelectedTakeId(job.take_id);
          if (job.action === "repaint" || job.action === "lego") clearRegion();
        }
      })
      .catch((err) => setErrorMsg(String(err)));
  }

  async function cancelCurrentJob(): Promise<void> {
    if (!currentJob) return;
    try {
      track(await api.cancelJob(currentJob.id));
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function retryCurrentJob(): Promise<void> {
    if (!currentJob || !activeId) return;
    const projectId = activeId;
    try {
      const retried = await api.retryJob(currentJob.id);
      startedHereRef.current.add(retried.id);
      if (activeIdRef.current === projectId) track(retried);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function copyJobSummary(): Promise<void> {
    if (!currentJob) return;
    try {
      await navigator.clipboard.writeText(errorSummary(currentJob));
    } catch {
      setErrorMsg(errorSummary(currentJob));
    }
  }

  // Same base-model-swap gate as extract()/lego()/complete() (SPEC.md sec
  // 4.3) -- a lora's weights only load against the studio_ops base
  // checkpoint, so selecting one forces the same swap those already gate
  // behind confirmation. Only shown when a lora is actually selected; a
  // plain generate never prompts.
  async function confirmStyleSwap(verb: string): Promise<boolean> {
    if (!selectedLoraId) return true;
    return requestConfirmation({
      title: "Load the studio model?",
      message: `${verb[0].toUpperCase()}${verb.slice(1)} needs the studio_ops base model for this style pack. Lyre will unload the current model before the job starts.`,
      confirmLabel: `Load model & ${verb}`,
    });
  }

  async function generate(creationMode?: "from_idea" | "from_brief") {
    if (!activeId) return;
    if (!(await confirmStyleSwap("generate"))) return;
    await startJob(jobBodies.generate(selectedLoraId, parseSeed(seedInput), creationMode));
  }

  async function cover() {
    if (!activeId || (!selectedTakeId && !uploadedSourcePath)) return;
    if (!(await confirmStyleSwap("cover"))) return;
    const source = selectedTakeId
      ? { takeId: selectedTakeId }
      : { uploadPath: uploadedSourcePath! };
    await startJob(jobBodies.cover(source, coverStrength, selectedLoraId, parseSeed(seedInput)));
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
    if (!(await confirmStyleSwap("repaint"))) return;
    const source = selectedTakeId
      ? { takeId: selectedTakeId }
      : { uploadPath: uploadedSourcePath! };
    const start = selectedTakeId ? region!.start : 0;
    const end = selectedTakeId ? region!.end : -1;
    await startJob(jobBodies.repaint(source, start, end, selectedLoraId, parseSeed(seedInput)));
  }

  // SPEC.md sec 4.3: one GPU occupant -- swapping between the iterate and
  // studio_ops base models unloads/reloads the DiT, so these must be a
  // deliberate, confirmed action, not a side effect of a stray click.
  async function confirmBaseModelSwap(message: string, confirmLabel: string): Promise<boolean> {
    return requestConfirmation({ title: "Load the studio model?", message, confirmLabel });
  }

  async function extract() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    const accepted = await confirmBaseModelSwap(
      "Extract uses the studio_ops base model. Lyre will unload the current model before isolating this track.",
      "Load model & extract",
    );
    if (!accepted) return;
    await startJob(jobBodies.extract(selectedTakeId, trackName));
  }

  async function lego() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    const accepted = await confirmBaseModelSwap(
      "Lego uses the studio_ops base model. Lyre will unload the current model before adding or replacing the track.",
      "Load model & add track",
    );
    if (!accepted) return;
    await startJob(jobBodies.lego(selectedTakeId, trackName, region));
  }

  async function complete() {
    if (!activeId || !selectedTakeId || !trackName.trim()) return;
    const accepted = await confirmBaseModelSwap(
      "Complete uses the studio_ops base model. Lyre will unload the current model before filling the arrangement.",
      "Load model & complete",
    );
    if (!accepted) return;
    await startJob(jobBodies.complete(selectedTakeId, trackName));
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
    // Training runs up to ~1 hour (SPEC.md sec 4.4). It is observed like any
    // job, so it survives a reload or a project switch; its completion
    // clears the source selection only on the page that started it.
    await startJob(jobBodies.trainLora(Array.from(loraSourceIds), loraName.trim()));
  }

  // While an extract/lego/complete job is busy, the base model may still be
  // mid-swap (SPEC.md sec 4.3) -- the already-running 5s health poll keeps
  // health.dit_loaded current, so this just reads that instead of polling
  // again. This is only accurate because server.jobs.run_claimed_job
  // publishes worker_status as soon as the worker backend confirms the swap
  // itself is done (right before generation starts), not just once the
  // whole job finishes -- otherwise dit_loaded would stay on the pre-swap
  // profile for the whole job and this would say "loading base model..."
  // throughout the actual extraction too. Falls back to the normal
  // "<verb>ing... (status)" text once the worker reports studio_ops loaded.
  const studioOpsActivity =
    Boolean(selectedLoraId) ||
    ["extract", "lego", "complete", "style pack training"].includes(activeJobAction ?? "");
  const studioOpsLoading = busy && studioOpsActivity && health?.dit_loaded !== "studio_ops";

  // The pack the load-half selection (SPEC.md sec 4.4) currently points at,
  // resolved against the project's pack list so the indicator next to
  // Generate/Cover/Repaint can show its name. Null when none is selected --
  // or when the id no longer resolves (a pack dir removed out of band), in
  // which case the <select> visually falls back to "None" too.
  const selectedLora = selectedLoraId ? (loras.find((l) => l.id === selectedLoraId) ?? null) : null;

  useKeyboardShortcuts({
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
  });

  const filteredProjects = projects
    .filter((project) => project.title.toLowerCase().includes(librarySearch.toLowerCase()))
    .sort((a, b) => Number(b.favorite) - Number(a.favorite));
  const selectedTake = detail?.takes.find((take) => take.id === selectedTakeId) ?? null;
  const jobStatusLabel = currentJob
    ? describeJob(currentJob, {
        loadingBaseModel: currentJob.status === "running" && studioOpsLoading,
      })
    : null;
  const jobIsActive = currentJob
    ? !["done", "error", "cancelled"].includes(currentJob.status)
    : false;
  const jobHeld = currentJob?.status === "queued" && currentJob.stage === "needs_confirmation";
  const saveStateLabel =
    saveState === "saving"
      ? "Saving..."
      : saveState === "saved"
        ? "Saved"
        : saveState === "error"
          ? planSaveRetrying
            ? "Save failed, retrying"
            : "Save failed"
          : saveState === "conflict"
            ? "Changed elsewhere"
            : "Local plan";

  return (
    <AppShell>
      <header className="topbar">
        <div className="brand-cluster">
          <button
            type="button"
            className="icon-button panel-toggle"
            aria-label="Projects"
            aria-controls="project-library"
            aria-expanded={libraryOpen}
            onClick={() => setLibraryOpen((open) => !open)}
          >
            <Icon name="library" />
          </button>
          <h1>
            <img
              className="brand-logo"
              src="/brand/lyre-header.svg"
              alt="The Wizard's Lyre"
              width="210"
            />
          </h1>
        </div>
        <div className="topbar-tools">
          <details className="shortcut-help">
            <summary>Keys</summary>
            <div className="shortcut-popover">
              <span>
                <kbd>G</kbd> Generate
              </span>
              <span>
                <kbd>Space</kbd> Play / pause
              </span>
              <span>
                <kbd>Up</kbd>
                <kbd>Down</kbd> Previous / next take
              </span>
              <span>
                <kbd>Ctrl</kbd>
                <kbd>S</kbd> Save plan
              </span>
            </div>
          </details>
          <RemoteGpuControl />
          <div
            className={`health ${
              health?.ok &&
              (!capabilities || ["ready", "busy", "loading"].includes(capabilities.worker.state))
                ? "ok"
                : "down"
            }`}
            title={healthError ?? undefined}
            role="status"
          >
            <span className="dot" />
            <span className="health-copy" title={health?.gpu}>
              <strong>
                {!health
                  ? "Server offline"
                  : capabilities
                    ? READINESS_LABEL[capabilities.worker.state]
                    : health.gpu}
              </strong>
              <small>
                {capabilities && capabilities.worker.state !== "ready"
                  ? (capabilities.worker.reason ?? "")
                  : health?.dit_loaded
                    ? `${health.dit_loaded} loaded`
                    : "local engine"}
              </small>
            </span>
          </div>
        </div>
      </header>

      <div className="body">
        <LibraryPane
          open={libraryOpen}
          onClose={() => setLibraryOpen(false)}
          projects={projects}
          filteredProjects={filteredProjects}
          librarySearch={librarySearch}
          onSearchChange={setLibrarySearch}
          activeId={activeId}
          onOpenProject={switchActiveProject}
          onToggleFavorite={toggleFavorite}
          onDeleteProject={deleteProject}
          creatingProject={creatingProject}
          onStartCreating={() => setCreatingProject(true)}
          onCancelCreating={() => setCreatingProject(false)}
          onCreateProject={createProject}
          newTitle={newTitle}
          onNewTitleChange={setNewTitle}
          newQuery={newQuery}
          onNewQueryChange={setNewQuery}
          previewProjectId={previewProjectId}
          onTogglePreview={togglePreview}
          previewAudioRef={previewAudioRef}
          onPreviewEnded={() => setPreviewProjectId(null)}
        />

        <main className="workspace">
          {errorMsg && (
            <div className="error" role="alert">
              <span>{errorMsg}</span>
              <button
                type="button"
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setErrorMsg(null)}
              >
                <Icon name="close" />
              </button>
            </div>
          )}

          {capabilities?.disk.low && (
            <div className="disk-warning" role="status">
              {capabilities.disk.blocked
                ? "The disk is nearly full: new takes are refused until space is freed under projects/."
                : "Disk space is low under projects/. A take needs tens of megabytes."}
            </div>
          )}

          {!detail && (
            <section className="workspace-empty" aria-labelledby="empty-title">
              <span className="empty-mark">
                <Icon name="wave" />
              </span>
              <span className="eyebrow">Make room for an idea</span>
              <h2 id="empty-title">Choose a project or start a fresh musical sketch.</h2>
              <p>Your prompt, listening space, and take history stay together here.</p>
              <button
                type="button"
                className="button-primary"
                onClick={() => {
                  setCreatingProject(true);
                  setLibraryOpen(true);
                }}
              >
                <Icon name="add" /> New project
              </button>
            </section>
          )}

          {detail && (
            <div className="project-workspace">
              <header className="workspace-header">
                <div className="project-heading">
                  <span className="eyebrow">Composition</span>
                  {editingTitle ? (
                    <input
                      className="project-title-input"
                      aria-label="Project title"
                      value={titleDraft}
                      autoFocus
                      onChange={(event) => setTitleDraft(event.target.value)}
                      onBlur={() => commitProjectTitle()}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") commitProjectTitle();
                        if (event.key === "Escape") {
                          setTitleDraft(detail.project.title);
                          setEditingTitle(false);
                        }
                      }}
                    />
                  ) : (
                    <h2>
                      <button
                        type="button"
                        className="title-edit-button"
                        title="Edit project title"
                        onClick={() => {
                          setTitleDraft(detail.project.title);
                          setEditingTitle(true);
                        }}
                      >
                        {detail.project.title}
                      </button>
                    </h2>
                  )}
                  <span className="project-context">
                    {detail.takes.length} {detail.takes.length === 1 ? "take" : "takes"}
                    <i />
                    {detail.project.dit_profile}
                  </span>
                </div>
                <div className="workspace-actions">
                  <button
                    type="button"
                    className="panel-action builder-toggle"
                    aria-controls="composition-plan"
                    aria-expanded={planOpen}
                    onClick={() => setPlanOpen(true)}
                  >
                    <Icon name="settings" /> Plan
                  </button>
                  <button
                    type="button"
                    className="panel-action"
                    aria-controls="studio-inspector"
                    aria-expanded={inspectorOpen}
                    onClick={() => {
                      setInspectorTab("takes");
                      setInspectorOpen(true);
                    }}
                  >
                    <Icon name="wave" /> Takes
                  </button>
                  <label className="include-stems">
                    <input
                      type="checkbox"
                      checked={includeStems}
                      onChange={(event) => setIncludeStems(event.target.checked)}
                    />
                    Include stems
                  </label>
                  <a
                    className="export-link"
                    href={`${api.exportUrl(detail.project.id)}?include_stems=${includeStems}`}
                    download={`${detail.project.title}-export.zip`}
                  >
                    Export project (.zip)
                  </a>
                </div>
              </header>

              {jobStatusLabel && currentJob && (
                <div className={`job-strip status-${currentJob.status}`} aria-live="polite">
                  <span className="job-pulse" />
                  <strong>{jobStatusLabel}</strong>
                  {currentJob.status === "running" && currentJob.cancel_requested ? (
                    <small className="job-note">
                      ACE-Step cannot stop mid-render: it finishes the current pass and its output
                      is discarded.
                    </small>
                  ) : jobHeld ? (
                    <small className="job-note">
                      Queued before this version and never snapshotted. Retry it with the current
                      plan, or cancel it.
                    </small>
                  ) : null}
                  <span className="job-model">{health?.dit_loaded ?? "worker"}</span>
                  {jobIsActive && !currentJob.cancel_requested && (
                    <button type="button" className="text-button" onClick={cancelCurrentJob}>
                      Cancel
                    </button>
                  )}
                  {(currentJob.status === "error" ||
                    currentJob.status === "cancelled" ||
                    jobHeld) && (
                    <button type="button" className="text-button" onClick={retryCurrentJob}>
                      Retry
                    </button>
                  )}
                  {currentJob.status === "error" && (
                    <button type="button" className="text-button" onClick={copyJobSummary}>
                      Copy error summary
                    </button>
                  )}
                </div>
              )}

              {draftOffer && (
                <div className="draft-offer" role="alert">
                  <span>
                    Unsaved plan edits from {new Date(draftOffer.savedAt).toLocaleString()} are
                    older than the saved plan.
                  </span>
                  <button type="button" className="button-secondary" onClick={restoreDraftOffer}>
                    Restore my edits
                  </button>
                  <button type="button" className="text-button" onClick={discardDraftOffer}>
                    Discard
                  </button>
                </div>
              )}

              <div className="panes">
                <PlanInspector open={planOpen}>
                  <div className="pane-heading">
                    <div>
                      <span className="eyebrow">Instrument deck</span>
                      <h3>Plan</h3>
                    </div>
                    <div
                      className={`save-state save-${saveState}`}
                      role="status"
                      aria-live="polite"
                    >
                      <span />
                      {saveStateLabel}
                      {saveState === "error" && (
                        <button
                          type="button"
                          className="text-button"
                          onClick={() => retryPlanSave().catch(() => {})}
                        >
                          Retry now
                        </button>
                      )}
                      {saveState === "conflict" && (
                        <>
                          <button
                            type="button"
                            className="text-button"
                            onClick={() => resolvePlanConflict("keep-mine").catch(() => {})}
                          >
                            Keep my edits
                          </button>
                          <button
                            type="button"
                            className="text-button"
                            onClick={() => {
                              resolvePlanConflict("use-saved").catch(() => {});
                              if (activeId) {
                                refreshDetail(activeId).catch((err) => setErrorMsg(String(err)));
                              }
                            }}
                          >
                            Use saved plan
                          </button>
                        </>
                      )}
                    </div>
                    <button
                      type="button"
                      className="icon-button drawer-close"
                      aria-label="Close plan"
                      onClick={() => setPlanOpen(false)}
                    >
                      <Icon name="close" />
                    </button>
                  </div>

                  <div className="plan-scroll">
                    <p className="builder-intro">
                      Begin with the feeling. Add structure only when the song asks for it.
                    </p>

                    <section
                      className="builder-section builder-brief"
                      aria-labelledby="builder-brief-title"
                    >
                      <div className="builder-section-heading">
                        <span className="builder-index">01</span>
                        <div>
                          <span className="eyebrow">The brief</span>
                          <h4 id="builder-brief-title">Set the scene</h4>
                        </div>
                      </div>
                      <label className="field-label intent-field">
                        Starting idea
                        <textarea
                          value={detail.plan.query}
                          placeholder="Describe the song you want to explore..."
                          onChange={(event) => savePlanField("query", event.target.value)}
                        />
                      </label>
                      <p className="builder-tip">
                        A scene, a sound, and a feeling is plenty. You can refine it after the first
                        take.
                      </p>
                      <div
                        className="song-kind"
                        role="radiogroup"
                        aria-label="Song or instrumental"
                      >
                        <label className={detail.plan.instrumental ? "" : "selected"}>
                          <input
                            type="radio"
                            name="song-kind"
                            checked={!detail.plan.instrumental}
                            onChange={() => savePlanField("instrumental", false)}
                          />
                          Song
                        </label>
                        <label className={detail.plan.instrumental ? "selected" : ""}>
                          <input
                            type="radio"
                            name="song-kind"
                            aria-label="Instrumental"
                            checked={detail.plan.instrumental}
                            onChange={() => savePlanField("instrumental", true)}
                          />
                          Instrumental
                        </label>
                      </div>
                      {(capabilities?.presets.length ?? 0) > 0 && (
                        <label className="field-label preset-field">
                          Starting point
                          <select
                            aria-label="Preset"
                            value={presetPreview?.preset.id ?? ""}
                            disabled={busy}
                            onChange={(event) => previewPreset(event.target.value)}
                          >
                            <option value="">
                              {detail.plan.preset
                                ? `Preset applied: ${
                                    capabilities?.presets.find(
                                      (preset) => preset.id === detail.plan.preset?.id,
                                    )?.name ?? detail.plan.preset.id
                                  }`
                                : "No preset"}
                            </option>
                            {capabilities?.presets.map((preset) => (
                              <option key={preset.id} value={preset.id} title={preset.purpose}>
                                {preset.name}
                              </option>
                            ))}
                          </select>
                        </label>
                      )}
                      {presetPreview && (
                        <div className="preset-preview" role="region" aria-label="Preset preview">
                          <strong>
                            {capabilities?.presets.find((p) => p.id === presetPreview.preset.id)
                              ?.purpose ?? presetPreview.preset.id}
                          </strong>
                          {presetPreview.changes.length === 0 ? (
                            <small>Nothing would change.</small>
                          ) : (
                            <ul>
                              {presetPreview.changes.map((change) => (
                                <li key={change.field}>
                                  {change.field}: {String(change.from ?? "unset")} →{" "}
                                  {String(change.to)}
                                </li>
                              ))}
                            </ul>
                          )}
                          {presetPreview.warnings.map((warning) => (
                            <small key={warning}>{warning}</small>
                          ))}
                          <small>Your own fields and lyrics are not changed.</small>
                          <div>
                            <button
                              type="button"
                              className="button-secondary"
                              onClick={() => applyPresetPreview()}
                            >
                              Apply preset
                            </button>
                            <button
                              type="button"
                              className="text-button"
                              onClick={() => setPresetPreview(null)}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}
                      {hasInstrumentalConflict(detail.plan) && (
                        <div className="creation-conflict" role="alert">
                          <span>
                            Instrumental asks for no vocals, but the plan has lyrics. Lyre will not
                            delete them for you.
                          </span>
                          <button
                            type="button"
                            className="button-secondary"
                            onClick={() => savePlanField("lyrics", "")}
                          >
                            Clear lyrics
                          </button>
                          <button
                            type="button"
                            className="text-button"
                            onClick={() => savePlanField("instrumental", false)}
                          >
                            Keep vocals
                          </button>
                        </div>
                      )}
                      {detail.plan.brief?.source_query &&
                        (detail.plan.brief.source_query === detail.plan.query ? (
                          <p className="builder-tip brief-source">
                            The caption and lyrics marked “planned” came from this idea.
                          </p>
                        ) : (
                          <p className="builder-tip brief-source">
                            You changed the idea since it was planned. Generate will plan the new
                            one; text you wrote yourself is kept.
                          </p>
                        ))}
                    </section>

                    <button
                      type="button"
                      className="disclosure-button builder-disclosure"
                      aria-expanded={planDetailsOpen}
                      aria-controls="composition-details"
                      onClick={() => setPlanDetailsOpen((open) => !open)}
                    >
                      <span className="builder-disclosure-copy">
                        <span className="builder-index">02</span>
                        <span>
                          <strong>Open the sound controls</strong>
                          <small>Voice, timing, and arrangement</small>
                        </span>
                      </span>
                      <span>{planDetailsOpen ? "Collapse" : "Open"}</span>
                    </button>

                    <div
                      id="composition-details"
                      className="plan-details builder-details"
                      hidden={!planDetailsOpen}
                    >
                      <section
                        className="builder-section builder-song-details"
                        aria-labelledby="song-details-title"
                      >
                        <div className="builder-section-heading">
                          <span className="builder-index">A</span>
                          <div>
                            <span className="eyebrow">Direction</span>
                            <h4 id="song-details-title">Tune the pulse</h4>
                          </div>
                        </div>
                        <label className="field-label">
                          Caption <FieldOrigin plan={detail.plan} field="caption" />
                          <input
                            aria-label="Caption"
                            value={detail.plan.caption}
                            onChange={(event) => savePlanField("caption", event.target.value)}
                          />
                        </label>
                        <div className="plan-grid builder-settings-grid">
                          <label>
                            BPM <FieldOrigin plan={detail.plan} field="bpm" />
                            <input
                              aria-label="BPM"
                              type="number"
                              value={detail.plan.bpm ?? ""}
                              onChange={(event) =>
                                savePlanField(
                                  "bpm",
                                  event.target.value === "" ? null : Number(event.target.value),
                                )
                              }
                            />
                          </label>
                          <label>
                            Key <FieldOrigin plan={detail.plan} field="keyscale" />
                            <input
                              aria-label="Key"
                              value={detail.plan.keyscale ?? ""}
                              onChange={(event) =>
                                savePlanField(
                                  "keyscale",
                                  event.target.value === "" ? null : event.target.value,
                                )
                              }
                            />
                          </label>
                          <label>
                            Time signature <FieldOrigin plan={detail.plan} field="timesignature" />
                            <input
                              aria-label="Time signature"
                              value={detail.plan.timesignature}
                              onChange={(event) =>
                                savePlanField("timesignature", event.target.value)
                              }
                            />
                          </label>
                          <label>
                            Duration (sec) <FieldOrigin plan={detail.plan} field="duration_sec" />
                            <input
                              aria-label="Duration (sec)"
                              type="number"
                              value={detail.plan.duration_sec ?? ""}
                              placeholder="auto"
                              onChange={(event) =>
                                savePlanField(
                                  "duration_sec",
                                  event.target.value === "" ? null : Number(event.target.value),
                                )
                              }
                            />
                          </label>
                          <label>
                            Language <FieldOrigin plan={detail.plan} field="vocal_language" />
                            <input
                              aria-label="Language"
                              value={detail.plan.vocal_language}
                              onChange={(event) =>
                                savePlanField("vocal_language", event.target.value)
                              }
                            />
                          </label>
                        </div>
                        <div className="toggle-row builder-switches">
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              aria-label="Allow caption rewrite (Custom mode LM thinking)"
                              checked={detail.plan.caption_rewrite}
                              onChange={(event) =>
                                savePlanField("caption_rewrite", event.target.checked)
                              }
                            />
                            <span>
                              <strong>Allow caption rewrite</strong>
                              <small>
                                Off keeps the caption you wrote exactly. On lets the language model
                                rework it; the take records what it used.
                              </small>
                            </span>
                          </label>
                        </div>
                      </section>

                      <section
                        className="builder-section builder-lyrics"
                        aria-labelledby="builder-lyrics-title"
                      >
                        <div className="builder-section-heading">
                          <span className="builder-index">B</span>
                          <div>
                            <span className="eyebrow">Words</span>
                            <h4 id="builder-lyrics-title">Give it a voice</h4>
                          </div>
                        </div>
                        <div className="field-label lyrics-field">
                          <div className="lyrics-label-row">
                            <label htmlFor="plan-lyrics">Lyrics</label>
                            <FieldOrigin plan={detail.plan} field="lyrics" />
                            <span>Optional</span>
                          </div>
                          <div
                            className="lyrics-tag-palette"
                            aria-label="Insert song structure tag"
                          >
                            {STRUCTURE_TAGS.map((tag) => (
                              <button
                                key={tag}
                                type="button"
                                className="lyrics-tag-button"
                                onMouseDown={(event) => event.preventDefault()}
                                onClick={() => insertLyricsTag(`[${tag}]`)}
                              >
                                [{tag}]
                              </button>
                            ))}
                          </div>
                          <textarea
                            id="plan-lyrics"
                            ref={lyricsTextareaRef}
                            value={detail.plan.lyrics}
                            placeholder="Add a lyric or a hook, or leave this blank..."
                            onChange={(event) => savePlanField("lyrics", event.target.value)}
                          />
                        </div>
                      </section>

                      <section
                        className="plan-sections builder-section builder-arrangement"
                        aria-labelledby="builder-arrangement-title"
                      >
                        <div className="plan-sections-header">
                          <div>
                            <span className="eyebrow">Map</span>
                            <span id="builder-arrangement-title" className="plan-sections-title">
                              Cue map
                            </span>
                            <small>
                              {detail.plan.sections.length} mapped, or draw on the waveform
                            </small>
                          </div>
                          <button
                            type="button"
                            className="button-secondary"
                            onClick={() => addSection()}
                          >
                            <Icon name="add" /> Add section
                          </button>
                        </div>
                        {detail.plan.sections.length === 0 && (
                          <p className="empty-copy">
                            No sections yet. Add one here or draw a region on the waveform.
                          </p>
                        )}
                        <ul className="section-list">
                          {detail.plan.sections.map((section, index) => (
                            <li key={index} className="section-row">
                              <span className="section-index">
                                {String(index + 1).padStart(2, "0")}
                              </span>
                              <input
                                className="section-name"
                                placeholder="name"
                                value={section.name}
                                onChange={(event) =>
                                  updateSection(index, { name: event.target.value })
                                }
                              />
                              <div className="section-range">
                                <input
                                  className="section-time"
                                  type="number"
                                  min={0}
                                  step={0.1}
                                  title="start (sec)"
                                  value={section.start_sec}
                                  onChange={(event) =>
                                    updateSection(index, { start_sec: Number(event.target.value) })
                                  }
                                />
                                <span className="section-sep">-</span>
                                <input
                                  className="section-time"
                                  type="number"
                                  min={0}
                                  step={0.1}
                                  title="end (sec)"
                                  value={section.end_sec}
                                  onChange={(event) =>
                                    updateSection(index, { end_sec: Number(event.target.value) })
                                  }
                                />
                              </div>
                              <input
                                className="section-lyrics"
                                placeholder="lyrics snippet"
                                value={section.lyrics}
                                onChange={(event) =>
                                  updateSection(index, { lyrics: event.target.value })
                                }
                              />
                              <button
                                type="button"
                                className="icon-button"
                                aria-label={`Delete section ${index + 1}`}
                                onClick={() => removeSection(index)}
                              >
                                <Icon name="delete" />
                                <span className="delete-text">Delete</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      </section>
                    </div>

                    <OperationDock>
                      <div className="operation-tabs" role="tablist" aria-label="Studio operations">
                        {(["create", "transform", "tracks"] as OperationGroup[]).map((group) => (
                          <button
                            key={group}
                            type="button"
                            role="tab"
                            aria-selected={operationGroup === group}
                            onClick={() => setOperationGroup(group)}
                          >
                            {group}
                          </button>
                        ))}
                      </div>

                      <section
                        className={`operation-group create-group ${operationGroup === "create" ? "is-active" : ""}`}
                        aria-label="Create operations"
                      >
                        <div className="operation-heading">
                          <span>01</span>
                          <div>
                            <strong>Play</strong>
                            <small>Turn this idea into a first take</small>
                          </div>
                        </div>
                        <div className="operation-controls create-controls">
                          <label className="lora-select">
                            Style pack
                            <select
                              value={selectedLoraId ?? ""}
                              onChange={(event) => setSelectedLoraId(event.target.value || null)}
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
                          {selectedLora && (
                            <span
                              className="lora-active-badge"
                              title={`Generate/Cover/Repaint will run against the studio_ops base model with style pack "${selectedLora.name}" (SPEC.md sec 4.4)`}
                            >
                              Style pack: {selectedLora.name}
                            </span>
                          )}
                          <label className="seed-input">
                            Seed
                            <input
                              type="number"
                              step={1}
                              min={-1}
                              placeholder="-1"
                              value={seedInput}
                              onChange={(event) => setSeedInput(event.target.value)}
                              disabled={busy}
                              title="Fixed seed for Generate/Cover/Repaint; empty or -1 lets the worker pick and record one (SPEC.md sec 7.3)"
                            />
                          </label>
                          <div
                            className="dit-profile-picker"
                            role="group"
                            aria-label="DiT profile"
                            title="Project default DiT checkpoint for Generate/Cover/Repaint (SPEC.md sec 4.1); a style pack always forces studio_ops"
                          >
                            <span className="dit-profile-caption">DiT</span>
                            {DIT_PROFILE_OPTIONS.map((profile) => {
                              const report = capabilities?.profiles.find((p) => p.id === profile);
                              const unavailable = report?.supported === false;
                              return (
                                <button
                                  key={profile}
                                  type="button"
                                  className={`dit-profile-option ${detail.project.dit_profile === profile ? "selected" : ""} ${unavailable ? "unavailable" : ""}`}
                                  disabled={busy || unavailable}
                                  onClick={() => setDitProfile(profile)}
                                  title={profileTitle(profile, report)}
                                >
                                  {profile}
                                </button>
                              );
                            })}
                          </div>
                          {(() => {
                            const intent = generateIntent(detail.plan);
                            const copy = INTENT_COPY[intent];
                            const locked = (
                              [
                                ["caption", "caption"],
                                ["lyrics", "lyrics"],
                                ["bpm", "BPM"],
                                ["keyscale", "key"],
                                ["timesignature", "time signature"],
                                ["duration_sec", "duration"],
                                ["vocal_language", "language"],
                              ] as const
                            )
                              .filter(([field]) => originOf(detail.plan, field) === "user")
                              .map(([, name]) => name);
                            return (
                              <div className="generate-intent" aria-live="polite">
                                <strong>{copy.title}</strong>
                                <small>{copy.detail}</small>
                                <small>
                                  {locked.length > 0
                                    ? `Yours, kept as set: ${locked.join(", ")}.`
                                    : "Nothing is locked yet."}
                                </small>
                                {intent === "another-take" && detail.plan.query.trim() && (
                                  <button
                                    type="button"
                                    className="text-button"
                                    disabled={busy}
                                    onClick={() => generate("from_idea")}
                                    title="Run the planner again from the idea. Text you wrote yourself is kept."
                                  >
                                    Re-plan from idea
                                  </button>
                                )}
                                {busy && activeJobs.length > 0 && (
                                  <small>
                                    A take is queued with the plan as it was when you clicked. Edits
                                    now apply to the next take.
                                  </small>
                                )}
                              </div>
                            );
                          })()}
                          <button
                            type="button"
                            className="generate-button"
                            onClick={() => generate()}
                            disabled={busy || hasInstrumentalConflict(detail.plan)}
                            title={
                              hasInstrumentalConflict(detail.plan)
                                ? "Resolve the Instrumental and lyrics conflict first"
                                : "Shortcuts: G generate; Space play/pause; Up/Down previous/next take; Ctrl/Cmd+S save plan"
                            }
                          >
                            <Icon name="spark" /> Generate
                          </button>
                        </div>
                      </section>

                      <section
                        className={`operation-group transform-group ${operationGroup === "transform" ? "is-active" : ""}`}
                        aria-label="Transform operations"
                      >
                        <div className="operation-heading">
                          <span>02</span>
                          <div>
                            <strong>Bend</strong>
                            <small>Reshape the selected sound</small>
                          </div>
                        </div>
                        <div className="operation-controls">
                          <label className="cover-strength">
                            Strength
                            <input
                              type="number"
                              min={0}
                              max={1}
                              step={0.05}
                              value={coverStrength}
                              onChange={(event) => setCoverStrength(Number(event.target.value))}
                            />
                          </label>
                          <button
                            type="button"
                            onClick={cover}
                            disabled={
                              busy ||
                              (!selectedTakeId && !uploadedSourcePath) ||
                              !!selectedTake?.error
                            }
                            title={
                              selectedTakeId || uploadedSourcePath
                                ? undefined
                                : "Select a take or drop a file first"
                            }
                          >
                            Cover
                          </button>
                          <button
                            type="button"
                            onClick={repaint}
                            disabled={
                              busy ||
                              (!selectedTakeId && !uploadedSourcePath) ||
                              (!!selectedTakeId && !region) ||
                              !!selectedTake?.error
                            }
                            title={
                              !selectedTakeId && !uploadedSourcePath
                                ? "Select a take or drop a file first"
                                : selectedTakeId && !region
                                  ? "Drag a region on the waveform first"
                                  : undefined
                            }
                          >
                            Repaint
                          </button>
                        </div>
                      </section>

                      <section
                        className={`operation-group tracks-group ${operationGroup === "tracks" ? "is-active" : ""}`}
                        aria-label="Track operations"
                      >
                        <div className="operation-heading">
                          <span>03</span>
                          <div>
                            <strong>Pull apart</strong>
                            <small>Work with the parts inside a take</small>
                          </div>
                        </div>
                        <div className="operation-controls">
                          <label className="track-name">
                            Track name / classes
                            <input
                              placeholder="vocals, drums, bass..."
                              value={trackName}
                              onChange={(event) => setTrackName(event.target.value)}
                            />
                          </label>
                          <button
                            type="button"
                            onClick={extract}
                            disabled={
                              busy || !selectedTakeId || !trackName.trim() || !!selectedTake?.error
                            }
                            title={
                              !selectedTakeId
                                ? "Select a take first"
                                : !trackName.trim()
                                  ? "Enter a track name first"
                                  : undefined
                            }
                          >
                            Extract
                          </button>
                          <button
                            type="button"
                            onClick={lego}
                            disabled={
                              busy || !selectedTakeId || !trackName.trim() || !!selectedTake?.error
                            }
                            title={
                              !selectedTakeId
                                ? "Select a take first"
                                : !trackName.trim()
                                  ? "Enter a track name first"
                                  : undefined
                            }
                          >
                            Lego
                          </button>
                          <button
                            type="button"
                            onClick={complete}
                            disabled={
                              busy || !selectedTakeId || !trackName.trim() || !!selectedTake?.error
                            }
                            title={
                              !selectedTakeId
                                ? "Select a take first"
                                : !trackName.trim()
                                  ? "Enter a track name / classes first"
                                  : undefined
                            }
                          >
                            Complete
                          </button>
                        </div>
                      </section>
                    </OperationDock>
                  </div>
                </PlanInspector>

                <StudioStage>
                  <div className="stage-heading">
                    <div>
                      <span className="eyebrow">Sound field</span>
                      <h3>Listen</h3>
                    </div>
                    {selectedTake ? (
                      <div className="selected-take-summary">
                        <span>{selectedTake.task_type}</span>
                        <code>seed {selectedTake.seed}</code>
                        <code>
                          {selectedTake.duration_sec != null
                            ? `${selectedTake.duration_sec.toFixed(1)}s`
                            : "n/a"}
                        </code>
                      </div>
                    ) : (
                      <span className="stage-empty-label">No take selected</span>
                    )}
                  </div>

                  <div
                    className={`source-shelf ${uploadedSourcePath ? "has-source" : ""}`}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={handleDropAudio}
                  >
                    <Icon name="wave" />
                    {uploadedSourcePath ? (
                      <span className="upload-dropzone-file">
                        <span>
                          <small>External source</small>
                          {uploadedSourceName ?? uploadedSourcePath}
                        </span>
                        <button
                          type="button"
                          className="button-secondary"
                          onClick={clearUploadedSource}
                        >
                          Clear
                        </button>
                      </span>
                    ) : (
                      <span className="source-copy">
                        <strong>Drop WAV or MP3</strong>
                        <small>Use local audio for Cover or Repaint</small>
                      </span>
                    )}
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/wav,audio/mpeg,.wav,.mp3"
                      className="sr-only"
                      aria-label="Choose audio source"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) uploadSourceFile(file);
                        event.target.value = "";
                      }}
                    />
                    {!uploadedSourcePath && (
                      <button
                        type="button"
                        className="button-secondary"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        Choose audio
                      </button>
                    )}
                    {uploadError && <p className="upload-error">{uploadError}</p>}
                  </div>

                  <div className="waveform-canvas">
                    {selectedTakeId && !selectedTake?.error ? (
                      <div ref={waveformContainerRef} className="waveform-wavesurfer" />
                    ) : (
                      <div className="waveform-empty">
                        <Icon name="wave" />
                        <p>
                          {selectedTake?.error
                            ? "This take did not produce playable audio."
                            : "Select a take to inspect its waveform."}
                        </p>
                      </div>
                    )}
                  </div>

                  <StudioPlayer>
                    <button
                      type="button"
                      className="transport-button"
                      aria-label={waveformPlaying ? "Pause selected take" : "Play selected take"}
                      disabled={!selectedTake || !!selectedTake.error}
                      onClick={toggleWaveformPlayback}
                    >
                      <Icon name={waveformPlaying ? "pause" : "play"} />
                    </button>
                    <span className="transport-time">{formatClock(waveformCurrentTime)}</span>
                    <input
                      className="stage-scrubber"
                      type="range"
                      min={0}
                      max={Math.max(waveformDuration, 0.01)}
                      step={0.1}
                      value={Math.min(waveformCurrentTime, Math.max(waveformDuration, 0.01))}
                      disabled={!selectedTake}
                      aria-label="Seek selected take"
                      onChange={(event) => {
                        seekWaveform(Number(event.target.value));
                      }}
                    />
                    <span className="transport-time">
                      {formatClock(waveformDuration || selectedTake?.duration_sec || 0)}
                    </span>
                    <LoudnessMeter audioEl={waveformMediaEl} />
                  </StudioPlayer>

                  <div className={`region-bar ${region ? "has-region" : ""}`}>
                    <span>
                      {region
                        ? `Region: ${region.start.toFixed(1)}s - ${region.end.toFixed(1)}s`
                        : "Drag across the waveform to select a repaint region."}
                    </span>
                    <div>
                      <button
                        type="button"
                        className="button-secondary"
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
                      {region && (
                        <button type="button" className="text-button" onClick={clearRegion}>
                          Clear region
                        </button>
                      )}
                    </div>
                  </div>

                  {compareTakeId &&
                    (() => {
                      const compareTake = detail.takes.find((take) => take.id === compareTakeId);
                      if (!compareTake || compareTake.error) return null;
                      return (
                        <section className="compare" aria-label="A/B comparison">
                          <div className="compare-heading">
                            <div>
                              <span className="eyebrow">Instant audition</span>
                              <h3>Compare</h3>
                            </div>
                            <button
                              type="button"
                              className="icon-button"
                              aria-label="Close compare"
                              onClick={() => setCompareTakeId(null)}
                            >
                              <Icon name="close" />
                            </button>
                          </div>
                          <div className="compare-panel">
                            <div className="compare-slot">
                              <span className="compare-label">A: selected</span>
                              {selectedTake && !selectedTake.error ? (
                                <audio
                                  controls
                                  src={api.takeAudioUrl(detail.project.id, selectedTake.id)}
                                />
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
                              <audio
                                controls
                                src={api.takeAudioUrl(detail.project.id, compareTakeId)}
                              />
                            </div>
                          </div>
                        </section>
                      );
                    })()}
                </StudioStage>

                <TakesRail open={inspectorOpen}>
                  <div className="inspector-tabs" role="tablist" aria-label="Inspector">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={inspectorTab === "takes"}
                      onClick={() => setInspectorTab("takes")}
                    >
                      Takes <span>{detail.takes.length}</span>
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={inspectorTab === "styles"}
                      onClick={() => setInspectorTab("styles")}
                    >
                      Style packs <span>{loras.length}</span>
                    </button>
                    <button
                      type="button"
                      className="icon-button drawer-close"
                      aria-label="Close inspector"
                      onClick={() => setInspectorOpen(false)}
                    >
                      <Icon name="close" />
                    </button>
                  </div>

                  <section className="takes" hidden={inspectorTab !== "takes"}>
                    <div className="sr-only">
                      <h3>Takes</h3>
                    </div>
                    {detail.takes.length === 0 && (
                      <div className="inspector-empty">
                        <Icon name="wave" />
                        <p>No takes yet.</p>
                        <span>Your first generation will appear here.</span>
                      </div>
                    )}
                    <ul className="take-list">
                      {detail.takes.map((take, index) => (
                        <li
                          key={take.id}
                          className={[
                            take.id === selectedTakeId ? "selected" : "",
                            take.id === detail.project.active_take_id ? "active-take" : "",
                          ]
                            .filter(Boolean)
                            .join(" ")}
                          tabIndex={0}
                          aria-label={`Take ${detail.takes.length - index}, ${take.task_type}, seed ${take.seed}`}
                          onClick={() => setSelectedTakeId(take.id)}
                          onKeyDown={(event: ReactKeyboardEvent<HTMLLIElement>) => {
                            if (
                              (event.key === "Enter" || event.key === " ") &&
                              event.target === event.currentTarget
                            ) {
                              event.preventDefault();
                              setSelectedTakeId(take.id);
                            }
                          }}
                        >
                          <div className="take-header">
                            <span className="take-number">
                              {String(detail.takes.length - index).padStart(2, "0")}
                            </span>
                            <div className="take-identity">
                              <strong>{take.task_type}</strong>
                              <span>seed {take.seed}</span>
                            </div>
                            <div className="take-statuses">
                              {take.id === detail.project.active_take_id && (
                                <span className="active-take-badge">active</span>
                              )}
                              {take.id === selectedTakeId && (
                                <span className="source-take-badge">source</span>
                              )}
                            </div>
                            <button
                              type="button"
                              className={`icon-button favorite-btn ${take.favorite ? "favorited" : ""}`}
                              title={take.favorite ? "Unfavorite" : "Favorite"}
                              aria-label={`${take.favorite ? "Unfavorite" : "Favorite"} take ${take.id}`}
                              onClick={(event) => {
                                event.stopPropagation();
                                toggleTakeFavorite(take);
                              }}
                            >
                              <Icon name="star" />
                            </button>
                          </div>
                          <div className="take-facts">
                            <span>
                              {take.duration_sec != null
                                ? `${take.duration_sec.toFixed(1)}s`
                                : "n/a"}
                            </span>
                            <span>score {take.score ?? "n/a"}</span>
                            <span>{take.dit_profile}</span>
                          </div>
                          {take.lora_id &&
                            (() => {
                              const pack = loras.find((lora) => lora.id === take.lora_id);
                              return pack && !pack.error ? (
                                <button
                                  type="button"
                                  className="take-lora-badge"
                                  title={`Generated with style pack "${pack.name}". Select it for the next Generate/Cover/Repaint`}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setSelectedLoraId(pack.id);
                                  }}
                                >
                                  style: {pack.name}
                                </button>
                              ) : (
                                <span
                                  className="take-lora-badge"
                                  title={
                                    pack
                                      ? `Generated with style pack "${pack.name}" (${take.lora_id}). Training failed, so it cannot be loaded`
                                      : `Generated with style pack ${take.lora_id} (not found in this project)`
                                  }
                                >
                                  style: {pack ? pack.name : take.lora_id.slice(0, 8)}
                                </span>
                              );
                            })()}
                          <textarea
                            className="take-notes"
                            placeholder="Notes..."
                            value={take.notes}
                            onClick={(event) => event.stopPropagation()}
                            onChange={(event) => saveTakeNotes(take.id, event.target.value)}
                            onBlur={() =>
                              flushTakeNotes(take.id).catch((err) => setErrorMsg(String(err)))
                            }
                          />
                          <div className="take-player-row">
                            {take.error ? (
                              <span className="take-error">failed: {take.error}</span>
                            ) : (
                              <TakeAudioPlayer
                                projectId={detail.project.id}
                                takeId={take.id}
                                registerRef={registerAudioRef}
                                onSelect={() => setSelectedTakeId(take.id)}
                              />
                            )}
                          </div>
                          <div className="take-actions">
                            <button
                              type="button"
                              disabled={take.id === detail.project.active_take_id || !!take.error}
                              onClick={(event) => {
                                event.stopPropagation();
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
                              onClick={(event) => {
                                event.stopPropagation();
                                setCompareTakeId((current) =>
                                  current === take.id ? null : take.id,
                                );
                              }}
                            >
                              {take.id === compareTakeId ? "Comparing" : "Compare"}
                            </button>
                            {take.parent_take_id &&
                              (detail.takes.find(
                                (candidate) => candidate.id === take.parent_take_id,
                              ) ? (
                                <button
                                  type="button"
                                  className="parent-take-link"
                                  onClick={(event) => {
                                    event.stopPropagation();
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
                            {!take.error && (
                              <>
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
                          </div>
                        </li>
                      ))}
                    </ul>
                  </section>

                  <StylePackPanel
                    active={inspectorTab === "styles"}
                    loras={loras}
                    trainingJobs={trainingJobs}
                    takes={detail.takes}
                    loraSourceIds={loraSourceIds}
                    onToggleSource={toggleLoraSource}
                    loraName={loraName}
                    onNameChange={setLoraName}
                    onTrain={trainLora}
                    busy={busy}
                    activeJobAction={activeJobAction}
                  />
                </TakesRail>
              </div>
            </div>
          )}
        </main>

        {(libraryOpen || planOpen || inspectorOpen) && (
          <button
            type="button"
            className="drawer-scrim"
            aria-label="Close open panel"
            onClick={() => {
              setLibraryOpen(false);
              setPlanOpen(false);
              setInspectorOpen(false);
            }}
          />
        )}
      </div>

      <ConfirmationDialog request={confirmation} onDecision={resolveConfirmation} />
    </AppShell>
  );
}
