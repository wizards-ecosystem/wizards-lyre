import { useEffect, useRef, useState } from "react";
import { api, Job, Plan, ProjectDetail, ProjectSummary } from "./api";

const JOB_POLL_INTERVAL_MS = 1000;
// Real ACE-Step generation can take a while (SPEC.md sec 5: it queues
// behind a dedicated worker process); give it a generous ceiling before
// giving up on polling rather than declaring failure too early.
const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000;

// Coalesce rapid keystrokes into one PUT instead of firing one per
// keystroke (which can complete out of order and let an older request
// overwrite a newer edit on disk).
const PLAN_SAVE_DEBOUNCE_MS = 500;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// A job only finishes async, via server.jobs' queued -> running -> done|error
// lifecycle (SPEC.md sec 5) -- the enqueue response is just the initial
// `queued` row, so the caller has to keep polling /api/jobs/{id} itself.
async function pollJob(jobId: string, onUpdate?: (job: Job) => void): Promise<Job> {
  const deadline = Date.now() + JOB_POLL_TIMEOUT_MS;
  for (;;) {
    const job = await api.getJob(jobId);
    onUpdate?.(job);
    if (job.status === "done" || job.status === "error") {
      return job;
    }
    if (Date.now() > deadline) {
      throw new Error(`job ${jobId} is still ${job.status} after ${JOB_POLL_TIMEOUT_MS / 1000}s`);
    }
    await sleep(JOB_POLL_INTERVAL_MS);
  }
}

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [newQuery, setNewQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyStatus, setBusyStatus] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

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

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, []);

  function enqueueSave(): Promise<void> {
    const runSave = saveChainRef.current.then(async () => {
      const pending = pendingSaveRef.current;
      if (!pending) return;
      pendingSaveRef.current = null;
      await api.savePlan(pending.projectId, pending.plan);
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
  // silently drop an edit when the user switches projects mid-debounce).
  async function switchActiveProject(id: string): Promise<void> {
    try {
      await flushPendingPlanSave();
    } catch (err) {
      setErrorMsg(String(err));
    }
    setActiveId(id);
  }

  async function refreshProjects() {
    setProjects(await api.listProjects());
  }

  async function refreshDetail(id: string) {
    setDetail(await api.getProject(id));
  }

  useEffect(() => {
    refreshProjects().catch((err) => setErrorMsg(String(err)));
  }, []);

  useEffect(() => {
    if (activeId) {
      refreshDetail(activeId).catch((err) => setErrorMsg(String(err)));
    } else {
      setDetail(null);
    }
  }, [activeId]);

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

  async function generate() {
    if (!activeId) return;
    setBusy(true);
    setBusyStatus("queued");
    setErrorMsg(null);
    try {
      // The plan can still be mid-debounce (or an earlier save still in
      // flight) when Generate is clicked -- without this, the job can read
      // the plan from before the user's last edit (e.g. the query/caption
      // they just typed).
      await flushPendingPlanSave();
      const queued = await api.generate(activeId);
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

  return (
    <div className="app">
      <aside className="library">
        <h1>Wizard's Bard</h1>
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
        <ul className="project-list">
          {projects.map((p) => (
            <li
              key={p.id}
              className={p.id === activeId ? "active" : ""}
              onClick={() => switchActiveProject(p.id)}
            >
              {p.title}
            </li>
          ))}
        </ul>
      </aside>

      <main className="workspace">
        {errorMsg && <div className="error">{errorMsg}</div>}
        {!detail && <p className="hint">Select or create a project.</p>}
        {detail && (
          <>
            <h2>{detail.project.title}</h2>

            <section className="plan">
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
              <button onClick={generate} disabled={busy}>
                {busy ? `Generating… (${busyStatus ?? "queued"})` : "Generate"}
              </button>
            </section>

            <section className="takes">
              <h3>Takes</h3>
              {detail.takes.length === 0 && <p className="hint">No takes yet.</p>}
              <ul>
                {detail.takes.map((take) => (
                  <li key={take.id}>
                    <div className="take-meta">
                      <span>{take.task_type}</span>
                      <span>seed {take.seed}</span>
                      <span>
                        {take.duration_sec != null ? `${take.duration_sec.toFixed(1)}s` : "—"}
                      </span>
                    </div>
                    {take.error ? (
                      // A take whose generation failed has no audio file
                      // (SPEC.md sec 10 point 5) -- show the error instead
                      // of an <audio> that would just 404.
                      <span className="take-error">failed: {take.error}</span>
                    ) : (
                      <>
                        <audio controls src={api.takeAudioUrl(detail.project.id, take.id)} />
                        <a
                          href={api.takeAudioUrl(detail.project.id, take.id)}
                          download={`${take.id}.wav`}
                        >
                          download
                        </a>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
