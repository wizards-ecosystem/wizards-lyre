import { useEffect, useState } from "react";
import { api, Plan, ProjectDetail, ProjectSummary } from "./api";

export default function App() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [newQuery, setNewQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

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
      setActiveId(project.id);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function savePlanField<K extends keyof Plan>(key: K, value: Plan[K]) {
    if (!activeId || !detail) return;
    const plan = { ...detail.plan, [key]: value };
    setDetail({ ...detail, plan });
    try {
      await api.savePlan(activeId, plan);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  async function generate() {
    if (!activeId) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      const job = await api.generate(activeId);
      if (job.status === "error") {
        setErrorMsg(job.error ?? "generate job failed");
      }
      await refreshDetail(activeId);
    } catch (err) {
      setErrorMsg(String(err));
    } finally {
      setBusy(false);
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
              onClick={() => setActiveId(p.id)}
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
                {busy ? "Generating…" : "Generate"}
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
                      <span>{take.duration_sec.toFixed(1)}s</span>
                    </div>
                    <audio controls src={api.takeAudioUrl(detail.project.id, take.id)} />
                    <a
                      href={api.takeAudioUrl(detail.project.id, take.id)}
                      download={`${take.id}.wav`}
                    >
                      download
                    </a>
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
