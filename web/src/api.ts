// Thin client for the FastAPI backend. See SPEC.md sec 8 for the contract.

export interface Health {
  ok: boolean;
  gpu: string;
  dit_loaded: string | null;
}

export interface ProjectSummary {
  id: string;
  title: string;
  updated_at: string;
  active_take_id: string | null;
}

export interface Project {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  dit_profile: string;
  lm_model: string;
  active_take_id: string | null;
}

export interface Plan {
  query: string;
  caption: string;
  negative: string[];
  lyrics: string;
  instrumental: boolean;
  vocal_language: string;
  bpm: number | null;
  keyscale: string | null;
  timesignature: string;
  duration_sec: number;
  sections: unknown[];
}

export interface Take {
  id: string;
  parent_take_id: string | null;
  task_type: string;
  dit_profile: string;
  seed: number;
  // A take whose generation failed is still recorded (SPEC.md sec 10 point
  // 5) with duration/caption/lyrics/bpm/keyscale left null and `error` set.
  duration_sec: number | null;
  caption: string | null;
  lyrics: string | null;
  bpm: number | null;
  keyscale: string | null;
  created_at: string;
  score: number | null;
  // True only when ACE-Step actually produced lyric timestamps for this
  // take and the server wrote lyrics.lrc (SPEC.md sec 7 / sec 12 Phase 4) --
  // the UI must not guess from a 404 whether one exists.
  has_lrc: boolean;
  error: string | null;
  track_name: string | null;
}

export interface ProjectDetail {
  project: Project;
  plan: Plan;
  takes: Take[];
}

export interface Job {
  id: string;
  project_id: string;
  action: string;
  dit_profile: string;
  status: string;
  take_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  listProjects: () => request<ProjectSummary[]>("/api/projects"),
  createProject: (title: string, query: string) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ title, query }),
    }),
  getProject: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  savePlan: (id: string, plan: Plan) =>
    request<Plan>(`/api/projects/${id}/plan`, {
      method: "PUT",
      body: JSON.stringify(plan),
    }),
  // dit_profile is intentionally omitted: the server falls back to the
  // project's own persisted dit_profile (PATCH /api/projects/{id}) when the
  // job body doesn't provide one. Hardcoding "iterate" here would silently
  // override that project-level choice on every generate request.
  generate: (id: string) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({ action: "generate", seed: -1 }),
    }),
  cover: (id: string, sourceTakeId: string, strength: number) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "cover",
        source_take_id: sourceTakeId,
        audio_cover_strength: strength,
        seed: -1,
      }),
    }),
  repaint: (id: string, sourceTakeId: string, start: number, end: number) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "repaint",
        source_take_id: sourceTakeId,
        repainting_start: start,
        repainting_end: end,
        seed: -1,
      }),
    }),
  extract: (id: string, sourceTakeId: string, trackName: string) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "extract",
        dit_profile: "studio_ops",
        source_take_id: sourceTakeId,
        track_name: trackName,
        seed: -1,
      }),
    }),
  lego: (
    id: string,
    sourceTakeId: string,
    trackName: string,
    region: { start: number; end: number } | null,
  ) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "lego",
        dit_profile: "studio_ops",
        source_take_id: sourceTakeId,
        track_name: trackName,
        seed: -1,
        ...(region ? { repainting_start: region.start, repainting_end: region.end } : {}),
      }),
    }),
  complete: (id: string, sourceTakeId: string, trackName: string) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "complete",
        dit_profile: "studio_ops",
        source_take_id: sourceTakeId,
        track_name: trackName,
        seed: -1,
      }),
    }),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  takeAudioUrl: (projectId: string, takeId: string) =>
    `/api/projects/${projectId}/takes/${takeId}/audio`,
  takeLrcUrl: (projectId: string, takeId: string) =>
    `/api/projects/${projectId}/takes/${takeId}/lrc`,
  exportUrl: (projectId: string) => `/api/projects/${projectId}/export`,
};
