// Thin client for the FastAPI backend. See SPEC.md sec 8 for the contract.

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
  duration_sec: number;
  caption: string;
  lyrics: string;
  bpm: number | null;
  keyscale: string | null;
  created_at: string;
  score: number | null;
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
  health: () => request<{ ok: boolean; gpu: string; dit_loaded: string | null }>("/api/health"),
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
  generate: (id: string) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({ action: "generate", dit_profile: "iterate", seed: -1 }),
    }),
  takeAudioUrl: (projectId: string, takeId: string) =>
    `/api/projects/${projectId}/takes/${takeId}/audio`,
};
