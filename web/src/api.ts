// Thin client for the FastAPI backend. See SPEC.md sec 8 for the contract.

export interface Health {
  ok: boolean;
  gpu: string;
  dit_loaded: string | null;
}

// Opt-in Remote GPU (docs/remote-gpu.md). Mirrors server/remote_gpu/service.py.
export interface RemoteGpuQuote {
  gpu: string;
  cloud: string;
  disk_gb: number;
  hourly_usd: number | null;
  memory_gb: number | null;
  availability: string | null;
}

export interface RemoteGpuSession {
  id: string;
  state: "starting" | "ready" | "stopping" | "error";
  detail: string;
  gpu: string;
  elapsed_sec: number;
  hourly_usd: number | null;
  cost_estimate_usd: number | null;
}

export interface RemoteGpuHost {
  connected: boolean;
  ready: boolean;
  message?: string | null;
  gpu?: string | null;
  stale_build?: boolean;
  loaded_dit_profile?: string | null;
  error: string | null;
}

export interface RemoteGpuStatus {
  provisioner: string;
  label: string;
  can_provision: boolean;
  stop_on_exit: boolean;
  quote: RemoteGpuQuote | null;
  connection: { base_url: string } | null;
  session: RemoteGpuSession | null;
  host: RemoteGpuHost | null;
}

export interface ProjectSummary {
  id: string;
  title: string;
  updated_at: string;
  active_take_id: string | null;
  favorite: boolean;
}

export interface Project {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  dit_profile: string;
  lm_model: string;
  active_take_id: string | null;
  favorite: boolean;
}

// A named song-structure region (SPEC.md sec 7.2). Purely a UI concern --
// region labels on the waveform; ACE-Step's text2music never reads it -- so
// the backend round-trips it verbatim inside plan.json.
export interface Section {
  name: string;
  start_sec: number;
  end_sec: number;
  lyrics: string;
}

export type CreationMode = "from_idea" | "from_brief";
export type FieldOrigin = "user" | "planner" | "preset" | "default" | "unknown";
export type TrackedField =
  "caption" | "lyrics" | "vocal_language" | "bpm" | "keyscale" | "timesignature" | "duration_sec";

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
  duration_sec: number | null;
  sections: Section[];
  // SPEC.md sec 9.2: Custom-mode checkbox -- whether ACE-Step's LM
  // ("thinking") is allowed to rewrite the user's caption. Simple mode
  // ignores this and always runs with thinking=true.
  caption_rewrite: boolean;
  // SPEC.md sec 16.3 creation contract. The server derives all four on
  // every write; the UI reads them and may name an origin or mode
  // explicitly. Optional so a plan from an older server still type-checks.
  creation_mode?: CreationMode;
  field_origins?: Partial<Record<TrackedField, FieldOrigin>>;
  revision?: number;
  brief?: { source_query: string; job_id: string; content_hash: string } | null;
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
  favorite: boolean;
  notes: string;
  // The style pack (LoRA) applied when this take was generated, if any
  // (SPEC.md sec 4.4 "LoRA train / load") -- server.storage
  // _normalize_take_meta guarantees the key is present even on takes
  // written before the field existed. Provenance for reproducing a styled
  // generation: the UI resolves it to the pack's name via GET
  // /api/projects/{id}/loras (falling back to this stable id).
  lora_id: string | null;
}

export interface ProjectDetail {
  project: Project;
  plan: Plan;
  takes: Take[];
}

// Mirrors the meta dict worker.acestep_worker.train_lora (and
// worker.mock_worker.train_lora) return -- server.storage persists it
// verbatim as <lora_dir>/meta.json and GET /api/projects/{id}/loras lists
// those dicts as-is (SPEC.md sec 4.4 "Style pack | LoRA train / load").
export interface Lora {
  id: string;
  name: string;
  created_at: string;
  source_take_count: number;
  base_checkpoint: string;
  dit_profile: string;
  final_step: number | null;
  final_loss: number | null;
  status: string | null;
  error: string | null;
}

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";
export const TERMINAL_JOB_STATUSES: readonly string[] = ["done", "error", "cancelled"];

export interface Job {
  id: string;
  project_id: string;
  action: string;
  dit_profile: string;
  status: string;
  // SPEC.md sec 15.7/16.5. Optional: older rows and the train_lora path may
  // omit them.
  stage?: string | null;
  cancel_requested?: number;
  submission_id?: string | null;
  retry_of?: string | null;
  take_id: string | null;
  // Set for train_lora jobs once the worker allocates the pack (the finished
  // pack id on `done`; kept on `error` too so a failed training remains a
  // tracked pack entry). Null while the job is still queued/running and for
  // every non-train_lora job.
  lora_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

/** A job body as SPEC.md sec 8.1 defines it. Built by lib/jobBodies.ts. */
export type JobRequest = { action: string } & Record<string, unknown>;

/** An HTTP error response. `status` lets callers tell a conflict (409) or a
 *  rejected request (4xx) from a lost response, which surfaces as a plain
 *  network `TypeError` from fetch instead. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new ApiError(resp.status, `${resp.status} ${resp.statusText}: ${body}`);
  }
  // DELETE /api/projects/{id} responds 204 No Content -- resp.json() would
  // throw on the empty body, so every other (JSON-bodied) response takes
  // the normal path and this one alone returns undefined.
  if (resp.status === 204) {
    return undefined as T;
  }
  return resp.json() as Promise<T>;
}

async function uploadAudio(projectId: string, file: File): Promise<{ upload_path: string }> {
  // Multipart, not JSON -- can't go through request<T>(), which always
  // sends Content-Type: application/json.
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`/api/projects/${projectId}/uploads`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
  }
  return resp.json() as Promise<{ upload_path: string }>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  remoteGpu: () => request<RemoteGpuStatus>("/api/remote-gpu"),
  connectRemoteGpu: (base_url: string, secret: string) =>
    request<RemoteGpuStatus>("/api/remote-gpu/connection", {
      method: "PUT",
      body: JSON.stringify({ base_url, secret }),
    }),
  disconnectRemoteGpu: () =>
    request<RemoteGpuStatus>("/api/remote-gpu/connection", { method: "DELETE" }),
  startRemoteGpu: () => request<RemoteGpuStatus>("/api/remote-gpu/session", { method: "POST" }),
  stopRemoteGpu: () => request<RemoteGpuStatus>("/api/remote-gpu/session", { method: "DELETE" }),
  uploadAudio,
  listProjects: () => request<ProjectSummary[]>("/api/projects"),
  createProject: (title: string, query: string) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ title, query }),
    }),
  getProject: (id: string) => request<ProjectDetail>(`/api/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  patchProject: (id: string, patch: Partial<Pick<Project, "title" | "dit_profile" | "favorite">>) =>
    request<Project>(`/api/projects/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  // `keepalive` lets a caller fire this from a `pagehide`/`beforeunload`
  // handler and have the browser complete the request even as the page is
  // being torn down -- a plain fetch gets aborted mid-flight in that window.
  patchTake: (
    projectId: string,
    takeId: string,
    patch: { favorite?: boolean; notes?: string },
    opts?: { keepalive?: boolean },
  ) =>
    request<Take>(`/api/projects/${projectId}/takes/${takeId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
      ...(opts?.keepalive ? { keepalive: true } : {}),
    }),
  // expectedRevision makes the write conditional (SPEC.md sec 16.3): a
  // plan that moved on since this edit started is a 409, never a silent
  // overwrite. keepalive lets a pagehide handler finish the last save.
  savePlan: (
    id: string,
    plan: Plan,
    opts?: { expectedRevision?: number | null; keepalive?: boolean },
  ) =>
    request<Plan>(`/api/projects/${id}/plan`, {
      method: "PUT",
      body: JSON.stringify(
        opts?.expectedRevision != null
          ? { ...plan, expected_revision: opts.expectedRevision }
          : plan,
      ),
      ...(opts?.keepalive ? { keepalive: true } : {}),
    }),
  // One enqueue for every action. `submission_id` is the browser's
  // identity for this submission (SPEC.md sec 16.4): resending the same id
  // and body after a lost response returns the job the first send created.
  enqueue: (projectId: string, body: JobRequest, submissionId: string) =>
    request<Job>(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      body: JSON.stringify({ ...body, submission_id: submissionId }),
    }),
  cancelJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  retryJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}/retry`, { method: "POST" }),
  listLoras: (projectId: string) => request<Lora[]>(`/api/projects/${projectId}/loras`),
  setActiveTake: (id: string, takeId: string) =>
    request<Project>(`/api/projects/${id}/active_take`, {
      method: "POST",
      body: JSON.stringify({ take_id: takeId }),
    }),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  // GET /api/jobs with optional narrowing filters, applied server-side
  // before the limit. The UI uses { projectId, action: "train_lora",
  // active: true } on project load to recover a style-pack training that
  // survived a page refresh (training runs up to ~1 hour, SPEC.md sec 4.4)
  // -- active returns the complete queued/running worklist with no recency
  // truncation (the server ignores `limit` in that case), so an older
  // long-running training can never be pushed out by newer or finished
  // jobs. The recency `limit` applies only when `active` is not set.
  listJobs: (opts?: { projectId?: string; action?: string; active?: boolean; limit?: number }) => {
    const params = new URLSearchParams();
    if (opts?.projectId) params.set("project_id", opts.projectId);
    if (opts?.action) params.set("action", opts.action);
    if (opts?.active) params.set("active", "true");
    if (opts?.limit != null) params.set("limit", String(opts.limit));
    const qs = params.toString();
    return request<Job[]>(`/api/jobs${qs ? `?${qs}` : ""}`);
  },
  takeAudioUrl: (projectId: string, takeId: string) =>
    `/api/projects/${projectId}/takes/${takeId}/audio`,
  takeLrcUrl: (projectId: string, takeId: string) =>
    `/api/projects/${projectId}/takes/${takeId}/lrc`,
  exportUrl: (projectId: string) => `/api/projects/${projectId}/export`,
};
