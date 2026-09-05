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
  sections: Section[];
  // SPEC.md sec 9.2: Custom-mode checkbox -- whether ACE-Step's LM
  // ("thinking") is allowed to rewrite the user's caption. Simple mode
  // ignores this and always runs with thinking=true.
  caption_rewrite: boolean;
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

export interface Job {
  id: string;
  project_id: string;
  action: string;
  dit_profile: string;
  status: string;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${body}`);
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
  savePlan: (id: string, plan: Plan) =>
    request<Plan>(`/api/projects/${id}/plan`, {
      method: "PUT",
      body: JSON.stringify(plan),
    }),
  // dit_profile is intentionally omitted: the server falls back to the
  // project's own persisted dit_profile (PATCH /api/projects/{id}) when the
  // job body doesn't provide one. Hardcoding "iterate" here would silently
  // override that project-level choice on every generate request.
  //
  // loraId is optional (SPEC.md sec 4.4 "LoRA train / load" -- the load
  // half): when set, server.jobs._resolve_dit_profile coerces the omitted
  // dit_profile to "studio_ops" for us, since a trained LoRA's weights are
  // only valid against that base checkpoint -- callers must not also pass an
  // explicit dit_profile here or the server rejects the conflict.
  //
  // seed (SPEC.md sec 7.3): -1 means the worker picks a seed and records
  // the actual one used in the take meta; any other integer is a fixed seed
  // the worker must honor verbatim.
  generate: (id: string, loraId?: string | null, seed: number = -1) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "generate",
        seed,
        ...(loraId ? { lora_id: loraId } : {}),
      }),
    }),
  // Exactly one of source_take_id/upload_path is ever sent, matching
  // server.jobs._resolve_source_audio's "source_take_id or upload_path"
  // contract (SPEC.md sec 8.1) -- callers pass whichever source is active.
  // seed: see generate() above (-1 = worker picks and records).
  cover: (
    id: string,
    source: { takeId: string } | { uploadPath: string },
    strength: number,
    loraId?: string | null,
    seed: number = -1,
  ) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "cover",
        ...("takeId" in source
          ? { source_take_id: source.takeId }
          : { upload_path: source.uploadPath }),
        audio_cover_strength: strength,
        seed,
        ...(loraId ? { lora_id: loraId } : {}),
      }),
    }),
  // seed: see generate() above (-1 = worker picks and records).
  repaint: (
    id: string,
    source: { takeId: string } | { uploadPath: string },
    start: number,
    end: number,
    loraId?: string | null,
    seed: number = -1,
  ) =>
    request<Job>(`/api/projects/${id}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "repaint",
        ...("takeId" in source
          ? { source_take_id: source.takeId }
          : { upload_path: source.uploadPath }),
        repainting_start: start,
        repainting_end: end,
        seed,
        ...(loraId ? { lora_id: loraId } : {}),
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
  listLoras: (projectId: string) => request<Lora[]>(`/api/projects/${projectId}/loras`),
  // source_take_ids must contain 8+ distinct take ids -- server.jobs
  // enforces MIN_LORA_SOURCE_TAKES and rejects with a clear JobError
  // otherwise (SPEC.md sec 4.4 "8+ songs").
  trainLora: (projectId: string, sourceTakeIds: string[], name: string) =>
    request<Job>(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      body: JSON.stringify({
        action: "train_lora",
        source_take_ids: sourceTakeIds,
        name,
        seed: -1,
      }),
    }),
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
