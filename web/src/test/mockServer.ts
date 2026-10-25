// Mocked FastAPI backend for the frontend regression tests (SPEC.md sec 11
// keeps pytest as the default testCommand for the Python side; this is the
// web-side counterpart). The tests render the real <App/> and swap
// global.fetch for a router shaped after SPEC.md sec 8's HTTP contract, so
// nothing here needs FastAPI, CUDA, ACE-Step, credentials, or generated
// audio.
//
// Everything is deterministic by construction: fixed timestamps, no
// Math.random, and jobs complete on the first /api/jobs/{id} poll unless a
// test explicitly scripts a different sequence via scriptNextJob().
import { vi } from "vitest";
import type {
  Capabilities,
  Job,
  Lora,
  Plan,
  Project,
  ProjectDetail,
  ProjectSummary,
  Take,
  TrackedField,
} from "../api";

export const PROJECT_ID = "proj-1";

export interface RecordedRequest {
  method: string;
  url: string;
  body: unknown;
  // For job POSTs: the submission identity (SPEC.md sec 16.4), kept out of
  // `body` so request-body assertions stay about the request itself.
  submissionId?: string;
}

// Successive statuses GET /api/jobs/{id} returns for one enqueued job; the
// final entry repeats for any extra poll. Mirrors server/jobs.py's
// queued -> running -> done|error lifecycle.
export interface JobScript {
  statuses: string[];
  error?: string | null;
  // Optional stage per status entry (same index), for the job strip.
  stages?: (string | null)[];
}

interface MockResponseLike {
  ok: boolean;
  status: number;
  statusText: string;
  json(): Promise<unknown>;
  text(): Promise<string>;
}

function jsonResponse(data: unknown, status = 200): MockResponseLike {
  const ok = status >= 200 && status < 300;
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

const CREATED_AT = "2026-01-01T00:00:00+00:00";

// ---------------------------------------------------------------------------
// Fixture factories -- shapes mirror web/src/api.ts (which mirrors server/).
// ---------------------------------------------------------------------------

export function makeTake(id: string, index: number, overrides: Partial<Take> = {}): Take {
  return {
    id,
    parent_take_id: null,
    task_type: "text2music",
    dit_profile: "iterate",
    seed: 1000 + index,
    duration_sec: 100 + index,
    caption: `caption ${index}`,
    lyrics: "[Verse]\nla la la",
    bpm: 120,
    keyscale: "C Major",
    created_at: CREATED_AT,
    score: null,
    has_lrc: false,
    error: null,
    track_name: null,
    lora_id: null,
    favorite: false,
    notes: "",
    ...overrides,
  };
}

export function makeTakes(count: number): Take[] {
  return Array.from({ length: count }, (_, i) =>
    makeTake(`take-${String(i + 1).padStart(2, "0")}`, i + 1),
  );
}

export function makeLora(id: string, name: string, overrides: Partial<Lora> = {}): Lora {
  return {
    id,
    name,
    created_at: CREATED_AT,
    source_take_count: 8,
    base_checkpoint: "studio_ops",
    dit_profile: "studio_ops",
    final_step: 300,
    final_loss: 0.1234,
    status: "done",
    error: null,
    ...overrides,
  };
}

export function makePlan(): Plan {
  return {
    query: "",
    caption: "dark wizard folk",
    negative: [],
    lyrics: "[Verse]\nla la la",
    instrumental: false,
    vocal_language: "en",
    bpm: 120,
    keyscale: "C Major",
    timesignature: "4/4",
    duration_sec: 120,
    sections: [],
    // Matches server.storage.default_plan() (and keeps this literal a valid
    // Plan per the type added by the caption-rewrite job, SPEC.md sec 9.2).
    caption_rewrite: true,
  };
}

// What server.storage.default_plan() writes for a brand-new project: an
// empty brief that resolves its idea on the first Generate (SPEC.md sec
// 16.3). New-project fixtures must start from this, not from makePlan().
export function makeDefaultPlan(): Plan {
  return {
    query: "",
    caption: "",
    negative: [],
    lyrics: "",
    instrumental: false,
    vocal_language: "en",
    bpm: null,
    keyscale: null,
    timesignature: "4/4",
    duration_sec: 120,
    sections: [],
    caption_rewrite: false,
    creation_mode: "from_idea",
    field_origins: {
      caption: "default",
      lyrics: "default",
      vocal_language: "default",
      bpm: "default",
      keyscale: "default",
      timesignature: "default",
      duration_sec: "default",
    },
    revision: 0,
    brief: null,
  };
}

// Mirrors server/generation/presets.py closely enough for UI tests: two
// entries, one song and one instrumental.
export const MOCK_PRESETS = [
  {
    id: "lyric_song",
    version: 1,
    name: "Lyric song",
    purpose: "A song built around a sung lead and a clear verse and chorus.",
    caption: "song with a clear lead vocal",
    instrumental: false,
    fields: { timesignature: "4/4" },
    example: "a hopeful song",
  },
  {
    id: "combat_cue",
    version: 1,
    name: "Combat cue",
    purpose: "Driving, tense music for a fight or a chase.",
    caption: "intense combat music",
    instrumental: true,
    fields: { bpm: 150, duration_sec: 90 },
    example: "a boss fight",
  },
];

export function makeCapabilities(overrides: Partial<Capabilities> = {}): Capabilities {
  const profile = (id: string, label: string): Capabilities["profiles"][number] => ({
    id,
    label,
    summary: `${label} summary`,
    checkpoint: `acestep-${id}`,
    inference_steps: 8,
    guidance_scale: 1,
    cpu_offload: id === "quality",
    files: "present",
    validated: id === "iterate",
    loaded: id === "iterate",
    supported: true,
    reason: null,
  });
  return {
    worker: {
      state: "ready",
      reason: "worker: 'iterate' DiT + LM currently loaded",
      backend: "mock",
      dit_loaded: "iterate",
      queue: { queued: 0, running: 0 },
    },
    profiles: [
      profile("iterate", "Iterate"),
      profile("polish", "Polish"),
      profile("quality", "Quality"),
    ],
    lm: { model: "acestep-5Hz-lm-1.7B", selectable: false },
    disk: { free_bytes: 100 * 1024 ** 3, low: false, blocked: false },
    presets: MOCK_PRESETS,
    ...overrides,
  };
}

export function makeProjectSummary(overrides: Partial<ProjectSummary> = {}): ProjectSummary {
  return {
    id: PROJECT_ID,
    title: "Test Song",
    updated_at: CREATED_AT,
    active_take_id: null,
    favorite: false,
    ...overrides,
  };
}

export function makeProjectDetail(takes: Take[]): ProjectDetail {
  return {
    project: {
      id: PROJECT_ID,
      title: "Test Song",
      created_at: CREATED_AT,
      updated_at: CREATED_AT,
      dit_profile: "iterate",
      lm_model: "acestep-5Hz-lm-1.7B",
      active_take_id: takes[0]?.id ?? null,
      favorite: false,
    },
    plan: makePlan(),
    takes,
  };
}

const TRACKED: TrackedField[] = [
  "caption",
  "lyrics",
  "vocal_language",
  "bpm",
  "keyscale",
  "timesignature",
  "duration_sec",
];

// The server's write rule (server/generation/brief.apply_write_semantics):
// a tracked field whose value changed becomes `user`, or `default` when
// cleared, unless the body names another origin; a changed idea switches
// to `from_idea` unless the body sets the mode.
function deriveWrite(current: Plan, next: Plan): Plan {
  const origins = { ...(current.field_origins ?? {}) };
  for (const field of TRACKED) {
    const claimed = next.field_origins?.[field];
    if (claimed && claimed !== current.field_origins?.[field]) {
      origins[field] = claimed;
    } else if (next[field] !== current[field]) {
      const value = next[field];
      origins[field] = value === null || String(value).trim() === "" ? "default" : "user";
    }
  }
  let mode = current.creation_mode;
  if (next.creation_mode && next.creation_mode !== current.creation_mode) {
    mode = next.creation_mode;
  } else if (next.query !== current.query) {
    mode = "from_idea";
  }
  return { ...next, field_origins: origins, creation_mode: mode, brief: current.brief ?? null };
}

// ---------------------------------------------------------------------------
// The mock server itself.
// ---------------------------------------------------------------------------

export function createMockLyreServer() {
  const requests: RecordedRequest[] = [];
  const state = {
    projects: [makeProjectSummary()],
    detail: makeProjectDetail(makeTakes(10)),
    // Projects created via POST /api/projects (SPEC.md sec 8), keyed by id.
    // The single `detail` above stays the one and only pre-seeded fixture
    // project (PROJECT_ID); everything created at runtime lives here so GET
    // /api/projects/{id} has something to return for it afterward.
    createdDetails: new Map<string, ProjectDetail>(),
    loras: [] as Lora[],
    capabilities: makeCapabilities(),
  };

  const jobEntries = new Map<
    string,
    {
      projectId: string;
      action: string;
      script: JobScript;
      pollIndex: number;
      takeAdded: boolean;
      submissionId: string | null;
      bodyKey: string;
      cancelRequested: boolean;
      retryOf: string | null;
    }
  >();
  let nextJobScript: JobScript | null = null;
  // Makes the next N job POSTs reach the server (the job is created) but
  // lose their response, the way a dropped connection does.
  let lostJobsPostResponses = 0;
  // Makes the next N GET /api/jobs/{id} polls fail with a network error.
  let failedJobPolls = 0;
  // Makes the next N PUT .../plan fail with this HTTP status.
  let planPutFailures: { count: number; status: number } | null = null;
  let jobsPostFailure: { status: number; body: unknown } | null = null;
  let uploadFailure: { status: number; body: unknown } | null = null;
  let jobCounter = 0;
  let loraCounter = 0;
  let uploadCounter = 0;
  let projectCounter = 0;

  function scriptNextJob(script: JobScript): void {
    nextJobScript = script;
  }

  function loseNextJobsPostResponses(count = 1): void {
    lostJobsPostResponses = count;
  }

  function failNextJobPolls(count: number): void {
    failedJobPolls = count;
  }

  function failNextPlanPuts(count: number, status = 500): void {
    planPutFailures = { count, status };
  }

  // Lets a test move a job along as another client or the worker would.
  function setJobScript(id: string, script: JobScript): void {
    const entry = jobEntries.get(id);
    if (entry) {
      entry.script = script;
      entry.pollIndex = 0;
    }
  }

  // Makes the next POST /api/projects/{id}/jobs fail with the given HTTP
  // status/body (server-side validation errors, e.g. the 8-take floor).
  function failNextJobsPost(status: number, body: unknown): void {
    jobsPostFailure = { status, body };
  }

  // Makes the next POST /api/projects/{id}/uploads fail with the given HTTP
  // status/body (e.g. an unsupported file type/size the server rejects).
  function failNextUpload(status: number, body: unknown): void {
    uploadFailure = { status, body };
  }

  function jobRow(
    id: string,
    projectId: string,
    action: string,
    status: string,
    script: JobScript,
  ): Job {
    const entry = jobEntries.get(id);
    const index = entry ? Math.min(entry.pollIndex, script.statuses.length - 1) : 0;
    return {
      stage: script.stages?.[index] ?? null,
      cancel_requested: entry?.cancelRequested ? 1 : 0,
      submission_id: entry?.submissionId ?? null,
      retry_of: entry?.retryOf ?? null,
      id,
      project_id: projectId,
      action,
      dit_profile: action === "train_lora" ? "studio_ops" : "iterate",
      status,
      take_id: status === "done" && action !== "train_lora" ? `take-of-${id}` : null,
      // The UI never reads a job row's lora_id (take provenance comes from
      // take.lora_id); null keeps the row shaped like server.jobs._row_to_dict.
      lora_id: null,
      error: status === "error" ? (script.error ?? "job failed") : null,
      created_at: CREATED_AT,
      updated_at: CREATED_AT,
    };
  }

  // Current status of an entry without advancing its poll script -- used by
  // the GET /api/jobs list route below, which (unlike GET /api/jobs/{id})
  // must not itself drive queued -> running -> done progression just by
  // being listed.
  function entryStatus(entry: { script: JobScript; pollIndex: number }): string {
    return entry.script.statuses[Math.min(entry.pollIndex, entry.script.statuses.length - 1)];
  }

  // Directly inserts a job row into the queue the way a job created by
  // another session/tab (or before this test's render) would already exist
  // -- for the LoRA training-recovery tests, which need a queued/running (or
  // finished) train_lora job to already be in the queue *before* the app
  // opens the project, so GET /api/jobs can recover it on project load.
  function seedJob(
    id: string,
    projectId: string,
    action: string,
    status: string,
    error: string | null = null,
  ): void {
    jobEntries.set(id, {
      projectId,
      action,
      script: { statuses: [status], error },
      pollIndex: 0,
      takeAdded: status === "done",
      submissionId: null,
      bodyKey: "",
      cancelRequested: false,
      retryOf: null,
    });
  }

  // Mirrors server.jobs writing the new take before flipping the job row to
  // done -- the first response to report "done" is what makes the take show
  // up in the next GET /api/projects/{id} (App refreshes when it sees it).
  function materializeTake(
    id: string,
    entry: { action: string; takeAdded: boolean; projectId: string },
  ): void {
    if (entry.action === "train_lora" || entry.takeAdded) return;
    entry.takeAdded = true;
    const take = makeTake(`take-of-${id}`, state.detail.takes.length + 1, {
      task_type: entry.action,
    });
    state.detail = { ...state.detail, takes: [...state.detail.takes, take] };
  }

  function handle(input: string | URL | Request, init?: RequestInit): Promise<MockResponseLike> {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    let body: unknown = null;
    if (init?.body != null && typeof init.body !== "object") {
      try {
        body = JSON.parse(String(init.body));
      } catch {
        body = String(init.body);
      }
    }
    const recorded: RecordedRequest = { method, url, body };
    if (method === "POST" && url.endsWith("/jobs") && body && typeof body === "object") {
      const { submission_id: submissionId, ...rest } = body as Record<string, unknown>;
      recorded.body = rest;
      recorded.submissionId = submissionId as string | undefined;
    }
    requests.push(recorded);

    if (method === "GET" && url === "/api/health") {
      return Promise.resolve(
        jsonResponse({ ok: true, gpu: "RTX 4070 Ti SUPER", dit_loaded: "iterate" }),
      );
    }
    if (method === "GET" && url === "/api/capabilities") {
      return Promise.resolve(jsonResponse(state.capabilities));
    }
    if (method === "GET" && url === "/api/remote-gpu") {
      return Promise.resolve(
        jsonResponse({
          provisioner: "manual",
          label: "Self-hosted",
          can_provision: false,
          stop_on_exit: true,
          quote: null,
          connection: null,
          session: null,
          host: null,
        }),
      );
    }
    if (method === "GET" && url === "/api/projects") {
      return Promise.resolve(jsonResponse(state.projects));
    }
    if (method === "POST" && url === "/api/projects") {
      const payload = (body ?? {}) as { title?: string; query?: string };
      projectCounter += 1;
      const id = `proj-created-${projectCounter}`;
      const title = payload.title?.trim() || "Untitled";
      const project: Project = {
        id,
        title,
        created_at: CREATED_AT,
        updated_at: CREATED_AT,
        dit_profile: "iterate",
        lm_model: "acestep-5Hz-lm-1.7B",
        active_take_id: null,
        favorite: false,
      };
      state.projects = [...state.projects, makeProjectSummary({ id, title })];
      const plan = makeDefaultPlan();
      if (payload.query) plan.query = payload.query;
      state.createdDetails.set(id, { project, plan, takes: [] });
      return Promise.resolve(jsonResponse(project));
    }

    let m = url.match(/^\/api\/projects\/([^/]+)$/);
    if (method === "GET" && m) {
      if (m[1] === PROJECT_ID) {
        return Promise.resolve(jsonResponse(state.detail));
      }
      const created = state.createdDetails.get(m[1]);
      if (created) {
        return Promise.resolve(jsonResponse(created));
      }
      return Promise.resolve(jsonResponse({ detail: `no such project ${m[1]}` }, 404));
    }
    if (method === "PATCH" && m) {
      const patch = (body ?? {}) as Partial<Pick<Project, "title" | "dit_profile" | "favorite">>;
      if (patch.title !== undefined) patch.title = patch.title.trim() || "Untitled";
      state.detail = { ...state.detail, project: { ...state.detail.project, ...patch } };
      state.projects = state.projects.map((p) => (p.id === m![1] ? { ...p, ...patch } : p));
      return Promise.resolve(jsonResponse(state.detail.project));
    }
    if (method === "DELETE" && m) {
      state.projects = state.projects.filter((p) => p.id !== m![1]);
      return Promise.resolve({
        ok: true,
        status: 204,
        statusText: "No Content",
        json: async () => undefined,
        text: async () => "",
      });
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/takes\/([^/]+)$/);
    if (method === "PATCH" && m) {
      const patch = (body ?? {}) as Partial<Pick<Take, "favorite" | "notes">>;
      let updated: Take | null = null;
      state.detail = {
        ...state.detail,
        takes: state.detail.takes.map((t) => {
          if (t.id !== m![2]) return t;
          updated = { ...t, ...patch };
          return updated;
        }),
      };
      if (!updated) {
        return Promise.resolve(jsonResponse({ detail: `no such take ${m[2]}` }, 404));
      }
      return Promise.resolve(jsonResponse(updated));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/active_take$/);
    if (method === "POST" && m) {
      const payload = (body ?? {}) as { take_id: string };
      state.detail = {
        ...state.detail,
        project: { ...state.detail.project, active_take_id: payload.take_id },
      };
      state.projects = state.projects.map((p) =>
        p.id === m![1] ? { ...p, active_take_id: payload.take_id } : p,
      );
      return Promise.resolve(jsonResponse(state.detail.project));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/plan$/);
    if (method === "PUT" && m) {
      if (planPutFailures && planPutFailures.count > 0) {
        planPutFailures.count -= 1;
        return Promise.resolve(
          jsonResponse({ detail: "plan save failed" }, planPutFailures.status),
        );
      }
      // SPEC.md sec 8: PUT /plan replaces plan.json outright, so the mock
      // swaps in whatever body it got verbatim (sections included), and,
      // like the server, bumps the revision and honors expected_revision
      // (sec 16.3).
      const { expected_revision: expected, ...plan } = (body ?? {}) as Plan & {
        expected_revision?: number;
      };
      const current = state.detail.plan.revision ?? 0;
      if (expected != null && expected !== current) {
        return Promise.resolve(jsonResponse({ detail: "plan revision changed" }, 409));
      }
      state.detail = {
        ...state.detail,
        plan: { ...deriveWrite(state.detail.plan, plan as Plan), revision: current + 1 },
      };
      return Promise.resolve(jsonResponse(state.detail.plan));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/preset$/);
    if (method === "POST" && m) {
      // server/generation/presets.apply_preset, reduced: user fields and
      // lyrics are kept; changed fields become origin "preset".
      const payload = (body ?? {}) as { preset_id: string; apply: boolean };
      const preset = MOCK_PRESETS.find((p) => p.id === payload.preset_id);
      if (!preset) return Promise.resolve(jsonResponse({ detail: "preset not found" }, 404));
      const current = state.detail.plan;
      const origins = { ...(current.field_origins ?? {}) };
      const plan = { ...current } as Plan & Record<string, unknown>;
      const changes: { field: string; from: unknown; to: unknown }[] = [];
      const warnings: string[] = [];
      const set = (field: string, to: unknown, tracked = true) => {
        if (plan[field] === to) return;
        changes.push({ field, from: plan[field], to });
        plan[field] = to;
        if (tracked) origins[field as TrackedField] = "preset";
      };
      if (origins.caption === "user" && current.caption.trim()) {
        warnings.push("kept your caption; the preset's caption was not applied");
      } else {
        set("caption", preset.caption);
      }
      for (const [field, value] of Object.entries(preset.fields)) {
        if (origins[field as TrackedField] === "user") warnings.push(`kept your ${field}`);
        else set(field, value);
      }
      const hasWords =
        current.lyrics.trim() !== "" && current.lyrics.trim().toLowerCase() !== "[instrumental]";
      if (preset.instrumental && hasWords) {
        warnings.push(
          "the preset suggests an instrumental, but the plan has lyrics, so it stays a song",
        );
      } else {
        set("instrumental", preset.instrumental, false);
      }
      plan.field_origins = origins;
      plan.preset = { id: preset.id, version: preset.version };
      if (payload.apply) {
        state.detail = {
          ...state.detail,
          plan: { ...plan, revision: (current.revision ?? 0) + 1 },
        };
      }
      return Promise.resolve(
        jsonResponse({
          plan: payload.apply ? state.detail.plan : plan,
          changes,
          warnings,
          preset: { id: preset.id, version: preset.version },
          applied: payload.apply,
        }),
      );
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/loras$/);
    if (method === "GET" && m) {
      return Promise.resolve(jsonResponse(state.loras));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/uploads$/);
    if (method === "POST" && m) {
      if (uploadFailure) {
        const failure = uploadFailure;
        uploadFailure = null;
        return Promise.resolve(jsonResponse(failure.body, failure.status));
      }
      uploadCounter += 1;
      return Promise.resolve(jsonResponse({ upload_path: `uploads/upload-${uploadCounter}.wav` }));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/jobs$/);
    if (method === "POST" && m) {
      if (jobsPostFailure) {
        const failure = jobsPostFailure;
        jobsPostFailure = null;
        return Promise.resolve(jsonResponse(failure.body, failure.status));
      }
      const payload = (body ?? {}) as Record<string, unknown>;
      const action = String(payload.action ?? "");
      const submissionId = (payload.submission_id as string | undefined) ?? null;
      const { submission_id: _ignored, ...rest } = payload;
      void _ignored;
      const bodyKey = JSON.stringify(rest);
      // SPEC.md sec 16.4: the same submission id and body return the job
      // the first send created; a different body is a 409.
      for (const [existingId, entry] of jobEntries) {
        if (submissionId && entry.submissionId === submissionId && entry.projectId === m[1]) {
          if (entry.bodyKey !== bodyKey) {
            return Promise.resolve(jsonResponse({ detail: "submission_id conflict" }, 409));
          }
          const status = entryStatus(entry);
          if (status === "done") materializeTake(existingId, entry);
          return Promise.resolve(
            jsonResponse(jobRow(existingId, m[1], action, status, entry.script)),
          );
        }
      }
      jobCounter += 1;
      const id = `job-${jobCounter}`;
      const script = nextJobScript ?? { statuses: ["done"] };
      nextJobScript = null;
      jobEntries.set(id, {
        projectId: m[1],
        action,
        script,
        pollIndex: 0,
        takeAdded: false,
        submissionId,
        bodyKey,
        cancelRequested: false,
        retryOf: null,
      });
      if (lostJobsPostResponses > 0) {
        lostJobsPostResponses -= 1;
        return Promise.reject(new TypeError("Failed to fetch"));
      }
      // A successful train_lora makes the pack listable right away, like
      // server.jobs._run_train_lora_job writing meta.json before the job
      // row flips to done (the UI re-fetches this list after the poll).
      if (action === "train_lora" && script.statuses[script.statuses.length - 1] === "done") {
        loraCounter += 1;
        state.loras = [
          ...state.loras,
          makeLora(`lora-trained-${loraCounter}`, String(payload.name ?? "Untitled LoRA"), {
            source_take_count: Array.isArray(payload.source_take_ids)
              ? payload.source_take_ids.length
              : 0,
          }),
        ];
      }
      return Promise.resolve(jsonResponse(jobRow(id, m[1], action, "queued", script)));
    }

    // GET /api/jobs?project_id=&action=&active=&limit= -- SPEC.md sec 8,
    // added for LoRA training recovery (App.tsx refreshTrainingJobs). Must
    // be checked before the /api/jobs/{id} regex below since both start
    // with "/api/jobs", but this route only ever appears as exactly
    // "/api/jobs" or with a "?" query string, never a "/" path segment.
    if (method === "GET" && (url === "/api/jobs" || url.startsWith("/api/jobs?"))) {
      const qs = url.includes("?") ? url.slice(url.indexOf("?") + 1) : "";
      const params = new URLSearchParams(qs);
      const projectId = params.get("project_id");
      const action = params.get("action");
      const active = params.get("active") === "true";
      const limit = params.has("limit") ? Number(params.get("limit")) : 20;

      // jobEntries preserves insertion (creation) order; newest-first
      // mirrors server/jobs.py's "ORDER BY created_at DESC".
      let rows = Array.from(jobEntries.entries()).reverse();
      if (projectId) rows = rows.filter(([, e]) => e.projectId === projectId);
      if (action) rows = rows.filter(([, e]) => e.action === action);
      if (active) {
        rows = rows.filter(([, e]) => {
          const status = entryStatus(e);
          return status === "queued" || status === "running";
        });
      } else {
        rows = rows.slice(0, limit);
      }
      return Promise.resolve(
        jsonResponse(
          rows.map(([id, e]) => jobRow(id, e.projectId, e.action, entryStatus(e), e.script)),
        ),
      );
    }

    m = url.match(/^\/api\/jobs\/([^/]+)\/cancel$/);
    if (method === "POST" && m) {
      const entry = jobEntries.get(m[1]);
      if (!entry) return Promise.resolve(jsonResponse({ detail: "no such job" }, 404));
      const status = entryStatus(entry);
      if (status === "queued") {
        entry.script = { statuses: ["cancelled"] };
        entry.pollIndex = 0;
      } else if (status === "running") {
        entry.cancelRequested = true;
      }
      return Promise.resolve(
        jsonResponse(jobRow(m[1], entry.projectId, entry.action, entryStatus(entry), entry.script)),
      );
    }

    m = url.match(/^\/api\/jobs\/([^/]+)\/retry$/);
    if (method === "POST" && m) {
      const original = jobEntries.get(m[1]);
      if (!original) return Promise.resolve(jsonResponse({ detail: "no such job" }, 404));
      jobCounter += 1;
      const id = `job-${jobCounter}`;
      const script = nextJobScript ?? { statuses: ["done"] };
      nextJobScript = null;
      jobEntries.set(id, {
        ...original,
        script,
        pollIndex: 0,
        takeAdded: false,
        submissionId: null,
        cancelRequested: false,
        retryOf: m[1],
      });
      return Promise.resolve(
        jsonResponse(jobRow(id, original.projectId, original.action, "queued", script)),
      );
    }

    m = url.match(/^\/api\/jobs\/([^/]+)$/);
    if (method === "GET" && m) {
      if (failedJobPolls > 0) {
        failedJobPolls -= 1;
        return Promise.reject(new TypeError("Failed to fetch"));
      }
      const entry = jobEntries.get(m[1]);
      if (!entry) {
        return Promise.resolve(jsonResponse({ detail: "no such job" }, 404));
      }
      const status = entryStatus(entry);
      entry.pollIndex += 1;
      if (status === "done") materializeTake(m[1], entry);
      return Promise.resolve(
        jsonResponse(jobRow(m[1], entry.projectId, entry.action, status, entry.script)),
      );
    }

    return Promise.resolve(jsonResponse({ detail: `no mock route for ${method} ${url}` }, 404));
  }

  function install(): void {
    vi.stubGlobal("fetch", (input: string | URL | Request, init?: RequestInit) =>
      handle(input, init),
    );
  }

  function uninstall(): void {
    vi.unstubAllGlobals();
  }

  // POSTs to .../jobs only, in request order -- what the LoRA tests assert
  // enqueue bodies against.
  function jobRequests(action?: string): RecordedRequest[] {
    return requests.filter((r) => {
      if (r.method !== "POST" || !r.url.endsWith("/jobs")) return false;
      if (!action) return true;
      return (r.body as Record<string, unknown> | null)?.action === action;
    });
  }

  function jobSubmissionIds(): (string | undefined)[] {
    return jobRequests().map((r) => r.submissionId);
  }

  return {
    state,
    requests,
    install,
    uninstall,
    scriptNextJob,
    failNextJobsPost,
    failNextUpload,
    jobRequests,
    jobSubmissionIds,
    seedJob,
    setJobScript,
    loseNextJobsPostResponses,
    failNextJobPolls,
    failNextPlanPuts,
  };
}

export type MockLyreServer = ReturnType<typeof createMockLyreServer>;
