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
import type { Job, Lora, Plan, ProjectDetail, ProjectSummary, Take } from "../api";

export const PROJECT_ID = "proj-1";

export interface RecordedRequest {
  method: string;
  url: string;
  body: unknown;
}

// Successive statuses GET /api/jobs/{id} returns for one enqueued job; the
// final entry repeats for any extra poll. Mirrors server/jobs.py's
// queued -> running -> done|error lifecycle.
export interface JobScript {
  statuses: string[];
  error?: string | null;
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
    favorite: false,
    notes: "",
    // Required by Take since the applied-style provenance UX; tests that
    // need a styled take override this.
    lora_id: null,
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

// ---------------------------------------------------------------------------
// The mock server itself.
// ---------------------------------------------------------------------------

export function createMockBardServer() {
  const requests: RecordedRequest[] = [];
  const state = {
    projects: [makeProjectSummary()],
    detail: makeProjectDetail(makeTakes(10)),
    loras: [] as Lora[],
  };

  const jobEntries = new Map<
    string,
    { projectId: string; action: string; script: JobScript; pollIndex: number }
  >();
  let nextJobScript: JobScript | null = null;
  let jobsPostFailure: { status: number; body: unknown } | null = null;
  let jobCounter = 0;
  let loraCounter = 0;

  function scriptNextJob(script: JobScript): void {
    nextJobScript = script;
  }

  // Makes the next POST /api/projects/{id}/jobs fail with the given HTTP
  // status/body (server-side validation errors, e.g. the 8-take floor).
  function failNextJobsPost(status: number, body: unknown): void {
    jobsPostFailure = { status, body };
  }

  function jobRow(
    id: string,
    projectId: string,
    action: string,
    status: string,
    script: JobScript,
  ): Job {
    return {
      id,
      project_id: projectId,
      action,
      dit_profile: action === "train_lora" ? "studio_ops" : "iterate",
      status,
      take_id: status === "done" && action !== "train_lora" ? `take-of-${id}` : null,
      // Required by Job since the training-recovery work; null matches
      // "queued/running or non-train_lora" (no pack allocated yet).
      lora_id: null,
      error: status === "error" ? (script.error ?? "job failed") : null,
      created_at: CREATED_AT,
      updated_at: CREATED_AT,
    };
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
    requests.push({ method, url, body });

    if (method === "GET" && url === "/api/health") {
      return Promise.resolve(
        jsonResponse({ ok: true, gpu: "RTX 4070 Ti SUPER", dit_loaded: "iterate" }),
      );
    }
    if (method === "GET" && url === "/api/projects") {
      return Promise.resolve(jsonResponse(state.projects));
    }

    let m = url.match(/^\/api\/projects\/([^/]+)$/);
    if (method === "GET" && m) {
      return Promise.resolve(jsonResponse(state.detail));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/plan$/);
    if (method === "PUT" && m) {
      // SPEC.md sec 8: PUT /plan replaces plan.json outright, so the mock
      // swaps in whatever body it got verbatim (sections included).
      state.detail = { ...state.detail, plan: (body ?? {}) as Plan };
      return Promise.resolve(jsonResponse(state.detail.plan));
    }

    m = url.match(/^\/api\/projects\/([^/]+)\/loras$/);
    if (method === "GET" && m) {
      return Promise.resolve(jsonResponse(state.loras));
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
      jobCounter += 1;
      const id = `job-${jobCounter}`;
      const script = nextJobScript ?? { statuses: ["done"] };
      nextJobScript = null;
      jobEntries.set(id, { projectId: m[1], action, script, pollIndex: 0 });
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

    m = url.match(/^\/api\/jobs\/([^/]+)$/);
    if (method === "GET" && m) {
      const entry = jobEntries.get(m[1]);
      if (!entry) {
        return Promise.resolve(jsonResponse({ detail: "no such job" }, 404));
      }
      const status =
        entry.script.statuses[
          Math.min(entry.pollIndex, entry.script.statuses.length - 1)
        ];
      entry.pollIndex += 1;
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

  return {
    state,
    requests,
    install,
    uninstall,
    scriptNextJob,
    failNextJobsPost,
    jobRequests,
  };
}

export type MockBardServer = ReturnType<typeof createMockBardServer>;
