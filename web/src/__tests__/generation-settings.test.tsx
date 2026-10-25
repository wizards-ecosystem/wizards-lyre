// Browser regression tests for the shipped seed control and project-level
// DiT-profile picker (SPEC.md sec 7.3 "store the actual seed used" / sec 4.1
// project default DiT checkpoint). Covers:
//
// - the seed field: empty and explicit -1 both send seed -1, a fixed integer
//   is forwarded unchanged through Generate/Cover/Repaint, and text that
//   can't parse as an integer never leaks an invalid value into a job (it
//   falls back to -1, same as empty);
// - the DiT picker: iterate/polish/quality each PATCH the project's
//   dit_profile, update which option reads as selected, survive a project
//   reload, and roll back with a visible error when the PATCH fails;
// - studio_ops is never offered in this picker -- only selecting a style
//   pack (LoRA) forces that base-model swap, gated behind its own confirm.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts -- no
// FastAPI, CUDA, ACE-Step, or real audio. New file; test-local wavesurfer
// stub and a test-local fetch wrapper for the PATCH-failure case only (same
// technique App.plan.test.tsx uses for a rejected PUT /plan), so nothing in
// mockServer.ts itself changes.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import {
  createMockLyreServer,
  makeCapabilities,
  makeDefaultPlan,
  makeLora,
  makeTakes,
  PROJECT_ID,
} from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

interface FakeRegion {
  id: string;
  start: number;
  end: number;
  remove: () => void;
}

// jsdom has no canvas/layout for wavesurfer.js, so the waveform stack is
// stubbed out -- same shape as App.lora.test.tsx / App.studio-loop.test.tsx.
const regionHooks = vi.hoisted(() => ({
  handlers: {} as Record<string, (region: FakeRegion) => void>,
}));

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => {}, destroy: () => {} }),
  },
}));

vi.mock("wavesurfer.js/plugins/regions", () => ({
  default: {
    create: () => ({
      on: (event: string, cb: (region: FakeRegion) => void) => {
        regionHooks.handlers[event] = cb;
      },
      getRegions: () => [] as FakeRegion[],
      enableDragSelection: () => {},
      clearRegions: () => {},
    }),
  },
}));

const GOOD_LORA = makeLora("lora-good", "good-style");

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function jobsPost(action?: string) {
  if (!app) throw new Error("app not rendered");
  return app.server.jobRequests(action);
}

// The seed <input> has no id/aria-label of its own, but its wrapping
// <label className="seed-input">Seed<input/></label> gives it an implicit
// accessible name via testing-library's label association.
function seedField(): HTMLInputElement {
  return screen.getByLabelText("Seed") as HTMLInputElement;
}

function loraSelect(): HTMLSelectElement {
  return screen.getByRole("combobox", { name: /Style pack/ }) as HTMLSelectElement;
}

function ditGroup(): HTMLElement {
  return screen.getByRole("group", { name: "DiT profile" });
}

function ditButton(profile: string): HTMLButtonElement {
  return within(ditGroup()).getByRole("button", { name: profile }) as HTMLButtonElement;
}

function isSelected(button: HTMLButtonElement): boolean {
  return button.className.split(/\s+/).includes("selected");
}

function takesPane(): HTMLElement {
  const pane = screen.getByRole("heading", { name: "Takes" }).closest("section");
  if (!pane) throw new Error("takes pane not found");
  return pane as HTMLElement;
}

function takeRows(): HTMLElement[] {
  return within(takesPane()).getAllByRole("listitem");
}

async function dragRegion(start: number, end: number): Promise<void> {
  await waitFor(() => expect(regionHooks.handlers["region-created"]).toBeDefined());
  await act(async () => {
    regionHooks.handlers["region-created"]({ id: "region-1", start, end, remove: () => {} });
  });
}

function projectPatchRequests() {
  if (!app) throw new Error("app not rendered");
  return app.server.requests.filter(
    (r) => r.method === "PATCH" && r.url === `/api/projects/${PROJECT_ID}`,
  );
}

describe("Seed control (SPEC.md sec 7.3)", () => {
  it("sends seed -1 to Generate when the seed field is left empty", async () => {
    app = await renderOpenedProject();
    expect(seedField().value).toBe("");

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({ action: "generate", seed: -1 });
  });

  it("sends seed -1 to Generate when the field explicitly holds -1", async () => {
    app = await renderOpenedProject();
    fireEvent.change(seedField(), { target: { value: "-1" } });
    expect(seedField().value).toBe("-1");

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({ action: "generate", seed: -1 });
  });

  it("forwards a fixed integer seed unchanged through Generate, Cover, and Repaint", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.change(seedField(), { target: { value: "42" } });

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({ action: "generate", seed: 42 });
    // Wait for the job to finish (button text reverts) before the next
    // action -- Generate/Cover/Repaint all gate on the same `busy` flag.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Generate" }).textContent?.trim()).toBe("Generate"),
    );

    fireEvent.click(takeRows()[0]);
    fireEvent.click(screen.getByRole("button", { name: "Cover" }));
    await waitFor(() => expect(jobsPost("cover")).toHaveLength(1));
    expect(jobsPost("cover")[0].body).toEqual({
      action: "cover",
      source_take_id: "take-01",
      audio_cover_strength: 0.7,
      seed: 42,
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Cover" }).textContent).toBe("Cover"),
    );

    await dragRegion(1, 3);
    fireEvent.click(screen.getByRole("button", { name: "Repaint" }));
    await waitFor(() => expect(jobsPost("repaint")).toHaveLength(1));
    expect(jobsPost("repaint")[0].body).toEqual({
      action: "repaint",
      // The newly completed Cover take is automatically selected as the
      // next operation's source.
      source_take_id: "take-of-job-2",
      repainting_start: 1,
      repainting_end: 3,
      seed: 42,
    });

    // The fixed seed itself was never cleared by any of the three jobs.
    expect(seedField().value).toBe("42");
  });

  it("falls back to seed -1 for invalid seed text -- it can never reach a job as-is", async () => {
    app = await renderOpenedProject();

    // A number input can't hold non-numeric text at all: the DOM sanitizes
    // it back to "" before React even sees a value (verified below), so
    // garbage text can't leak into a job's seed field either directly or
    // through parseSeed's own non-integer fallback.
    fireEvent.change(seedField(), { target: { value: "42" } });
    expect(seedField().value).toBe("42");
    fireEvent.change(seedField(), { target: { value: "not-a-seed" } });
    expect(seedField().value).toBe("");

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({ action: "generate", seed: -1 });
  });
});

describe("DiT profile picker (SPEC.md sec 4.1)", () => {
  it("starts with the project's persisted dit_profile selected", async () => {
    app = await renderOpenedProject();
    expect(isSelected(ditButton("iterate"))).toBe(true);
    expect(isSelected(ditButton("polish"))).toBe(false);
    expect(isSelected(ditButton("quality"))).toBe(false);
  });

  it("PATCHes the project's dit_profile for iterate/polish/quality and updates the selected option", async () => {
    app = await renderOpenedProject();

    for (const profile of ["polish", "quality", "iterate"] as const) {
      fireEvent.click(ditButton(profile));
      await waitFor(() => expect(isSelected(ditButton(profile))).toBe(true));
      for (const other of ["iterate", "polish", "quality"] as const) {
        if (other !== profile) expect(isSelected(ditButton(other))).toBe(false);
      }
    }

    expect(projectPatchRequests().map((r) => r.body)).toEqual([
      { dit_profile: "polish" },
      { dit_profile: "quality" },
      { dit_profile: "iterate" },
    ]);
  });

  it("survives a project reload", async () => {
    const server = createMockLyreServer();
    server.install();
    let rendered = render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
    await screen.findByRole("group", { name: "DiT profile" });

    fireEvent.click(ditButton("quality"));
    await waitFor(() => expect(isSelected(ditButton("quality"))).toBe(true));
    expect(server.state.detail.project.dit_profile).toBe("quality");

    // Unmount (dropping all in-memory React state) and re-render <App/>
    // against the same mock server, so the picker can only be showing what
    // a fresh GET /api/projects/{id} actually returned.
    rendered.unmount();
    rendered = render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
    await screen.findByRole("group", { name: "DiT profile" });

    expect(isSelected(ditButton("quality"))).toBe(true);
    expect(isSelected(ditButton("iterate"))).toBe(false);

    rendered.unmount();
    server.uninstall();
  });

  it("rolls back the dit_profile selection and shows a visible error when the PATCH fails", async () => {
    app = await renderOpenedProject();
    expect(isSelected(ditButton("iterate"))).toBe(true);

    // Make the *next* PATCH /api/projects/{id} fail, without touching
    // mockServer.ts's shared PATCH handler used by every other test -- same
    // technique App.plan.test.tsx uses for a rejected PUT /plan.
    const previousFetch = globalThis.fetch;
    let failNext = true;
    vi.stubGlobal("fetch", (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (failNext && method === "PATCH" && url === `/api/projects/${PROJECT_ID}`) {
        failNext = false;
        const errorBody = { detail: "worker busy loading another dit_profile" };
        return Promise.resolve({
          ok: false,
          status: 409,
          statusText: "Conflict",
          json: async () => errorBody,
          text: async () => JSON.stringify(errorBody),
        } as Response);
      }
      return previousFetch(input, init);
    });

    fireEvent.click(ditButton("quality"));

    await screen.findByText(/409|worker busy/i);
    // Reverted back to the project's last-known-good dit_profile.
    expect(isSelected(ditButton("iterate"))).toBe(true);
    expect(isSelected(ditButton("quality"))).toBe(false);
    // Never actually persisted server-side either.
    expect(app.server.state.detail.project.dit_profile).toBe("iterate");

    vi.stubGlobal("fetch", previousFetch);
  });

  it("never offers studio_ops in this picker -- only a selected style pack forces that base-model swap", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });

    expect(within(ditGroup()).queryByRole("button", { name: "studio_ops" })).toBeNull();
    expect(
      within(ditGroup())
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual(["iterate", "polish", "quality"]);

    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    fireEvent.click(
      within(screen.getByRole("alertdialog", { name: "Load the studio model?" })).getByRole(
        "button",
        { name: "Load model & generate" },
      ),
    );

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({
      action: "generate",
      seed: -1,
      lora_id: "lora-good",
    });

    // The lora-driven swap never touches the project's own dit_profile
    // picker or PATCHes it to studio_ops.
    expect(isSelected(ditButton("iterate"))).toBe(true);
    expect(projectPatchRequests()).toHaveLength(0);
  });
});

async function openWithPlan(plan: Partial<ReturnType<typeof makeDefaultPlan>>) {
  app = await renderOpenedProject();
  // Reopen against a plan in the given state.
  app.cleanup();
  const server = createMockLyreServer();
  server.state.detail = {
    ...server.state.detail,
    plan: { ...makeDefaultPlan(), ...plan },
  };
  server.install();
  const rendered = render(<App />);
  app = {
    server,
    confirm: vi.fn(),
    cleanup: () => {
      rendered.unmount();
      server.uninstall();
    },
  };
  fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
  await screen.findByLabelText("Caption");
  return server;
}

describe("Creation screen (SPEC.md sec 16.3/16.6)", () => {
  it("starts a new project from an empty brief that plans from the idea", async () => {
    app = await renderOpenedProject();
    fireEvent.click(screen.getByRole("button", { name: "Projects" }));
    fireEvent.click(screen.getAllByRole("button", { name: /New project/ })[0]);
    fireEvent.change(await screen.findByLabelText(/Title/), { target: { value: "Fresh" } });
    fireEvent.click(screen.getByRole("button", { name: /^Create/ }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Fresh" })).toBeTruthy());
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value).toBe("");
    expect(screen.getByText("Nothing is locked yet.")).toBeTruthy();
    expect(screen.getByText("Generate lets ACE-Step choose everything")).toBeTruthy();
    fireEvent.change(screen.getByPlaceholderText(/Describe the song/), {
      target: { value: "rain on a tin roof" },
    });
    expect(screen.getByText("Generate plans from your idea")).toBeTruthy();
  });

  it("makes another take of a planned brief without re-planning, and re-plans on request", async () => {
    const server = await openWithPlan({
      query: "rain",
      caption: "auto: rain",
      lyrics: "[Verse]\nrain",
      creation_mode: "from_brief",
      field_origins: { ...makeDefaultPlan().field_origins, caption: "planner", lyrics: "planner" },
      brief: { source_query: "rain", job_id: "job-0", content_hash: "sha256:x" },
    });
    expect(screen.getByText("Generate makes another take of this brief")).toBeTruthy();
    expect(screen.getAllByText("planned").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => expect(server.jobRequests("generate")).toHaveLength(1));
    expect(server.jobRequests("generate")[0].body).toEqual({ action: "generate", seed: -1 });
    await screen.findByText("Complete: generate");

    fireEvent.click(screen.getByRole("button", { name: "Re-plan from idea" }));
    await waitFor(() => expect(server.jobRequests("generate")).toHaveLength(2));
    expect(server.jobRequests("generate")[1].body).toEqual({
      action: "generate",
      seed: -1,
      creation_mode: "from_idea",
    });
  });

  it("re-plans a changed idea and keeps text the user wrote", async () => {
    await openWithPlan({
      query: "rain",
      caption: "auto: rain",
      lyrics: "[Verse]\nmy own words",
      creation_mode: "from_brief",
      field_origins: { ...makeDefaultPlan().field_origins, caption: "planner", lyrics: "user" },
      brief: { source_query: "rain", job_id: "job-0", content_hash: "sha256:x" },
    });
    fireEvent.change(screen.getByPlaceholderText(/Describe the song/), {
      target: { value: "snow" },
    });
    expect(await screen.findByText(/You changed the idea since it was planned/)).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Generate plans from your idea")).toBeTruthy());
    expect(screen.getByText("Yours, kept as set: lyrics.")).toBeTruthy();
  });

  it("marks an explicit field as yours", async () => {
    await openWithPlan({ query: "rain" });
    fireEvent.change(screen.getByLabelText("BPM"), { target: { value: "92" } });
    await screen.findByText("Saved");
    expect(await screen.findByText("Yours, kept as set: BPM.")).toBeTruthy();
    const bpmLabel = screen.getByLabelText("BPM").closest("label")!;
    expect(within(bpmLabel).getByText("yours")).toBeTruthy();
  });

  it("asks for a deliberate choice when Instrumental conflicts with lyrics", async () => {
    const server = await openWithPlan({
      query: "rain",
      lyrics: "[Verse]\nwords I wrote",
      field_origins: { ...makeDefaultPlan().field_origins, lyrics: "user" },
    });
    fireEvent.click(screen.getByLabelText("Instrumental"));
    expect(await screen.findByText(/Lyre will not delete them for you/)).toBeTruthy();
    const generate = screen.getByRole("button", { name: "Generate" }) as HTMLButtonElement;
    expect(generate.disabled).toBe(true);
    expect(generate.title).toMatch(/conflict/);
    // The words are still there.
    expect((screen.getByLabelText("Lyrics") as HTMLTextAreaElement).value).toBe(
      "[Verse]\nwords I wrote",
    );

    fireEvent.click(screen.getByRole("button", { name: "Keep vocals" }));
    await waitFor(() => expect(generate.disabled).toBe(false));
    await waitFor(() => expect(server.state.detail.plan.instrumental).toBe(false));
    expect(server.state.detail.plan.lyrics).toBe("[Verse]\nwords I wrote");
  });
});

describe("Readiness and presets (SPEC.md sec 16.6)", () => {
  it("says what each profile is and disables one that cannot load, with the reason", async () => {
    app = await renderOpenedProject();
    const caps = makeCapabilities();
    caps.profiles = caps.profiles.map((profile) =>
      profile.id === "quality"
        ? {
            ...profile,
            files: "missing",
            supported: false,
            reason: "model files missing: acestep-v15-xl-turbo",
          }
        : profile,
    );
    app.server.state.capabilities = caps;
    await waitFor(() => expect(ditButton("quality").disabled).toBe(true), { timeout: 7000 });
    expect(ditButton("quality").title).toMatch(/Unavailable: model files missing/);
    expect(ditButton("iterate").title).toMatch(/has rendered on this machine/);
    expect(ditButton("polish").title).toMatch(/not yet validated/);
  }, 10000);

  it("tells a stopped worker apart from a ready one", async () => {
    app = await renderOpenedProject();
    expect(await screen.findByText("Ready")).toBeTruthy();
    app.server.state.capabilities = makeCapabilities({
      worker: {
        state: "stopped",
        reason: "No worker has reported in. Start it with ./scripts/lyre worker.",
        backend: "acestep",
        dit_loaded: null,
        queue: { queued: 0, running: 0 },
      },
    });
    expect(await screen.findByText("Worker stopped", {}, { timeout: 7000 })).toBeTruthy();
    expect(screen.getByText(/Start it with .\/scripts\/lyre worker/)).toBeTruthy();
  }, 10000);

  it("previews a preset before applying it and keeps the user's lyrics", async () => {
    await openWithPlan({
      lyrics: "[Verse]\nmine",
      revision: 2,
      field_origins: { ...makeDefaultPlan().field_origins, lyrics: "user" },
    });

    fireEvent.change(await screen.findByLabelText("Preset"), { target: { value: "combat_cue" } });
    const preview = await screen.findByRole("region", { name: "Preset preview" });
    expect(within(preview).getByText(/bpm: unset → 150/)).toBeTruthy();
    expect(within(preview).getByText(/stays a song/)).toBeTruthy();
    // Nothing is written by a preview.
    expect(app!.server.state.detail.plan.bpm).toBeNull();

    fireEvent.click(within(preview).getByRole("button", { name: "Apply preset" }));
    await waitFor(() => expect(app!.server.state.detail.plan.bpm).toBe(150));
    expect(app!.server.state.detail.plan.lyrics).toBe("[Verse]\nmine");
    expect(app!.server.state.detail.plan.instrumental).toBe(false);
    expect((screen.getByLabelText("BPM") as HTMLInputElement).value).toBe("150");
    expect(screen.queryByRole("region", { name: "Preset preview" })).toBeNull();
  });
});
