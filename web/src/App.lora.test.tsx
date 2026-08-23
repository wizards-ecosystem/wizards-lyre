// Regression tests for the shipped LoRA train/load UX (SPEC.md sec 4.4
// "Style pack | LoRA train / load"): selecting 8+ takes, the Train button's
// validation, enqueueing train_lora, listing successful/failed packs,
// selecting a successful pack, and forwarding its lora_id through
// Generate/Cover/Repaint. Everything runs against the mocked fetch backend
// in src/test/mockServer.ts -- no FastAPI, CUDA, ACE-Step, credentials, or
// generated audio.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { createMockBardServer, makeLora, makeTake, makeTakes, PROJECT_ID } from "./test/mockServer";
import { LORA_SOURCE_TITLE, renderOpenedProject, type OpenedProject } from "./test/renderApp";

// Mirrors App.tsx's LORA_TRAIN_RECOVERY_POLL_MS (not exported) -- the
// cadence of the recovery poll that watches a still-active train_lora job
// found via GET /api/jobs.
const LORA_TRAIN_RECOVERY_POLL_MS = 3000;

interface FakeRegion {
  id: string;
  start: number;
  end: number;
  remove: () => void;
}

// jsdom has no canvas/layout for wavesurfer.js, so the waveform stack is
// stubbed out. The regions stub keeps the event handlers App registers so
// tests can simulate "the user dragged a region" by invoking them directly.
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

const GOOD_LORA = makeLora("lora-good", "good-style", { final_loss: 0.1234 });
const FAILED_LORA = makeLora("lora-bad", "bad-style", {
  status: null,
  final_step: null,
  final_loss: null,
  error: "training diverged",
});

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function loraSourceCheckboxes(): HTMLElement[] {
  return screen.getAllByTitle(LORA_SOURCE_TITLE);
}

function checkTakes(count: number): void {
  for (const cb of loraSourceCheckboxes().slice(0, count)) {
    fireEvent.click(cb);
  }
}

function trainButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: /Train style pack/ }) as HTMLButtonElement;
}

function nameInput(): HTMLInputElement {
  return screen.getByLabelText("Style pack name") as HTMLInputElement;
}

function loraSelect(): HTMLSelectElement {
  // The only <select> in the app is the "Style pack" picker in the
  // Waveform pane's action bar.
  return screen.getByRole("combobox") as HTMLSelectElement;
}

function takeRows(): HTMLElement[] {
  const pane = screen.getByRole("heading", { name: "Takes" }).closest("section");
  if (!pane) throw new Error("takes pane not found");
  return within(pane).getAllByRole("listitem");
}

function jobsPost(action?: string) {
  if (!app) throw new Error("app not rendered");
  return app.server.jobRequests(action);
}

const RECOVERED_JOB_ID = "job-recovered";

// Like renderOpenedProject, but seeds a train_lora job directly into the
// mock server's job queue *before* the app ever renders -- simulating a
// training that was started before this page load (a refresh mid-training,
// or another tab) and must be recovered via GET /api/jobs on project open,
// not via this test clicking Train.
async function renderProjectWithSeededTraining(status: string): Promise<OpenedProject> {
  const server = createMockBardServer();
  server.seedJob(RECOVERED_JOB_ID, PROJECT_ID, "train_lora", status);
  server.install();

  const originalConfirm = window.confirm;
  const confirm = vi.fn(() => true);
  window.confirm = confirm as unknown as typeof window.confirm;

  const rendered = render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "Open" }));
  await waitFor(() => {
    expect(screen.getAllByTitle(LORA_SOURCE_TITLE)).toHaveLength(
      server.state.detail.takes.length,
    );
  });

  return {
    server,
    confirm,
    cleanup: () => {
      rendered.unmount();
      server.uninstall();
      window.confirm = originalConfirm;
    },
  };
}

describe("LoRA style packs (SPEC.md sec 4.4)", () => {
  it("keeps Train style pack disabled until 8+ distinct takes are checked and a name is entered", async () => {
    app = await renderOpenedProject();
    const train = trainButton();
    expect(train.disabled).toBe(true);
    expect(train.title).toBe("Select at least 8 takes first");

    // Toggling the same take twice counts it once (selection is a Set).
    fireEvent.click(loraSourceCheckboxes()[0]);
    fireEvent.click(loraSourceCheckboxes()[0]);
    expect(screen.getByText(/Selected: 0\/8/)).toBeTruthy();

    checkTakes(7);
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    expect(trainButton().disabled).toBe(true);

    fireEvent.click(loraSourceCheckboxes()[7]);
    expect(screen.getByText(/Selected: 8\/8/)).toBeTruthy();
    expect(trainButton().disabled).toBe(false);

    // The name is required too.
    fireEvent.change(nameInput(), { target: { value: "   " } });
    expect(trainButton().disabled).toBe(true);
    expect(trainButton().title).toBe("Enter a style pack name first");

    // Dropping below the 8-take floor disables it again.
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    fireEvent.click(loraSourceCheckboxes()[0]);
    expect(trainButton().disabled).toBe(true);
  });

  it("can never enable Train style pack when the project has fewer than 8 takes", async () => {
    app = await renderOpenedProject({ takes: makeTakes(7) });
    checkTakes(7);
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    expect(trainButton().disabled).toBe(true);
    expect(trainButton().title).toBe("Select at least 8 takes first");
  });

  it("enqueues train_lora with the selected take IDs and pack name", async () => {
    app = await renderOpenedProject();
    checkTakes(8);
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    fireEvent.click(trainButton());

    // The trained pack shows up both in the list and (being successful) in
    // the Style pack picker once the post-train refresh lands.
    await waitFor(() => {
      expect(screen.getAllByText("my-style")).toHaveLength(2);
    });

    const posts = jobsPost("train_lora");
    expect(posts).toHaveLength(1);
    expect(posts[0].url).toBe("/api/projects/proj-1/jobs");
    expect(posts[0].body).toEqual({
      action: "train_lora",
      source_take_ids: [
        "take-01",
        "take-02",
        "take-03",
        "take-04",
        "take-05",
        "take-06",
        "take-07",
        "take-08",
      ],
      name: "my-style",
      seed: -1,
    });

    // A successful train resets the train panel for the next pack.
    expect(loraSourceCheckboxes().some((el) => (el as HTMLInputElement).checked)).toBe(false);
    expect(nameInput().value).toBe("");
    expect(trainButton().disabled).toBe(true);
  });

  it("lists successful and failed style packs with their status", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA, FAILED_LORA] });
    // A successful pack shows up twice: in the Style packs list and as an
    // option in the Style pack picker.
    expect(await screen.findAllByText("good-style")).toHaveLength(2);
    expect(screen.getByText("bad-style")).toBeTruthy();
    expect(screen.getByText("done")).toBeTruthy();
    expect(screen.getByText("error: training diverged")).toBeTruthy();
    expect(screen.getByText("loss 0.1234")).toBeTruthy();
  });

  it("lets only successful packs be selected for use", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA, FAILED_LORA] });
    const select = await screen.findByRole("combobox");
    const optionNames = Array.from((select as HTMLSelectElement).options).map(
      (o) => o.textContent,
    );
    // The failed pack is listed (with its error) but never offered for use.
    expect(optionNames).toEqual(["None", "good-style"]);

    fireEvent.change(select, { target: { value: "lora-good" } });
    expect((select as HTMLSelectElement).value).toBe("lora-good");
  });

  it("forwards the selected lora_id through Generate after the base-swap confirm", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });
    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(String(app.confirm.mock.calls[0][0])).toMatch(/studio_ops base model/);
    expect(jobsPost("generate")[0].body).toEqual({
      action: "generate",
      seed: -1,
      lora_id: "lora-good",
    });
  });

  it("enqueues nothing when the base-swap confirmation is declined", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });
    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    app.confirm.mockReturnValue(false);
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(jobsPost()).toHaveLength(0);
  });

  it("forwards the selected lora_id through Cover", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });
    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    fireEvent.click(takeRows()[0]); // use the newest take as the cover source
    fireEvent.click(screen.getByRole("button", { name: "Cover" }));

    await waitFor(() => expect(jobsPost("cover")).toHaveLength(1));
    expect(jobsPost("cover")[0].body).toEqual({
      action: "cover",
      source_take_id: "take-01",
      audio_cover_strength: 0.7,
      seed: -1,
      lora_id: "lora-good",
    });
  });

  it("forwards the selected lora_id through Repaint", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });
    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    fireEvent.click(takeRows()[0]);

    // Simulate dragging a region on the waveform once the regions plugin
    // for the selected take has registered its handlers.
    await waitFor(() => expect(regionHooks.handlers["region-created"]).toBeDefined());
    await act(async () => {
      regionHooks.handlers["region-created"]({
        id: "region-1",
        start: 4.5,
        end: 9.25,
        remove: () => {},
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "Repaint" }));
    await waitFor(() => expect(jobsPost("repaint")).toHaveLength(1));
    expect(jobsPost("repaint")[0].body).toEqual({
      action: "repaint",
      source_take_id: "take-01",
      repainting_start: 4.5,
      repainting_end: 9.25,
      seed: -1,
      lora_id: "lora-good",
    });
  });

  it("keeps a failed train_lora job's error visible and the selection intact", async () => {
    app = await renderOpenedProject();
    app.server.scriptNextJob({ statuses: ["error"], error: "CUDA out of memory" });
    checkTakes(8);
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    fireEvent.click(trainButton());

    expect(await screen.findByText("CUDA out of memory")).toBeTruthy();
    // The user's selection survives the failure so they can retry.
    expect(loraSourceCheckboxes().filter((el) => (el as HTMLInputElement).checked)).toHaveLength(
      8,
    );
    expect(nameInput().value).toBe("my-style");
    // No pack was added by the failed training.
    expect(screen.getByText("No style packs trained yet.")).toBeTruthy();
    expect(jobsPost("train_lora")).toHaveLength(1);
  });

  it("keeps the API error visible when enqueueing train_lora is rejected", async () => {
    app = await renderOpenedProject();
    app.server.failNextJobsPost(400, {
      detail: "action 'train_lora' requires at least 8 distinct source takes",
    });
    checkTakes(8);
    fireEvent.change(nameInput(), { target: { value: "my-style" } });
    fireEvent.click(trainButton());

    expect(
      await screen.findByText(/requires at least 8 distinct source takes/),
    ).toBeTruthy();
    expect(jobsPost()).toHaveLength(1); // the request was made and rejected
  });

  it("keeps a failed generate job's error visible", async () => {
    app = await renderOpenedProject({ loras: [GOOD_LORA] });
    app.server.scriptNextJob({ statuses: ["error"], error: "worker crashed" });
    fireEvent.change(loraSelect(), { target: { value: "lora-good" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("worker crashed")).toBeTruthy();
  });

  it("recovers an in-progress training after opening a project with an active train_lora job", async () => {
    app = await renderProjectWithSeededTraining("running");

    // Recovered purely from GET /api/jobs on project open -- this test
    // never clicked Train.
    expect(screen.getByText("Training style pack…")).toBeTruthy();
    expect(screen.getByText("running")).toBeTruthy();
    expect(app.server.jobRequests("train_lora")).toHaveLength(0);
    const train = screen.getByRole("button", { name: /Training…/ }) as HTMLButtonElement;
    expect(train.disabled).toBe(true);
    expect(train.title).toBe("A style pack is already training for this project");
  });

  it(
    "refreshes the pack list once a recovered training job finishes",
    async () => {
      // jsdom has no MessageChannel, so React 18 falls back to setTimeout to
      // flush state updates that land outside a synthetic event handler
      // (e.g. after an awaited fetch resolves) -- vitest's fake timers
      // would freeze that flush entirely along with the recovery poll's own
      // setInterval, hanging the test. Advancing past the real poll
      // interval on the real clock avoids that trap.
      app = await renderProjectWithSeededTraining("running");
      expect(screen.getByText("Training style pack…")).toBeTruthy();

      // The training finishes behind the scenes (as it would in another
      // tab, or a worker that outlives this page's own poller) -- flip the
      // job to done and make the pack listable, mirroring what
      // server.jobs._run_train_lora_job does before the job row flips.
      app.server.state.loras = [GOOD_LORA];
      app.server.seedJob(RECOVERED_JOB_ID, PROJECT_ID, "train_lora", "done");

      await waitFor(
        () => {
          expect(screen.queryByText("Training style pack…")).toBeNull();
        },
        { timeout: LORA_TRAIN_RECOVERY_POLL_MS + 2000 },
      );
      expect(screen.getAllByText("good-style").length).toBeGreaterThan(0);
    },
    LORA_TRAIN_RECOVERY_POLL_MS + 5000,
  );

  it("shows a clickable style badge for a take generated with a still-loadable pack", async () => {
    app = await renderOpenedProject({
      takes: [makeTake("take-01", 1, { lora_id: GOOD_LORA.id })],
      loras: [GOOD_LORA],
    });

    const badge = await screen.findByRole("button", { name: "style: good-style" });
    expect(badge.title).toBe(
      'Generated with style pack "good-style" — click to select it for the next Generate/Cover/Repaint',
    );

    fireEvent.click(badge);
    expect(loraSelect().value).toBe("lora-good");
  });

  it("shows a non-clickable style badge for a take whose pack failed training", async () => {
    app = await renderOpenedProject({
      takes: [makeTake("take-01", 1, { lora_id: FAILED_LORA.id })],
      loras: [FAILED_LORA],
    });

    const badge = await screen.findByText("style: bad-style");
    expect(badge.tagName).toBe("SPAN");
    expect(badge.title).toBe(
      `Generated with style pack "bad-style" (${FAILED_LORA.id}) — it failed training and cannot be loaded`,
    );
  });

  it("shows a truncated-id style badge for a take whose pack no longer resolves in this project", async () => {
    app = await renderOpenedProject({
      takes: [makeTake("take-01", 1, { lora_id: "lora-vanished-id" })],
      loras: [],
    });

    const badge = await screen.findByText("style: lora-van");
    expect(badge.tagName).toBe("SPAN");
    expect(badge.title).toBe(
      "Generated with style pack lora-vanished-id (not found in this project)",
    );
  });
});
