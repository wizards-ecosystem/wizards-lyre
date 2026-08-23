// Regression tests for the shipped LoRA train/load UX (SPEC.md sec 4.4
// "Style pack | LoRA train / load"): selecting 8+ takes, the Train button's
// validation, enqueueing train_lora, listing successful/failed packs,
// selecting a successful pack, and forwarding its lora_id through
// Generate/Cover/Repaint. Everything runs against the mocked fetch backend
// in src/test/mockServer.ts -- no FastAPI, CUDA, ACE-Step, credentials, or
// generated audio.
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { makeLora, makeTakes } from "./test/mockServer";
import { LORA_SOURCE_TITLE, renderOpenedProject, type OpenedProject } from "./test/renderApp";

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
});
