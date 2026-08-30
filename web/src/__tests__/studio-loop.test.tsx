// Browser regression tests for the shipped Phase 1/2 studio loop: saving the
// current plan before Generate, polling a queued job until its take
// appears, picking a take (or a dropped file) as the Cover/Repaint source,
// forwarding cover strength and a dragged waveform region, clearing the
// region on selection changes, and uploading a WAV as an alternate source.
// Runs against the mocked fetch backend in src/test/mockServer.ts -- no
// FastAPI, audio device, GPU, or ACE-Step dependency. Independent of the
// library/take-metadata/base-swap browser suites: new file, test-local
// wavesurfer/regions stub and mock-server fixtures only.
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { makeTakes } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

interface FakeRegion {
  id: string;
  start: number;
  end: number;
  remove: () => void;
}

// jsdom has no canvas/layout for wavesurfer.js, so the waveform stack is
// stubbed out -- same shape as App.lora.test.tsx's stub. The regions stub
// keeps the event handlers App registers so a test can simulate "the user
// dragged a region" by invoking region-created directly.
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

function planPuts() {
  if (!app) throw new Error("app not rendered");
  return app.server.requests.filter((r) => r.method === "PUT" && r.url.endsWith("/plan"));
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
    regionHooks.handlers["region-created"]({
      id: "region-1",
      start,
      end,
      remove: () => {},
    });
  });
}

function dropzone(): HTMLElement {
  return screen.getByText("Drop WAV or MP3").closest(".source-shelf") as HTMLElement;
}

function dropFile(name: string): File {
  return new File(["RIFF....WAVEfmt "], name, { type: "audio/wav" });
}

describe("Studio loop: Generate/Cover/Repaint and upload-source (SPEC.md sec 4/7/9)", () => {
  it("flushes a pending plan edit before Generate enqueues a job, and the new take appears after polling", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });

    // Edit the caption but click Generate immediately, inside the plan-save
    // debounce window (PLAN_SAVE_DEBOUNCE_MS) -- Generate must flush this
    // edit itself rather than racing it.
    fireEvent.change(screen.getByLabelText("Caption"), {
      target: { value: "a freshly rewritten caption" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(planPuts()).toHaveLength(1);
    expect(planPuts()[0].body).toMatchObject({ caption: "a freshly rewritten caption" });

    // The PUT /plan must land before the job is enqueued, not just at some
    // point during the click.
    const putIndex = app.server.requests.indexOf(planPuts()[0]);
    const postIndex = app.server.requests.indexOf(jobsPost("generate")[0]);
    expect(putIndex).toBeLessThan(postIndex);

    // Generate stays busy until the queued job's poll reports done, then the
    // resulting take shows up in the Takes list without a manual reload.
    await waitFor(() => expect(screen.getAllByText("seed 1002").length).toBeGreaterThan(0));
    expect(screen.getByRole("button", { name: "Generate" })).toBeTruthy();
  });

  it("keeps a failed generate job's error visible", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["error"], error: "worker crashed" });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("worker crashed")).toBeTruthy();
    // A failed job never manufactures a take.
    expect(screen.queryByText("seed 1002")).toBeNull();
  });

  it("selects a take as the Cover source and forwards the chosen cover strength", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.click(takeRows()[0]);

    fireEvent.change(screen.getByLabelText("Strength"), { target: { value: "0.35" } });
    fireEvent.click(screen.getByRole("button", { name: "Cover" }));

    await waitFor(() => expect(jobsPost("cover")).toHaveLength(1));
    expect(jobsPost("cover")[0].body).toEqual({
      action: "cover",
      source_take_id: "take-01",
      audio_cover_strength: 0.35,
      seed: -1,
    });
  });

  it("keeps a failed cover job's error visible and the take selection intact", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["error"], error: "cover job crashed" });
    fireEvent.click(takeRows()[0]);
    fireEvent.click(screen.getByRole("button", { name: "Cover" }));

    expect(await screen.findByText("cover job crashed")).toBeTruthy();
    expect(takeRows()[0].className).toContain("selected");
  });

  it("forwards a dragged waveform region as repainting_start/repainting_end, then clears it on success", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.click(takeRows()[0]);

    await dragRegion(4.5, 9.25);
    expect(screen.getByText(/Region: 4\.5s.*9\.3s/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Repaint" }));
    await waitFor(() => expect(jobsPost("repaint")).toHaveLength(1));
    expect(jobsPost("repaint")[0].body).toEqual({
      action: "repaint",
      source_take_id: "take-01",
      repainting_start: 4.5,
      repainting_end: 9.25,
      seed: -1,
    });

    // A successful repaint clears the drawn region so a stale selection
    // can't be resubmitted against the next take.
    await waitFor(() => expect(screen.queryByText(/Region:/)).toBeNull());
  });

  it("clears the region when the selected take changes", async () => {
    app = await renderOpenedProject({ takes: makeTakes(2) });
    fireEvent.click(takeRows()[0]);
    await dragRegion(2, 6);
    expect(screen.getByText(/Region: 2\.0s.*6\.0s/)).toBeTruthy();

    fireEvent.click(takeRows()[1]);
    expect(screen.queryByText(/Region:/)).toBeNull();
  });

  it("keeps a failed repaint job's error visible", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["error"], error: "repaint job crashed" });
    fireEvent.click(takeRows()[0]);
    await dragRegion(1, 3);
    fireEvent.click(screen.getByRole("button", { name: "Repaint" }));

    expect(await screen.findByText("repaint job crashed")).toBeTruthy();
    // The region survives a failure so the user can retry without re-dragging.
    expect(screen.getByText(/Region: 1\.0s.*3\.0s/)).toBeTruthy();
  });

  it("uploads a WAV as an alternate Cover/Repaint source and forwards it instead of a take", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.click(takeRows()[0]); // starts as the selected source

    fireEvent.drop(dropzone(), { dataTransfer: { files: [dropFile("my-song.wav")] } });

    await screen.findByText("my-song.wav");
    // Uploading a file deselects whatever take was picked (the two are
    // alternative sources, never both).
    expect(takeRows()[0].className).not.toContain("selected");

    fireEvent.click(screen.getByRole("button", { name: "Cover" }));
    await waitFor(() => expect(jobsPost("cover")).toHaveLength(1));
    expect(jobsPost("cover")[0].body).toMatchObject({
      action: "cover",
      upload_path: "uploads/upload-1.wav",
    });
    expect(jobsPost("cover")[0].body).not.toHaveProperty("source_take_id");
  });

  it("uploads a WAV and forwards it through Repaint as the full-track source", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.drop(dropzone(), { dataTransfer: { files: [dropFile("alt-source.wav")] } });
    await screen.findByText("alt-source.wav");

    fireEvent.click(screen.getByRole("button", { name: "Repaint" }));
    await waitFor(() => expect(jobsPost("repaint")).toHaveLength(1));
    expect(jobsPost("repaint")[0].body).toEqual({
      action: "repaint",
      upload_path: "uploads/upload-1.wav",
      repainting_start: 0,
      repainting_end: -1,
      seed: -1,
    });
  });

  it("keeps a rejected upload's error visible and the dropzone usable", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.failNextUpload(400, { detail: "unsupported file type" });

    fireEvent.drop(dropzone(), { dataTransfer: { files: [dropFile("bad.aiff")] } });

    expect(await screen.findByText(/unsupported file type/)).toBeTruthy();
    // The rejected upload never became the active source.
    expect(screen.queryByText("bad.aiff")).toBeNull();
    expect(screen.getByText("Drop WAV or MP3")).toBeTruthy();

    // The dropzone still works for a subsequent, successful upload.
    fireEvent.drop(dropzone(), { dataTransfer: { files: [dropFile("good.wav")] } });
    await screen.findByText("good.wav");
  });
});
