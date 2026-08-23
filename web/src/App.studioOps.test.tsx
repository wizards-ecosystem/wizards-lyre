// Regression tests for the shipped Extract/Lego/Complete studio-ops flows
// (SPEC.md sec 4.3 "studio_ops base model"): the track-name-gated buttons in
// the waveform action bar, the base-model-swap confirm() gate they all share
// with Generate/Cover/Repaint, and the job bodies they post. Everything runs
// against the mocked fetch backend in src/test/mockServer.ts -- no FastAPI,
// CUDA, ACE-Step, credentials, or generated audio.
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderOpenedProject, type OpenedProject } from "./test/renderApp";

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

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function takeRows(): HTMLElement[] {
  const pane = screen.getByRole("heading", { name: "Takes" }).closest("section");
  if (!pane) throw new Error("takes pane not found");
  return within(pane).getAllByRole("listitem");
}

function selectFirstTake(): void {
  fireEvent.click(takeRows()[0]);
}

function trackNameInput(): HTMLInputElement {
  return screen.getByLabelText("Track name / classes") as HTMLInputElement;
}

function setTrackName(value: string): void {
  fireEvent.change(trackNameInput(), { target: { value } });
}

function extractButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: "Extract" }) as HTMLButtonElement;
}

function legoButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: "Lego" }) as HTMLButtonElement;
}

function completeButton(): HTMLButtonElement {
  return screen.getByRole("button", { name: "Complete" }) as HTMLButtonElement;
}

function jobsPost(action?: string) {
  if (!app) throw new Error("app not rendered");
  return app.server.jobRequests(action);
}

describe("Extract/Lego/Complete studio-ops actions (SPEC.md sec 4.3)", () => {
  it("keeps Extract/Lego/Complete disabled until a take is selected and a track name is entered", async () => {
    app = await renderOpenedProject();

    expect(extractButton().disabled).toBe(true);
    expect(legoButton().disabled).toBe(true);
    expect(completeButton().disabled).toBe(true);
    expect(extractButton().title).toBe("Select a take first");
    expect(legoButton().title).toBe("Select a take first");
    expect(completeButton().title).toBe("Select a take first");

    // A track name alone (no take) is not enough.
    setTrackName("vocals");
    expect(extractButton().disabled).toBe(true);
    expect(legoButton().disabled).toBe(true);
    expect(completeButton().disabled).toBe(true);

    // Reset the name and select a take first -- a take alone is not enough
    // either.
    setTrackName("");
    selectFirstTake();
    expect(extractButton().disabled).toBe(true);
    expect(extractButton().title).toBe("Enter a track name first");
    expect(legoButton().disabled).toBe(true);
    expect(completeButton().disabled).toBe(true);
    expect(completeButton().title).toBe("Enter a track name / classes first");

    setTrackName("vocals");
    expect(extractButton().disabled).toBe(false);
    expect(legoButton().disabled).toBe(false);
    expect(completeButton().disabled).toBe(false);
  });

  it("posts an extract job after the base-swap confirm", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("vocals");
    fireEvent.click(extractButton());

    await waitFor(() => expect(jobsPost("extract")).toHaveLength(1));
    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(String(app.confirm.mock.calls[0][0])).toMatch(/studio_ops base model/);
    expect(jobsPost("extract")[0].url).toBe("/api/projects/proj-1/jobs");
    expect(jobsPost("extract")[0].body).toEqual({
      action: "extract",
      dit_profile: "studio_ops",
      source_take_id: "take-01",
      track_name: "vocals",
      seed: -1,
    });
  });

  it("posts nothing when the base-swap confirmation is declined for Extract", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("vocals");
    app.confirm.mockReturnValue(false);
    fireEvent.click(extractButton());

    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(jobsPost("extract")).toHaveLength(0);
  });

  it("posts a lego job after the base-swap confirm", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("drums");
    fireEvent.click(legoButton());

    await waitFor(() => expect(jobsPost("lego")).toHaveLength(1));
    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(String(app.confirm.mock.calls[0][0])).toMatch(/studio_ops base model/);
    expect(jobsPost("lego")[0].body).toEqual({
      action: "lego",
      dit_profile: "studio_ops",
      source_take_id: "take-01",
      track_name: "drums",
      seed: -1,
    });
  });

  it("posts nothing when the base-swap confirmation is declined for Lego", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("drums");
    app.confirm.mockReturnValue(false);
    fireEvent.click(legoButton());

    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(jobsPost("lego")).toHaveLength(0);
  });

  it("carries the waveform region into the lego job body when one is set", async () => {
    app = await renderOpenedProject();
    selectFirstTake();

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

    setTrackName("bass");
    fireEvent.click(legoButton());

    await waitFor(() => expect(jobsPost("lego")).toHaveLength(1));
    expect(jobsPost("lego")[0].body).toEqual({
      action: "lego",
      dit_profile: "studio_ops",
      source_take_id: "take-01",
      track_name: "bass",
      seed: -1,
      repainting_start: 4.5,
      repainting_end: 9.25,
    });
  });

  it("posts a complete job after the base-swap confirm", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("full mix");
    fireEvent.click(completeButton());

    await waitFor(() => expect(jobsPost("complete")).toHaveLength(1));
    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(String(app.confirm.mock.calls[0][0])).toMatch(/studio_ops base model/);
    expect(jobsPost("complete")[0].body).toEqual({
      action: "complete",
      dit_profile: "studio_ops",
      source_take_id: "take-01",
      track_name: "full mix",
      seed: -1,
    });
  });

  it("posts nothing when the base-swap confirmation is declined for Complete", async () => {
    app = await renderOpenedProject();
    selectFirstTake();
    setTrackName("full mix");
    app.confirm.mockReturnValue(false);
    fireEvent.click(completeButton());

    expect(app.confirm).toHaveBeenCalledTimes(1);
    expect(jobsPost("complete")).toHaveLength(0);
  });

  it("keeps a failed extract job's error visible", async () => {
    app = await renderOpenedProject();
    app.server.scriptNextJob({ statuses: ["error"], error: "CUDA out of memory" });
    selectFirstTake();
    setTrackName("vocals");
    fireEvent.click(extractButton());

    expect(await screen.findByText("CUDA out of memory")).toBeTruthy();
  });
});
