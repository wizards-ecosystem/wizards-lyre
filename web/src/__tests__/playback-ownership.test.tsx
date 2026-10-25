// SPEC.md sec 16.6: one playback owner, A/B that keeps the playhead, a failed
// preview that does not hide the file, shortcuts that leave typing alone,
// and "Use settings" that copies a take into the plan without touching the
// kept take.
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeTake } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

const waveform = vi.hoisted(() => ({ playPause: vi.fn(() => Promise.resolve()) }));

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => {}, destroy: () => {}, playPause: waveform.playPause }),
  },
}));
vi.mock("wavesurfer.js/plugins/regions", () => ({
  default: {
    create: () => ({
      on: () => {},
      getRegions: () => [],
      enableDragSelection: () => {},
      clearRegions: () => {},
    }),
  },
}));

// jsdom never changes `paused`; this fake tracks it so ownership is visible.
const playing = new WeakSet<HTMLMediaElement>();
const durations = new WeakMap<HTMLMediaElement, number>();
const positions = new WeakMap<HTMLMediaElement, number>();
const descriptors = {
  play: Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "play"),
  pause: Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "pause"),
  paused: Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "paused"),
  duration: Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "duration"),
  currentTime: Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "currentTime"),
};

beforeEach(() => {
  Object.defineProperty(HTMLMediaElement.prototype, "play", {
    configurable: true,
    value(this: HTMLMediaElement) {
      playing.add(this);
      this.dispatchEvent(new Event("play"));
      return Promise.resolve();
    },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "pause", {
    configurable: true,
    value(this: HTMLMediaElement) {
      if (!playing.delete(this)) return;
      this.dispatchEvent(new Event("pause"));
    },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "paused", {
    configurable: true,
    get(this: HTMLMediaElement) {
      return !playing.has(this);
    },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "duration", {
    configurable: true,
    get(this: HTMLMediaElement) {
      return durations.get(this) ?? NaN;
    },
  });
  Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
    configurable: true,
    get(this: HTMLMediaElement) {
      return positions.get(this) ?? 0;
    },
    set(this: HTMLMediaElement, value: number) {
      positions.set(this, value);
    },
  });
});

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
  for (const [name, descriptor] of Object.entries(descriptors)) {
    if (descriptor) Object.defineProperty(HTMLMediaElement.prototype, name, descriptor);
  }
  waveform.playPause.mockClear();
});

function takeRows(): HTMLElement[] {
  const pane = screen.getByRole("heading", { name: "Takes" }).closest("section")!;
  return within(pane as HTMLElement).getAllByRole("listitem");
}

function rowAudio(row: HTMLElement): HTMLAudioElement {
  return row.querySelector("audio")!;
}

describe("Playback ownership (SPEC.md sec 16.6)", () => {
  it("pauses whatever was playing when another take starts", async () => {
    app = await renderOpenedProject();
    const [first, second] = takeRows();
    fireEvent.click(within(first).getByRole("button", { name: "Play take take-01" }));
    expect(rowAudio(first).paused).toBe(false);

    fireEvent.click(within(second).getByRole("button", { name: "Play take take-02" }));
    expect(rowAudio(second).paused).toBe(false);
    expect(rowAudio(first).paused).toBe(true);
  });

  it("keeps the download link when a preview fails to load", async () => {
    app = await renderOpenedProject();
    const [first] = takeRows();
    fireEvent.error(rowAudio(first));
    expect(within(first).getByText(/Preview failed to load/)).toBeTruthy();
    expect(within(first).getByRole("link", { name: "download" })).toBeTruthy();
  });

  it("switches A/B at about the same position, clamped to the shorter take", async () => {
    app = await renderOpenedProject();
    const rows = takeRows();
    fireEvent.click(rows[1]); // A: take-02 (selected)
    fireEvent.click(within(rows[2]).getByRole("button", { name: "Compare" })); // B: take-03
    const pane = screen.getByRole("heading", { name: "Compare" }).closest("section")!;
    const [audioA, audioB] = Array.from(pane.querySelectorAll("audio"));
    expect(within(pane as HTMLElement).getByText("A: Take 09")).toBeTruthy();
    durations.set(audioA, 100);
    durations.set(audioB, 30);
    audioA.currentTime = 80;
    await act(async () => {
      await audioA.play();
    });

    fireEvent.click(screen.getByRole("button", { name: "Switch A ↔ B" }));
    expect(audioA.paused).toBe(true);
    expect(audioB.paused).toBe(false);
    expect(audioB.currentTime).toBeCloseTo(29.95);

    audioB.currentTime = 12;
    fireEvent.click(screen.getByRole("button", { name: "Switch A ↔ B" }));
    expect(audioA.currentTime).toBe(12);
    expect(audioA.paused).toBe(false);
    expect(audioB.paused).toBe(true);
    // Auditioning never writes anything: not the active take, not the plan.
    expect(app.server.requests.some((r) => r.method !== "GET")).toBe(false);
    expect(takeRows()[1].className).toMatch(/selected/);
  });

  it("leaves Space and G to the text field being typed in", async () => {
    app = await renderOpenedProject();
    const caption = screen.getByLabelText("Caption");
    caption.focus();
    fireEvent.keyDown(caption, { key: " ", code: "Space" });
    fireEvent.keyDown(caption, { key: "g" });
    const preset = screen.getByLabelText("Preset");
    preset.focus();
    fireEvent.keyDown(preset, { key: " ", code: "Space" });
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    expect(waveform.playPause).not.toHaveBeenCalled();
    expect(app.server.jobRequests()).toHaveLength(0);

    preset.blur();
    fireEvent.keyDown(window, { key: " ", code: "Space" });
    expect(waveform.playPause).toHaveBeenCalledTimes(1);
  });

  it("copies a take's rendered settings into the plan as a draft and keeps the kept take", async () => {
    const takes = [
      makeTake("take-01", 1),
      makeTake("take-02", 2, {
        seed: 4242,
        caption: "input caption",
        provenance: {
          complete: true,
          brief: {
            caption: "input caption",
            lyrics: "[Verse]\nfrom the take",
            instrumental: false,
          },
          actual: { caption: "what it rendered", bpm: 101, keyscale: "E minor", duration_sec: 33 },
        },
      }),
    ];
    app = await renderOpenedProject({ takes });
    const row = takeRows()[1];
    fireEvent.click(within(row).getByRole("button", { name: "Replay settings" }));

    expect(await screen.findByText(/holds Take 01's settings and seed as a draft/)).toBeTruthy();
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("what it rendered");
    expect((screen.getByLabelText("BPM") as HTMLInputElement).value).toBe("101");
    expect((screen.getByLabelText("Seed") as HTMLInputElement).value).toBe("4242");
    await waitFor(() => expect(app!.server.state.detail.plan.caption).toBe("what it rendered"));
    expect(app.server.state.detail.plan.lyrics).toBe("[Verse]\nfrom the take");
    // Nothing rendered and the kept take did not move.
    expect(app.server.jobRequests()).toHaveLength(0);
    expect(app.server.requests.some((r) => r.url.endsWith("/active_take"))).toBe(false);
    expect(takeRows()[0].className).toMatch(/active-take/);
  });

  it("warns that an older take only recorded part of its settings", async () => {
    app = await renderOpenedProject({ takes: [makeTake("take-01", 1, { caption: "old one" })] });
    fireEvent.click(within(takeRows()[0]).getByRole("button", { name: "Use settings" }));
    expect(await screen.findByText(/predates full provenance/)).toBeTruthy();
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("old one");
  });
});
