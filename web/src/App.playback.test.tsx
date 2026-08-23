// Regression tests for the shipped Phase 4/6 playback metadata (SPEC.md sec
// 10 "score/has_lrc" and sec 12 Phase 6 "live peak meter"): a take's quality
// score, the has_lrc-gated lyrics download, WAV playback/download targeting
// the right take, and the LoudnessMeter's Web Audio graph lifecycle
// (lazy-build on first play, live analyser readout, and teardown on
// pause/unmount/losing its row). This does not duplicate App.takes.test.tsx
// (favorite/notes/set-active/export) or App.ergonomics.test.tsx (A/B
// compare, keyboard shortcuts, parent-take restore) -- those own take
// metadata edits and general interaction shortcuts respectively; this file
// owns the read-only take metadata display and the audio/Web-Audio wiring
// neither of them touches.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with
// wavesurfer stubbed out (jsdom has no canvas/layout for the real library,
// same as every other App-level test) plus two mocks that live only in this
// file, since jsdom implements neither:
//   - window.AudioContext / AnalyserNode: LoudnessMeter builds a real Web
//     Audio graph lazily on an <audio> element's first 'play' event, which
//     jsdom does not implement at all.
//   - requestAnimationFrame, faked (via vitest fake timers) only for the one
//     test that needs to observe the meter's per-frame peak readout --
//     everything else in this file runs on real timers so the shared
//     mock-fetch/debounce machinery elsewhere in App.tsx is unaffected.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import {
  createMockBardServer,
  makeProjectDetail,
  makeProjectSummary,
  makeTake,
  makeTakes,
  PROJECT_ID,
} from "./test/mockServer";
import { LORA_SOURCE_TITLE, renderOpenedProject, type OpenedProject } from "./test/renderApp";

const { waveCreateSpy } = vi.hoisted(() => ({ waveCreateSpy: vi.fn() }));

vi.mock("wavesurfer.js", () => ({
  default: {
    create: (...args: unknown[]) => {
      waveCreateSpy(...args);
      return { on: () => {}, destroy: () => {} };
    },
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

// ---------------------------------------------------------------------------
// Web Audio mock -- jsdom has no AudioContext/AnalyserNode at all. This
// stands in for real hardware: `currentSample` drives what every analyser's
// getByteTimeDomainData() reports, and audioContextInstances records every
// context LoudnessMeter builds so tests can assert on close()/state without
// reaching into App internals.
// ---------------------------------------------------------------------------
let currentSample = 128; // 128 == silence: |128 - 128| / 128 == 0 peak.
let audioContextInstances: MockAudioContext[] = [];

class MockAnalyserNode {
  fftSize = 2048;
  connect = vi.fn();
  getByteTimeDomainData(arr: Uint8Array) {
    arr.fill(currentSample);
  }
}

class MockAudioContext {
  state: "running" | "suspended" | "closed" = "running";
  destination = {};
  createMediaElementSource = vi.fn(() => ({ connect: vi.fn() }));
  createAnalyser = vi.fn(() => new MockAnalyserNode());
  resume = vi.fn(async () => {
    this.state = "running";
  });
  close = vi.fn(async () => {
    this.state = "closed";
  });
  constructor() {
    audioContextInstances.push(this);
  }
}

let app: OpenedProject | undefined;

beforeEach(() => {
  currentSample = 128;
  audioContextInstances = [];
  waveCreateSpy.mockClear();
  vi.stubGlobal("AudioContext", MockAudioContext as unknown as typeof AudioContext);
});

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

function takeRows(): HTMLElement[] {
  const pane = screen.getByRole("heading", { name: "Takes" }).closest("section");
  if (!pane) throw new Error("takes pane not found");
  return within(pane).getAllByRole("listitem");
}

function audioForTake(takeId: string): HTMLAudioElement {
  const el = document.querySelector(`audio[src="${api.takeAudioUrl(PROJECT_ID, takeId)}"]`);
  if (!el) throw new Error(`no <audio> for ${takeId}`);
  return el as HTMLAudioElement;
}

function meterFillFor(takeId: string): HTMLElement {
  const audio = audioForTake(takeId);
  const wrapper = audio.closest(".take-audio-player");
  if (!wrapper) throw new Error(`no take-audio-player wrapper for ${takeId}`);
  const fill = wrapper.querySelector(".loudness-meter-fill");
  if (!fill) throw new Error(`no loudness-meter-fill for ${takeId}`);
  return fill as HTMLElement;
}

describe("Take quality score (SPEC.md sec 10)", () => {
  it("renders a take's score, or an em dash when it has none", async () => {
    const takes = makeTakes(2).map((t, i) => (i === 0 ? { ...t, score: 88 } : t));
    app = await renderOpenedProject({ takes });
    const [scored, unscored] = takeRows();
    expect(within(scored).getByText("score 88")).toBeTruthy();
    expect(within(unscored).getByText("score —")).toBeTruthy();
  });
});

describe("LRC download (SPEC.md sec 10)", () => {
  it("shows the lyrics (.lrc) link only for takes with has_lrc, pointing at the lrc endpoint", async () => {
    const takes = makeTakes(2).map((t, i) => (i === 0 ? { ...t, has_lrc: true } : t));
    app = await renderOpenedProject({ takes });
    const [withLrc, withoutLrc] = takeRows();

    const lrcLink = within(withLrc).getByRole("link", { name: "lyrics (.lrc)" }) as HTMLAnchorElement;
    expect(lrcLink.getAttribute("href")).toBe(api.takeLrcUrl(PROJECT_ID, takes[0].id));
    expect(lrcLink.getAttribute("download")).toBe(`${takes[0].id}.lrc`);

    expect(within(withoutLrc).queryByRole("link", { name: "lyrics (.lrc)" })).toBeNull();
  });
});

describe("WAV playback and download target the right take (SPEC.md sec 10)", () => {
  it("points each take's own audio element and download link at that take's audio endpoint", async () => {
    app = await renderOpenedProject();
    const [row1, row2] = takeRows();

    expect(audioForTake("take-01").getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-01"));
    const download1 = within(row1).getByRole("link", { name: "download" }) as HTMLAnchorElement;
    expect(download1.getAttribute("href")).toBe(api.takeAudioUrl(PROJECT_ID, "take-01"));
    expect(download1.getAttribute("download")).toBe("take-01.wav");

    expect(audioForTake("take-02").getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-02"));
    const download2 = within(row2).getByRole("link", { name: "download" }) as HTMLAnchorElement;
    expect(download2.getAttribute("href")).toBe(api.takeAudioUrl(PROJECT_ID, "take-02"));
    expect(download2.getAttribute("download")).toBe("take-02.wav");
  });

  it("rebuilds the shared waveform against whichever take is currently selected", async () => {
    app = await renderOpenedProject();
    const rows = takeRows();

    fireEvent.click(rows[0]); // select take-01
    await waitFor(() => expect(waveCreateSpy).toHaveBeenCalledTimes(1));
    expect(waveCreateSpy.mock.calls[0][0]).toMatchObject({
      url: api.takeAudioUrl(PROJECT_ID, "take-01"),
    });

    fireEvent.click(rows[2]); // select take-03
    await waitFor(() => expect(waveCreateSpy).toHaveBeenCalledTimes(2));
    expect(waveCreateSpy.mock.calls[1][0]).toMatchObject({
      url: api.takeAudioUrl(PROJECT_ID, "take-03"),
    });
  });
});

describe("Loudness meter lifecycle (SPEC.md sec 12 Phase 6)", () => {
  it("builds a Web Audio graph lazily on first play and tracks the live peak from the analyser", async () => {
    app = await renderOpenedProject();
    expect(audioContextInstances).toHaveLength(0);
    expect(meterFillFor("take-01").style.width).toBe("0%");

    // requestAnimationFrame must be faked before the first play -- the
    // meter's tick loop schedules its very first frame from the 'play'
    // handler itself, and a fake clock installed after that call can't see
    // a frame already queued against the real one.
    vi.useFakeTimers({ toFake: ["requestAnimationFrame", "cancelAnimationFrame"] });
    const audio = audioForTake("take-01");
    fireEvent.play(audio);

    expect(audioContextInstances).toHaveLength(1);
    const ctx = audioContextInstances[0];
    expect(ctx.createMediaElementSource).toHaveBeenCalledWith(audio);
    expect(ctx.createAnalyser).toHaveBeenCalledTimes(1);

    // Drive the mocked analyser to a known peak (|64 - 128| / 128 == 50%)
    // and advance one animation frame to pick it up.
    currentSample = 64;
    await act(async () => {
      vi.advanceTimersByTime(50);
    });
    vi.useRealTimers();
    expect(meterFillFor("take-01").style.width).toBe("50%");

    fireEvent.pause(audio);
    expect(meterFillFor("take-01").style.width).toBe("0%");
    expect(ctx.close).not.toHaveBeenCalled(); // pausing stops the loop, not the graph
  });

  it("does not rebuild the graph on repeated plays of the same take", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");

    fireEvent.play(audio);
    fireEvent.pause(audio);
    fireEvent.play(audio);

    expect(audioContextInstances).toHaveLength(1);
  });

  it("closes the audio graph when the workspace unmounts", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");
    fireEvent.play(audio);
    const ctx = audioContextInstances[0];
    expect(ctx.close).not.toHaveBeenCalled();

    app.cleanup();
    app = undefined;

    expect(ctx.close).toHaveBeenCalledTimes(1);
  });

  it("closes a take's audio graph once its row is no longer rendered after switching to a different project", async () => {
    // Deliberately bypasses renderOpenedProject to control a second
    // project's detail response -- the shared mock server (mockServer.ts)
    // always answers GET /api/projects/:id with its single `state.detail`
    // regardless of id, so a second, distinctly-keyed project's takes are
    // faked locally here rather than by editing the shared file.
    const server = createMockBardServer();
    const secondTakes = [makeTake("proj2-take-1", 1)];
    const secondDetail = makeProjectDetail(secondTakes);
    secondDetail.project.id = "proj-2";
    secondDetail.project.title = "Second Song";
    server.state.projects = [
      ...server.state.projects,
      makeProjectSummary({ id: "proj-2", title: "Second Song" }),
    ];
    server.install();
    const previousFetch = globalThis.fetch;
    vi.stubGlobal("fetch", (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (method === "GET" && url === "/api/projects/proj-2") {
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: "OK",
          json: async () => secondDetail,
          text: async () => JSON.stringify(secondDetail),
        } as Response);
      }
      return previousFetch(input, init);
    });

    const rendered = render(<App />);
    const firstProjectRow = (await screen.findByText("Test Song")).closest("li")!;
    fireEvent.click(within(firstProjectRow).getByRole("button", { name: "Open" }));
    await waitFor(() => expect(screen.getAllByTitle(LORA_SOURCE_TITLE).length).toBeGreaterThan(0));

    const audio = audioForTake("take-01");
    fireEvent.play(audio);
    expect(audioContextInstances).toHaveLength(1);
    const ctx = audioContextInstances[0];
    expect(ctx.close).not.toHaveBeenCalled();

    const secondProjectRow = screen.getByText("Second Song").closest("li")!;
    fireEvent.click(within(secondProjectRow).getByRole("button", { name: "Open" }));

    await waitFor(() =>
      expect(screen.getByRole("heading", { level: 2, name: "Second Song" })).toBeTruthy(),
    );
    expect(ctx.close).toHaveBeenCalledTimes(1);

    rendered.unmount();
    server.uninstall();
  });
});

describe("Failed takes cannot become playable sources (SPEC.md sec 10)", () => {
  it("renders no audio/download/lrc controls for an errored take, and never builds a meter for it", async () => {
    const takes = makeTakes(2).map((t, i) =>
      i === 0 ? { ...t, error: "generation crashed", has_lrc: true } : t,
    );
    app = await renderOpenedProject({ takes });
    const [failedRow, okRow] = takeRows();

    expect(within(failedRow).getByText("failed: generation crashed")).toBeTruthy();
    expect(failedRow.querySelector("audio")).toBeNull();
    expect(within(failedRow).queryByRole("link", { name: "download" })).toBeNull();
    expect(within(failedRow).queryByRole("link", { name: "lyrics (.lrc)" })).toBeNull();
    expect(okRow.querySelector("audio")).toBeTruthy();

    expect(audioContextInstances).toHaveLength(0);

    // Selecting the failed take and hitting the play shortcut must stay
    // inert -- there is no <audio> for it to act on, and no graph to build.
    fireEvent.click(failedRow);
    fireEvent.keyDown(window, { key: " " });
    expect(audioContextInstances).toHaveLength(0);
  });
});
