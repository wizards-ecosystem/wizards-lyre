// Regression tests for LoudnessMeter (App.tsx, SPEC.md sec 12 Phase 6): the
// per-take live peak meter built directly on the Web Audio API. jsdom has no
// real Web Audio implementation, so this file hand-builds a minimal
// AudioContext/AnalyserNode mock (installed via vi.stubGlobal, restored in
// afterEach -- matching how src/test/renderApp.tsx restores window.confirm)
// plus a controllable requestAnimationFrame/cancelAnimationFrame stub that
// only advances the render loop when a test explicitly asks it to, via
// flushRaf(). This does not duplicate App.ergonomics.test.tsx, which
// deliberately avoids firing real play/pause DOM events on take <audio>
// elements for exactly this reason (LoudnessMeter would try to build a real
// Web Audio graph jsdom doesn't have) -- this file is the other half, with
// the mock that makes exercising that graph safe.
import { act, cleanup, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { PROJECT_ID } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => {}, destroy: () => {} }),
  },
}));

vi.mock("wavesurfer.js/plugins/regions", () => ({
  default: {
    create: () => ({
      on: () => {},
      getRegions: () => [],
      enableDragSelection: () => {},
      addRegion: () => {},
      clearRegions: () => {},
    }),
  },
}));

// ---------------------------------------------------------------------------
// Minimal Web Audio mock -- just enough surface for LoudnessMeter's graph:
// ctx.createMediaElementSource(el).connect(analyser),
// analyser.connect(ctx.destination), analyser.fftSize, and
// analyser.getByteTimeDomainData(buffer), which each test points at a fixed
// fill value to drive a specific peak percentage (the real component
// computes max(|sample-128|/128) across the buffer -- 0 or 255 -> ~100%,
// 128 -> 0%).
// ---------------------------------------------------------------------------
let fillValue = 128;

function makeAnalyserMock() {
  return {
    fftSize: 2048,
    connect: vi.fn(),
    getByteTimeDomainData: vi.fn((buffer: Uint8Array) => {
      buffer.fill(fillValue);
    }),
  };
}

interface MockAudioContextInstance {
  state: string;
  resume: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  createMediaElementSource: ReturnType<typeof vi.fn>;
  createAnalyser: ReturnType<typeof vi.fn>;
}

let audioContextCtor: ReturnType<typeof vi.fn>;

// requestAnimationFrame stub: queues callbacks instead of firing them on a
// real timer, so a test controls exactly when (and how many times) the
// render loop advances by calling flushRaf() -- otherwise a real jsdom RAF
// would let LoudnessMeter's self-rescheduling tick() loop spin uncontrolled
// for the life of the test process.
let rafQueue: Map<number, FrameRequestCallback>;
let rafIdSeq: number;
let cancelRafSpy: ReturnType<typeof vi.fn>;

function flushRaf() {
  // The queued callback calls setPeak() directly (not from inside a React
  // event handler), so it must be wrapped in act() for the resulting
  // re-render to be flushed before the test's next assertion.
  act(() => {
    const pending = Array.from(rafQueue.entries());
    rafQueue.clear();
    for (const [, cb] of pending) cb(0);
  });
}

function installWebAudioMocks() {
  fillValue = 128;
  rafQueue = new Map();
  rafIdSeq = 0;

  audioContextCtor = vi.fn(function (this: MockAudioContextInstance) {
    this.state = "running";
    this.resume = vi.fn();
    this.close = vi.fn(() => Promise.resolve());
    this.createMediaElementSource = vi.fn(() => ({ connect: vi.fn() }));
    this.createAnalyser = vi.fn(() => makeAnalyserMock());
  });
  vi.stubGlobal("AudioContext", audioContextCtor);

  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((cb: FrameRequestCallback) => {
      rafIdSeq += 1;
      rafQueue.set(rafIdSeq, cb);
      return rafIdSeq;
    }),
  );
  cancelRafSpy = vi.fn((id: number) => {
    rafQueue.delete(id);
  });
  vi.stubGlobal("cancelAnimationFrame", cancelRafSpy);
}

beforeEach(() => {
  installWebAudioMocks();
});

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function audioForTake(takeId: string): HTMLAudioElement {
  const el = document.querySelector(`audio[src="${api.takeAudioUrl(PROJECT_ID, takeId)}"]`);
  if (!el) throw new Error(`no <audio> for ${takeId}`);
  return el as HTMLAudioElement;
}

function meterFillFor(audio: HTMLAudioElement): HTMLElement {
  const wrapper = audio.closest(".take-audio-player");
  if (!wrapper) throw new Error("no take-audio-player wrapper");
  const fill = wrapper.querySelector(".loudness-meter-fill");
  if (!fill) throw new Error("no loudness-meter-fill");
  return fill as HTMLElement;
}

function analyserFor(ctxCallIndex: number) {
  const instance = audioContextCtor.mock.instances[
    ctxCallIndex
  ] as unknown as MockAudioContextInstance;
  const result = instance.createAnalyser.mock.results[0];
  if (!result) throw new Error(`AudioContext #${ctxCallIndex} never called createAnalyser`);
  return result.value as ReturnType<typeof makeAnalyserMock>;
}

describe("LoudnessMeter lifecycle (SPEC.md sec 12 Phase 6)", () => {
  it("renders a 0% fill before any playback, without touching Web Audio", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");
    expect(meterFillFor(audio).style.width).toBe("0%");
    expect(audioContextCtor).not.toHaveBeenCalled();
  });

  it("builds the Web Audio graph lazily on first play and renders the analyser's peak", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");

    fillValue = 0; // max(|0-128|/128) == 1.0 -> 100% peak
    fireEvent.play(audio);
    expect(audioContextCtor).toHaveBeenCalledTimes(1);

    const instance = audioContextCtor.mock.instances[0] as unknown as MockAudioContextInstance;
    expect(instance.createMediaElementSource).toHaveBeenCalledTimes(1);
    expect(instance.createAnalyser).toHaveBeenCalledTimes(1);
    const analyser = analyserFor(0);
    expect(analyser.fftSize).toBe(256); // set by LoudnessMeter itself
    expect(analyser.connect).toHaveBeenCalledTimes(1); // analyser -> ctx.destination

    // Nothing rendered yet -- the peak only updates once a frame runs.
    expect(meterFillFor(audio).style.width).toBe("0%");

    flushRaf();
    expect(analyser.getByteTimeDomainData).toHaveBeenCalledTimes(1);
    expect(meterFillFor(audio).style.width).toBe("100%");
  });

  it("resets the fill to 0% and stops requesting frames on pause", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");

    fillValue = 0;
    fireEvent.play(audio);
    flushRaf();
    expect(meterFillFor(audio).style.width).toBe("100%");
    const analyser = analyserFor(0);
    const callsBeforePause = analyser.getByteTimeDomainData.mock.calls.length;

    fireEvent.pause(audio);
    expect(meterFillFor(audio).style.width).toBe("0%");
    expect(cancelRafSpy).toHaveBeenCalled();

    // The stopped loop must not still be queued -- a forced tick produces no
    // further analyser reads.
    flushRaf();
    expect(analyser.getByteTimeDomainData).toHaveBeenCalledTimes(callsBeforePause);

    // The AudioContext itself is not torn down on pause -- only on unmount
    // /audio-element change -- so a second play reuses the same graph rather
    // than constructing a new one.
    fireEvent.play(audio);
    expect(audioContextCtor).toHaveBeenCalledTimes(1);
  });

  it("also resets the fill to 0% and stops frames on ended", async () => {
    app = await renderOpenedProject();
    const audio = audioForTake("take-01");

    fillValue = 0;
    fireEvent.play(audio);
    flushRaf();
    expect(meterFillFor(audio).style.width).toBe("100%");

    fireEvent.ended(audio);
    expect(meterFillFor(audio).style.width).toBe("0%");
    expect(cancelRafSpy).toHaveBeenCalled();
  });

  it("gives a second take its own independent AudioContext, without reusing or disturbing the first", async () => {
    app = await renderOpenedProject();
    const audioA = audioForTake("take-01");
    const audioB = audioForTake("take-02");

    fillValue = 0;
    fireEvent.play(audioA);
    flushRaf();
    expect(audioContextCtor).toHaveBeenCalledTimes(1);
    expect(meterFillFor(audioA).style.width).toBe("100%");
    fireEvent.pause(audioA);
    expect(meterFillFor(audioA).style.width).toBe("0%");
    const analyserA = analyserFor(0);
    const callsOnAAfterPause = analyserA.getByteTimeDomainData.mock.calls.length;

    // Second take: a fresh AudioContext, not the first take's.
    fillValue = 255; // max(|255-128|/128) == 127/128 -> 99% (rounded)
    fireEvent.play(audioB);
    expect(audioContextCtor).toHaveBeenCalledTimes(2);
    const instanceB = audioContextCtor.mock.instances[1] as unknown as MockAudioContextInstance;
    expect(instanceB).not.toBe(audioContextCtor.mock.instances[0]);

    flushRaf();
    expect(meterFillFor(audioB).style.width).toBe("99%");

    // The first take's already-stopped graph was not touched by the second
    // take's frame.
    expect(analyserA.getByteTimeDomainData).toHaveBeenCalledTimes(callsOnAAfterPause);
    expect(meterFillFor(audioA).style.width).toBe("0%");
  });
});
