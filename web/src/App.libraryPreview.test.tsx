// Regression tests for the Library pane's inline take preview (SPEC.md sec
// 9.1 "Play last take inline (optional)"): a project's last (active) take
// can be previewed straight from the project list, jukebox-style, without
// opening the project. Runs against the mocked fetch backend in
// src/test/mockServer.ts, seeding a second ProjectSummary directly into
// server.state.projects before install() so the list has both a project
// with an active take and one without. No wavesurfer stubbing is needed --
// these tests never open a project's workspace, so the Takes pane (and its
// waveform) never mounts.
//
// jsdom's HTMLMediaElement.prototype.play() is a "not implemented" stub
// that throws/rejects, and .pause() is a silent no-op; both are spied on
// here (restored in afterEach) so togglePreview()'s audio.play()/
// audio.pause() calls don't crash the test.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import { createMockBardServer, makeProjectSummary } from "./test/mockServer";

let playSpy: ReturnType<typeof vi.spyOn> | undefined;
let pauseSpy: ReturnType<typeof vi.spyOn> | undefined;

afterEach(() => {
  cleanup();
  playSpy?.mockRestore();
  pauseSpy?.mockRestore();
  playSpy = undefined;
  pauseSpy = undefined;
});

function projectRow(title: string): HTMLElement {
  const titleEl = screen.getByText(title, { selector: ".project-title" });
  const row = titleEl.closest("li");
  if (!row) throw new Error(`project row for "${title}" not found`);
  return row as HTMLElement;
}

function previewBtn(title: string): HTMLButtonElement {
  const btn = projectRow(title).querySelector(".preview-btn");
  if (!btn) throw new Error(`preview button for "${title}" not found`);
  return btn as HTMLButtonElement;
}

describe("Library inline take preview", () => {
  it("disables the preview button with a 'No takes yet' title when a project has no active take", async () => {
    const server = createMockBardServer();
    server.state.projects = [
      makeProjectSummary({ id: "proj-1", title: "Has A Take", active_take_id: "take-01" }),
      makeProjectSummary({ id: "proj-2", title: "No Takes", active_take_id: null }),
    ];
    server.install();

    render(<App />);
    await screen.findByText("Has A Take", { selector: ".project-title" });

    const btn = previewBtn("No Takes");
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("No takes yet");
    // The project with an active take stays enabled and unlit.
    const otherBtn = previewBtn("Has A Take");
    expect(otherBtn.disabled).toBe(false);
    expect(otherBtn.title).toBe("Play last take");
    expect(otherBtn.getAttribute("aria-label")).toBe("Play Has A Take preview");

    server.uninstall();
  });

  it("plays, pauses, jukebox-swaps between projects, and resets on the audio 'ended' event", async () => {
    playSpy = vi
      .spyOn(window.HTMLMediaElement.prototype, "play")
      .mockImplementation(() => Promise.resolve());
    pauseSpy = vi
      .spyOn(window.HTMLMediaElement.prototype, "pause")
      .mockImplementation(() => {});

    const server = createMockBardServer();
    server.state.projects = [
      makeProjectSummary({ id: "proj-1", title: "Song One", active_take_id: "take-01" }),
      makeProjectSummary({ id: "proj-2", title: "Song Two", active_take_id: "take-02" }),
    ];
    server.install();

    render(<App />);
    await screen.findByText("Song One", { selector: ".project-title" });

    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio).toBeTruthy();

    // Clicking project one's preview button starts playback.
    fireEvent.click(previewBtn("Song One"));
    expect(playSpy).toHaveBeenCalledTimes(1);
    expect(audio.src).toBe(new URL(api.takeAudioUrl("proj-1", "take-01"), window.location.href).href);
    expect(previewBtn("Song One").getAttribute("aria-label")).toBe("Pause Song One preview");
    expect(previewBtn("Song One").title).toBe("Pause preview");
    expect(previewBtn("Song Two").getAttribute("aria-label")).toBe("Play Song Two preview");

    // Clicking the same project's button again pauses and reverts the glyph.
    fireEvent.click(previewBtn("Song One"));
    expect(pauseSpy).toHaveBeenCalledTimes(1);
    expect(previewBtn("Song One").getAttribute("aria-label")).toBe("Play Song One preview");
    expect(previewBtn("Song One").title).toBe("Play last take");

    // Start project one again, then jukebox-swap to project two while it's
    // still "playing" -- no explicit pause() needed for the swap itself,
    // just a new play() against the swapped src.
    fireEvent.click(previewBtn("Song One"));
    expect(playSpy).toHaveBeenCalledTimes(2);
    fireEvent.click(previewBtn("Song Two"));
    expect(playSpy).toHaveBeenCalledTimes(3);
    expect(pauseSpy).toHaveBeenCalledTimes(1);
    expect(audio.src).toBe(new URL(api.takeAudioUrl("proj-2", "take-02"), window.location.href).href);
    expect(previewBtn("Song One").getAttribute("aria-label")).toBe("Play Song One preview");
    expect(previewBtn("Song Two").getAttribute("aria-label")).toBe("Pause Song Two preview");
    expect(previewBtn("Song Two").title).toBe("Pause preview");

    // The shared <audio> element's native 'ended' event resets whichever
    // row was playing back to the play glyph.
    fireEvent(audio, new Event("ended"));
    expect(previewBtn("Song Two").getAttribute("aria-label")).toBe("Play Song Two preview");
    expect(previewBtn("Song Two").title).toBe("Play last take");

    server.uninstall();
  });
});
