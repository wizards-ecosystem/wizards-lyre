// Regression tests for the Takes pane's score display and per-take media
// links (SPEC.md sec 8/10): the "score N" / "score n/a" fallback, the
// download link's href/filename, and the conditional "lyrics (.lrc)" link
// that only appears when a take has an LRC file. Everything runs against
// the mocked fetch backend in src/test/mockServer.ts -- no FastAPI, CUDA,
// ACE-Step, credentials, or generated audio.
import { cleanup, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { makeTake, PROJECT_ID } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

// jsdom has no canvas/layout for wavesurfer.js, so the waveform stack is
// stubbed out (App renders it unconditionally once a take is selected).
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

describe("Takes pane score and media links", () => {
  it("renders a take's numeric score, and an 'n/a' fallback when score is null", async () => {
    const takes = [makeTake("take-01", 1, { score: 87 }), makeTake("take-02", 2, { score: null })];
    app = await renderOpenedProject({ takes });
    const [firstRow, secondRow] = takeRows();

    expect(within(firstRow).getByText("score 87")).toBeTruthy();
    expect(within(secondRow).getByText("score n/a")).toBeTruthy();
  });

  it("shows the 'lyrics (.lrc)' link only for takes with has_lrc: true", async () => {
    const takes = [
      makeTake("take-01", 1, { has_lrc: true }),
      makeTake("take-02", 2, { has_lrc: false }),
    ];
    app = await renderOpenedProject({ takes });
    const [firstRow, secondRow] = takeRows();

    const lrcLink = within(firstRow).getByRole("link", {
      name: "lyrics (.lrc)",
    }) as HTMLAnchorElement;
    expect(lrcLink.getAttribute("href")).toBe(api.takeLrcUrl(PROJECT_ID, "take-01"));
    expect(lrcLink.getAttribute("download")).toBe("take-01.lrc");

    expect(within(secondRow).queryByRole("link", { name: "lyrics (.lrc)" })).toBeNull();
    expect(within(secondRow).queryByText("lyrics (.lrc)")).toBeNull();
  });

  it("gives every non-errored take a download link with the correct href/filename", async () => {
    const takes = [makeTake("take-01", 1, {}), makeTake("take-02", 2, {})];
    app = await renderOpenedProject({ takes });
    const [firstRow, secondRow] = takeRows();

    const firstDownload = within(firstRow).getByRole("link", {
      name: "download",
    }) as HTMLAnchorElement;
    expect(firstDownload.getAttribute("href")).toBe(api.takeAudioUrl(PROJECT_ID, "take-01"));
    expect(firstDownload.getAttribute("download")).toBe("take-01.wav");

    const secondDownload = within(secondRow).getByRole("link", {
      name: "download",
    }) as HTMLAnchorElement;
    expect(secondDownload.getAttribute("href")).toBe(api.takeAudioUrl(PROJECT_ID, "take-02"));
    expect(secondDownload.getAttribute("download")).toBe("take-02.wav");
  });

  it("shows only the failure message -- no player, download, or LRC link -- for an errored take", async () => {
    const takes = [
      makeTake("take-01", 1, { error: "generation failed: OOM", has_lrc: true, score: 42 }),
    ];
    app = await renderOpenedProject({ takes });
    const [row] = takeRows();

    expect(within(row).getByText("failed: generation failed: OOM")).toBeTruthy();
    expect(within(row).queryByRole("link", { name: "download" })).toBeNull();
    expect(within(row).queryByRole("link", { name: "lyrics (.lrc)" })).toBeNull();
    expect(within(row).queryByRole("audio")).toBeNull();
    expect(row.querySelector("audio")).toBeNull();
  });
});
