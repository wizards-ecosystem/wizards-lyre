import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { makeTakes } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

const waveform = vi.hoisted(() => ({
  playPause: vi.fn(() => Promise.resolve()),
}));

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({
      on: () => {},
      destroy: () => {},
      playPause: waveform.playPause,
    }),
  },
}));

vi.mock("wavesurfer.js/plugins/regions", () => ({
  default: {
    create: () => ({
      on: () => {},
      getRegions: () => [],
      enableDragSelection: () => {},
      addRegion: () => {},
    }),
  },
}));

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  waveform.playPause.mockClear();
  cleanup();
});

describe("Resonance Workbench interactions", () => {
  it("auto-selects the active take and routes Space to the selected waveform transport", async () => {
    app = await renderOpenedProject({ takes: makeTakes(2) });
    const active = screen.getByRole("listitem", { name: /seed 1001/ });
    expect(active.className).toMatch(/active-take/);
    expect(active.className).toMatch(/selected/);

    fireEvent.keyDown(window, { key: " ", code: "Space" });
    expect(waveform.playPause).toHaveBeenCalledTimes(1);
  });

  it("exposes Takes and Style Packs as an accessible tab pair", async () => {
    app = await renderOpenedProject();
    const takes = screen.getByRole("tab", { name: /Takes/ });
    const styles = screen.getByRole("tab", { name: /Style packs/ });
    expect(takes.getAttribute("aria-selected")).toBe("true");

    fireEvent.click(styles);
    expect(styles.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("button", { name: "Train style pack" })).toBeTruthy();

    fireEvent.click(takes);
    expect(takes.getAttribute("aria-selected")).toBe("true");
  });

  it("announces plan autosave progress and completion", async () => {
    app = await renderOpenedProject();
    fireEvent.change(screen.getByPlaceholderText("Describe the song you want to explore..."), {
      target: { value: "a restrained chamber pulse" },
    });
    expect(screen.getByText("Saving...")).toBeTruthy();
    await screen.findByText("Saved");
  });

  it("uploads through the keyboard-accessible file picker", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    const file = new File(["RIFF....WAVEfmt "], "picker-source.wav", {
      type: "audio/wav",
    });
    fireEvent.change(screen.getByLabelText("Choose audio source"), {
      target: { files: [file] },
    });

    await screen.findByText("picker-source.wav");
    expect(
      app.server.requests.some(
        (request) => request.method === "POST" && request.url.endsWith("/uploads"),
      ),
    ).toBe(true);
    expect(screen.getByRole("listitem", { name: /seed 1001/ }).className).not.toMatch(/selected/);
  });

  it("commits project-title edits with Enter and restores edits with Escape", async () => {
    app = await renderOpenedProject();
    fireEvent.click(screen.getByTitle("Edit project title"));
    const input = screen.getByLabelText("Project title") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Resonant Study" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(
        app!.server.requests.some(
          (request) =>
            request.method === "PATCH" &&
            request.url === "/api/projects/proj-1" &&
            (request.body as { title?: string }).title === "Resonant Study",
        ),
      ).toBe(true);
    });
    await screen.findByRole("heading", { name: "Resonant Study" });

    fireEvent.click(screen.getByTitle("Edit project title"));
    const secondInput = screen.getByLabelText("Project title") as HTMLInputElement;
    fireEvent.change(secondInput, { target: { value: "Discard this" } });
    fireEvent.keyDown(secondInput, { key: "Escape" });
    expect(screen.getByRole("heading", { name: "Resonant Study" })).toBeTruthy();
  });

  it("closes confirmation with Escape and restores focus to its trigger", async () => {
    app = await renderOpenedProject();
    const library = screen.getByRole("complementary", { name: "Project library" });
    const trigger = within(library).getByRole("button", { name: "Delete Test Song" });
    trigger.focus();
    fireEvent.click(trigger);
    expect(screen.getByRole("alertdialog", { name: /Delete.*Test Song/ })).toBeTruthy();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(trigger));
    expect(app.server.requests.some((request) => request.method === "DELETE")).toBe(false);
  });
});
