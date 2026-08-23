// Regression tests for the Takes pane's per-take controls (SPEC.md sec 8/10):
// favorite toggling, debounced notes saving, "Set active", and the export
// link's query string. Everything runs against the mocked fetch backend in
// src/test/mockServer.ts -- no FastAPI, CUDA, ACE-Step, credentials, or
// generated audio.
import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PROJECT_ID } from "./test/mockServer";
import { renderOpenedProject, type OpenedProject } from "./test/renderApp";

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

function patchTakeRequests() {
  if (!app) throw new Error("app not rendered");
  return app.server.requests.filter(
    (r) => r.method === "PATCH" && /\/takes\//.test(r.url),
  );
}

function activeTakeRequests() {
  if (!app) throw new Error("app not rendered");
  return app.server.requests.filter(
    (r) => r.method === "POST" && r.url.endsWith("/active_take"),
  );
}

describe("Takes pane controls", () => {
  it("toggles a take's favorite star and PATCHes the flip", async () => {
    app = await renderOpenedProject();
    const row = takeRows()[0];
    const favoriteBtn = within(row).getByTitle("Favorite");
    expect(within(row).getByText("☆")).toBeTruthy();

    fireEvent.click(favoriteBtn);

    expect(within(row).getByText("★")).toBeTruthy();
    expect(within(row).getByTitle("Unfavorite")).toBeTruthy();

    await waitFor(() => expect(patchTakeRequests()).toHaveLength(1));
    expect(patchTakeRequests()[0].url).toBe(`/api/projects/${PROJECT_ID}/takes/take-01`);
    expect(patchTakeRequests()[0].body).toEqual({ favorite: true });

    // Clicking again flips it back the other way.
    fireEvent.click(within(row).getByTitle("Unfavorite"));
    expect(within(row).getByText("☆")).toBeTruthy();
    await waitFor(() => expect(patchTakeRequests()).toHaveLength(2));
    expect(patchTakeRequests()[1].body).toEqual({ favorite: false });
  });

  it("debounce-saves typed notes for a take", async () => {
    app = await renderOpenedProject();
    const row = takeRows()[0];
    const notes = within(row).getByPlaceholderText("Notes...") as HTMLTextAreaElement;

    fireEvent.change(notes, { target: { value: "great take, keep the bridge" } });
    expect(notes.value).toBe("great take, keep the bridge");

    // Nothing sent yet -- the save is debounced.
    expect(patchTakeRequests()).toHaveLength(0);

    await waitFor(() => expect(patchTakeRequests()).toHaveLength(1));
    expect(patchTakeRequests()[0].url).toBe(`/api/projects/${PROJECT_ID}/takes/take-01`);
    expect(patchTakeRequests()[0].body).toEqual({ notes: "great take, keep the bridge" });
  });

  it("moves the active-take indicator when 'Set active' is clicked on another take", async () => {
    app = await renderOpenedProject();
    const [firstRow, secondRow] = takeRows();

    // take-01 starts active (renderOpenedProject's fixture default).
    expect(within(firstRow).getByText("active")).toBeTruthy();
    expect(within(secondRow).queryByText("active")).toBeNull();
    const setActiveOnSecond = within(secondRow).getByRole("button", { name: "Set active" });
    expect(setActiveOnSecond.hasAttribute("disabled")).toBe(false);

    fireEvent.click(setActiveOnSecond);

    await waitFor(() => expect(activeTakeRequests()).toHaveLength(1));
    expect(activeTakeRequests()[0].url).toBe(`/api/projects/${PROJECT_ID}/active_take`);
    expect(activeTakeRequests()[0].body).toEqual({ take_id: "take-02" });

    await waitFor(() => {
      const [newFirstRow, newSecondRow] = takeRows();
      expect(within(newFirstRow).queryByText("active")).toBeNull();
      expect(within(newSecondRow).getByText("active")).toBeTruthy();
    });
  });

  it("keeps the export link's include_stems query in sync with the checkbox", async () => {
    app = await renderOpenedProject();
    const exportLink = screen.getByRole("link", { name: "Export project (.zip)" }) as HTMLAnchorElement;
    expect(exportLink.getAttribute("href")).toBe(
      `/api/projects/${PROJECT_ID}/export?include_stems=true`,
    );
    expect(exportLink.getAttribute("download")).toBe("Test Song-export.zip");

    fireEvent.click(screen.getByLabelText("Include stems (extract / lego takes)"));

    expect(screen.getByRole("link", { name: "Export project (.zip)" }).getAttribute("href")).toBe(
      `/api/projects/${PROJECT_ID}/export?include_stems=false`,
    );
  });
});
