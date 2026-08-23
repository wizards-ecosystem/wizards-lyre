// Regression tests for the Library pane (SPEC.md sec 9.1): searching
// projects by title, toggling favorite (PATCH /api/projects/{id}), and
// deleting a project behind a window.confirm gate (DELETE
// /api/projects/{id}). Everything runs against the mocked fetch backend in
// src/test/mockServer.ts -- no FastAPI, CUDA, ACE-Step, credentials, or
// generated audio.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { ProjectSummary } from "./api";
import { createMockBardServer, makeProjectSummary, type MockBardServer } from "./test/mockServer";

interface LibraryApp {
  server: MockBardServer;
  confirm: ReturnType<typeof vi.fn>;
  cleanup: () => void;
}

// Renders the real <App/> against a mock server seeded with the given
// project list, without opening any of them -- the library pane loads its
// list on mount (App.tsx's `useEffect(() => { refreshProjects() }, [])`), so
// there's no need to click "Open" for these tests.
function renderLibrary(projects: ProjectSummary[]): LibraryApp {
  const server = createMockBardServer();
  server.state.projects = projects;
  server.install();

  const originalConfirm = window.confirm;
  const confirm = vi.fn(() => true);
  window.confirm = confirm as unknown as typeof window.confirm;

  const rendered = render(<App />);

  return {
    server,
    confirm,
    cleanup: () => {
      rendered.unmount();
      server.uninstall();
      window.confirm = originalConfirm;
    },
  };
}

function projectRow(title: string): HTMLElement {
  const row = screen.getByText(title).closest("li");
  if (!row) throw new Error(`project row not found for "${title}"`);
  return row as HTMLElement;
}

function listedTitles(): (string | null | undefined)[] {
  return screen
    .getAllByRole("listitem")
    .map((li) => li.querySelector(".project-title")?.textContent);
}

let app: LibraryApp | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

describe("Library pane (SPEC.md sec 9.1)", () => {
  it("filters the project list by title as the user types", async () => {
    app = renderLibrary([
      makeProjectSummary({ id: "proj-a", title: "Dark Wizard Folk" }),
      makeProjectSummary({ id: "proj-b", title: "Sunny Pop Song" }),
    ]);

    expect(await screen.findByText("Dark Wizard Folk")).toBeTruthy();
    expect(screen.getByText("Sunny Pop Song")).toBeTruthy();

    const search = screen.getByPlaceholderText("Search projects");
    fireEvent.change(search, { target: { value: "wizard" } });
    expect(screen.getByText("Dark Wizard Folk")).toBeTruthy();
    expect(screen.queryByText("Sunny Pop Song")).toBeNull();

    // Search is case-insensitive (App.tsx lower-cases both sides).
    fireEvent.change(search, { target: { value: "SUNNY" } });
    expect(screen.getByText("Sunny Pop Song")).toBeTruthy();
    expect(screen.queryByText("Dark Wizard Folk")).toBeNull();

    fireEvent.change(search, { target: { value: "" } });
    expect(screen.getByText("Dark Wizard Folk")).toBeTruthy();
    expect(screen.getByText("Sunny Pop Song")).toBeTruthy();
  });

  it("toggles favorite via PATCH and sorts favorited projects to the top", async () => {
    app = renderLibrary([
      makeProjectSummary({ id: "proj-a", title: "Alpha" }),
      makeProjectSummary({ id: "proj-b", title: "Beta" }),
    ]);
    await screen.findByText("Alpha");
    expect(listedTitles()).toEqual(["Alpha", "Beta"]);

    const favoriteBtn = within(projectRow("Beta")).getByTitle("Favorite");
    expect(favoriteBtn.textContent).toBe("☆");
    fireEvent.click(favoriteBtn);

    await waitFor(() => {
      const patches = app!.server.requests.filter(
        (r) => r.method === "PATCH" && r.url === "/api/projects/proj-b",
      );
      expect(patches).toHaveLength(1);
      expect(patches[0].body).toEqual({ favorite: true });
    });

    await waitFor(() => {
      const toggled = within(projectRow("Beta")).getByTitle("Unfavorite");
      expect(toggled.textContent).toBe("★");
    });

    // Favoriting Beta moves it above the still-unfavorited Alpha.
    await waitFor(() => {
      expect(listedTitles()).toEqual(["Beta", "Alpha"]);
    });

    // Clicking again unfavorites it and it drops back to the bottom.
    fireEvent.click(within(projectRow("Beta")).getByTitle("Unfavorite"));
    await waitFor(() => {
      const patches = app!.server.requests.filter(
        (r) => r.method === "PATCH" && r.url === "/api/projects/proj-b",
      );
      expect(patches).toHaveLength(2);
      expect(patches[1].body).toEqual({ favorite: false });
    });
    await waitFor(() => {
      expect(listedTitles()).toEqual(["Alpha", "Beta"]);
    });
  });

  it("deletes a project only after window.confirm is accepted", async () => {
    app = renderLibrary([
      makeProjectSummary({ id: "proj-a", title: "Alpha" }),
      makeProjectSummary({ id: "proj-b", title: "Beta" }),
    ]);
    await screen.findByText("Alpha");

    const deleteBtn = within(projectRow("Beta")).getByTitle("Delete project");

    // Declined: window.confirm is asked, but no DELETE fires and the
    // project stays listed.
    app.confirm.mockReturnValueOnce(false);
    fireEvent.click(deleteBtn);
    expect(app.confirm).toHaveBeenCalledWith(
      'Delete "Beta"? This permanently removes its takes and cannot be undone.',
    );
    expect(app.server.requests.some((r) => r.method === "DELETE")).toBe(false);
    expect(screen.getByText("Beta")).toBeTruthy();

    // Accepted (the helper's default confirm mock returns true): DELETE
    // fires and the project drops out of the rendered list.
    fireEvent.click(deleteBtn);
    await waitFor(() => {
      expect(
        app!.server.requests.some(
          (r) => r.method === "DELETE" && r.url === "/api/projects/proj-b",
        ),
      ).toBe(true);
    });
    await waitFor(() => {
      expect(screen.queryByText("Beta")).toBeNull();
    });
    expect(screen.getByText("Alpha")).toBeTruthy();
  });
});
