// Regression tests for the "New project" form in the Library pane (SPEC.md
// sec 8 POST /api/projects), the one fundamental action nothing else in this
// suite exercises: every other App.*.test.tsx file opens the pre-seeded
// fixture project via renderOpenedProject()'s "Open" click and never drives
// createProject() itself.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with the
// wavesurfer stack stubbed out like the other App.*.test.tsx files (jsdom
// has no canvas/layout for the real library) even though these tests never
// select a take, since App.tsx statically imports the real package.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "./App";
import { createMockBardServer } from "./test/mockServer";

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

afterEach(() => {
  cleanup();
});

it("creates a project from a title and simple query, and switches to it", async () => {
  const server = createMockBardServer();
  server.install();

  render(<App />);

  // Library loads with just the fixture project until the form is used.
  await screen.findByText("Test Song");

  fireEvent.change(screen.getByPlaceholderText("title"), {
    target: { value: "Goblin Ballad" },
  });
  fireEvent.change(screen.getByPlaceholderText("simple query (optional)"), {
    target: { value: "upbeat goblin tavern song" },
  });
  fireEvent.click(screen.getByRole("button", { name: "New project" }));

  // POSTs the exact body createProject() builds.
  await waitFor(() => {
    const posts = server.requests.filter(
      (r) => r.method === "POST" && r.url === "/api/projects",
    );
    expect(posts).toHaveLength(1);
    expect(posts[0].body).toEqual({
      title: "Goblin Ballad",
      query: "upbeat goblin tavern song",
    });
  });

  // The new project shows up in the library list alongside the fixture one,
  // and createProject() auto-switches the workspace to it (no manual "Open"
  // click) -- the heading proves the workspace opened against the *new*
  // project, not the fixture one. Both the library row and the workspace
  // heading render the same title text, hence findAllByText here.
  await waitFor(() => {
    expect(screen.getAllByText("Goblin Ballad").length).toBeGreaterThanOrEqual(2);
  });
  expect(screen.getByText("Test Song")).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Goblin Ballad" })).toBeTruthy();
  const queryInput = screen.getByLabelText("Simple query") as HTMLInputElement;
  expect(queryInput.value).toBe("upbeat goblin tavern song");

  // The title input is cleared after a successful create.
  expect((screen.getByPlaceholderText("title") as HTMLInputElement).value).toBe("");

  server.uninstall();
});

it("creates a project with a fallback title when the title field is left blank", async () => {
  const server = createMockBardServer();
  server.install();

  render(<App />);
  await screen.findByText("Test Song");

  // Title left blank; only a query is provided.
  fireEvent.change(screen.getByPlaceholderText("simple query (optional)"), {
    target: { value: "mock backend created it" },
  });
  fireEvent.click(screen.getByRole("button", { name: "New project" }));

  await waitFor(() => {
    const posts = server.requests.filter(
      (r) => r.method === "POST" && r.url === "/api/projects",
    );
    expect(posts).toHaveLength(1);
    // App.tsx's createProject() substitutes "Untitled" client-side when the
    // title field is empty (matching server/storage.py::create_project's own
    // `title or "Untitled"` default), so the mock server -- like the real
    // backend -- never actually sees an empty title.
    expect(posts[0].body).toEqual({ title: "Untitled", query: "mock backend created it" });
  });

  await waitFor(() => {
    expect(screen.getByRole("heading", { name: "Untitled" })).toBeTruthy();
  });

  server.uninstall();
});

it("opens the new project's workspace with an empty takes pane and no Open click needed", async () => {
  const server = createMockBardServer();
  server.install();

  render(<App />);
  await screen.findByText("Test Song");

  fireEvent.change(screen.getByPlaceholderText("title"), {
    target: { value: "Fresh Track" },
  });
  fireEvent.click(screen.getByRole("button", { name: "New project" }));

  // The workspace pane (Plan/Takes) renders for the new project immediately
  // -- the "Select or create a project." hint that shows when no project is
  // active must be gone.
  await waitFor(() => {
    expect(screen.getByRole("heading", { name: "Fresh Track" })).toBeTruthy();
  });
  expect(screen.queryByText("Select or create a project.")).toBeNull();
  expect(screen.getByRole("heading", { name: "Plan" })).toBeTruthy();

  server.uninstall();
});
