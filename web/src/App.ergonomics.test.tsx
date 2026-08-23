// Regression tests for the shipped Phase 5 interaction workflows (SPEC.md
// sec 12 Phase 5): A/B compare, the g/space/arrow/ctrl+s keyboard
// shortcuts (and their text-entry focus guard), and restoring an ancestor
// take through parent_take_id via the active-take endpoint. This does not
// duplicate the take favorite/notes/set-active job -- that one owns take
// metadata and direct "Set active" clicks in general; this file's "Set
// active" coverage is narrowly about the parent-chain restore path (follow
// a take's "from <parent>" link, then activate the ancestor) and asserting
// history survives it.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with the
// wavesurfer stack stubbed out like App.lora.test.tsx does (jsdom has no
// canvas/layout for the real library) plus two test-local mocks that live
// only in this file:
//   - HTMLMediaElement.prototype.play/pause/paused, since jsdom's real
//     implementation is a no-op stub that never flips `paused`.
//   - a fetch wrapper adding POST /api/projects/{id}/active_take, which the
//     shared mock backend does not implement.
import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { makeTakes, PROJECT_ID, type MockBardServer } from "./test/mockServer";
import { renderOpenedProject, type OpenedProject } from "./test/renderApp";

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

// jsdom's HTMLMediaElement.play()/pause() are "not implemented" stubs that
// never touch `paused` -- back them with a real per-element flag so the
// Space shortcut's `audio.paused ? audio.play() : audio.pause()` branch is
// actually exercised. No 'play'/'pause' DOM events are dispatched here on
// purpose: the LoudnessMeter (App.tsx) listens for exactly those events on
// the Takes-list <audio> elements and lazily builds a real Web Audio graph
// on first 'play', which jsdom doesn't implement -- dispatching a real
// event would crash the test for a feature this suite isn't testing.
beforeAll(() => {
  Object.defineProperty(HTMLMediaElement.prototype, "paused", {
    configurable: true,
    get(this: HTMLMediaElement & { __paused?: boolean }) {
      return this.__paused ?? true;
    },
  });
  HTMLMediaElement.prototype.play = vi.fn(function (this: HTMLMediaElement & { __paused?: boolean }) {
    this.__paused = false;
    return Promise.resolve();
  }) as unknown as () => Promise<void>;
  HTMLMediaElement.prototype.pause = vi.fn(function (this: HTMLMediaElement & { __paused?: boolean }) {
    this.__paused = true;
  }) as unknown as () => void;
});

// Adds POST /api/projects/{id}/active_take on top of whatever fetch
// server.install() already stubbed -- the shared mock backend
// (src/test/mockServer.ts) has no route for it. Wrapping rather than
// editing that shared file keeps this endpoint local to the one test (the
// parent-chain restore) that actually needs it.
function installActiveTakeEndpoint(server: MockBardServer): void {
  const previousFetch = globalThis.fetch;
  const patched: typeof fetch = (input, init) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    const match = url.match(/^\/api\/projects\/([^/]+)\/active_take$/);
    if (method === "POST" && match) {
      const projectId = match[1];
      const body = init?.body ? JSON.parse(String(init.body)) : {};
      const takeId = String(body.take_id ?? "");
      server.requests.push({ method, url, body });
      if (server.state.detail.project.id === projectId) {
        server.state.detail = {
          ...server.state.detail,
          project: { ...server.state.detail.project, active_take_id: takeId },
        };
      }
      server.state.projects = server.state.projects.map((p) =>
        p.id === projectId ? { ...p, active_take_id: takeId } : p,
      );
      const ok = true;
      return Promise.resolve({
        ok,
        status: 200,
        statusText: "OK",
        json: async () => server.state.detail.project,
        text: async () => JSON.stringify(server.state.detail.project),
      } as Response);
    }
    return previousFetch(input, init);
  };
  vi.stubGlobal("fetch", patched);
}

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

function audioForTake(takeId: string): HTMLAudioElement {
  const el = document.querySelector(`audio[src="${api.takeAudioUrl(PROJECT_ID, takeId)}"]`);
  if (!el) throw new Error(`no <audio> for ${takeId}`);
  return el as HTMLAudioElement;
}

function jobsPost(action?: string) {
  if (!app) throw new Error("app not rendered");
  return app.server.jobRequests(action);
}

function planPutRequests() {
  if (!app) throw new Error("app not rendered");
  return app.server.requests.filter((r) => r.method === "PUT" && r.url.endsWith("/plan"));
}

describe("A/B compare (SPEC.md sec 12 Phase 5)", () => {
  it("selects a comparison take and lets both sides play independently, without touching the active take", async () => {
    app = await renderOpenedProject(); // takes-01..10, active_take_id = take-01
    const rows = takeRows();
    fireEvent.click(rows[1]); // A: take-02
    fireEvent.click(within(rows[2]).getByRole("button", { name: "Compare" })); // B: take-03

    const comparePane = screen.getByRole("heading", { name: "Compare" }).closest("section")!;
    const [audioA, audioB] = Array.from(comparePane.querySelectorAll("audio"));
    expect(audioA.getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-02"));
    expect(audioB.getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-03"));

    // Playing/pausing either compared take is transient player UI state --
    // it must never call the active-take endpoint or any other write.
    fireEvent.play(audioA);
    fireEvent.play(audioB);
    fireEvent.pause(audioA);

    expect(takeRows()[0].className).toMatch(/active-take/); // still take-01
    expect(app.server.requests.some((r) => r.method !== "GET")).toBe(false);
  });

  it("swaps which take is A/B via Swap A/B without touching the active take", async () => {
    app = await renderOpenedProject();
    const rows = takeRows();
    fireEvent.click(rows[1]); // A: take-02
    fireEvent.click(within(rows[2]).getByRole("button", { name: "Compare" })); // B: take-03

    fireEvent.click(screen.getByRole("button", { name: "Swap A/B" }));

    // A/B swapped: take-03 is now selected (A), take-02 is now "Comparing".
    expect(takeRows()[2].className).toMatch(/selected/);
    expect(within(takeRows()[1]).getByRole("button", { name: "Comparing" })).toBeTruthy();
    const comparePane = screen.getByRole("heading", { name: "Compare" }).closest("section")!;
    const [audioA, audioB] = Array.from(comparePane.querySelectorAll("audio"));
    expect(audioA.getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-03"));
    expect(audioB.getAttribute("src")).toBe(api.takeAudioUrl(PROJECT_ID, "take-02"));

    expect(takeRows()[0].className).toMatch(/active-take/); // still take-01
    expect(app.server.requests.some((r) => r.method !== "GET")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "Close compare" }));
    expect(screen.queryByRole("heading", { name: "Compare" })).toBeNull();
  });
});

describe("Keyboard shortcuts (SPEC.md sec 12 Phase 5)", () => {
  it("'g' enqueues Generate when no text field is focused", async () => {
    app = await renderOpenedProject();
    fireEvent.keyDown(window, { key: "g" });
    await waitFor(() => expect(jobsPost("generate")).toHaveLength(1));
    expect(jobsPost("generate")[0].body).toEqual({ action: "generate", seed: -1 });
  });

  it("Space toggles play/pause for the selected take", async () => {
    app = await renderOpenedProject();
    fireEvent.click(takeRows()[0]); // select take-01
    const audio = audioForTake("take-01");
    expect(audio.paused).toBe(true);

    fireEvent.keyDown(window, { key: " " });
    expect(audio.paused).toBe(false);

    fireEvent.keyDown(window, { key: " " });
    expect(audio.paused).toBe(true);
  });

  it("ArrowUp/ArrowDown step through takes newest-first, clamped at both ends", async () => {
    app = await renderOpenedProject(); // 10 takes, take-01 (newest) .. take-10 (oldest)
    fireEvent.click(takeRows()[4]); // take-05
    expect(takeRows()[4].className).toMatch(/selected/);

    fireEvent.keyDown(window, { key: "ArrowDown" }); // -> take-06 (older)
    expect(takeRows()[5].className).toMatch(/selected/);

    fireEvent.keyDown(window, { key: "ArrowUp" });
    fireEvent.keyDown(window, { key: "ArrowUp" }); // -> take-04 (newer)
    expect(takeRows()[3].className).toMatch(/selected/);

    for (let i = 0; i < 10; i++) fireEvent.keyDown(window, { key: "ArrowUp" });
    expect(takeRows()[0].className).toMatch(/selected/); // clamped at newest

    for (let i = 0; i < 12; i++) fireEvent.keyDown(window, { key: "ArrowDown" });
    expect(takeRows()[9].className).toMatch(/selected/); // clamped at oldest
  });

  it("Ctrl+S / Cmd+S flushes a debounced plan save immediately, even while a text field is focused", async () => {
    app = await renderOpenedProject();
    const caption = screen.getByLabelText("Caption") as HTMLInputElement;
    caption.focus();
    fireEvent.change(caption, { target: { value: "new caption text" } });

    // Debounced (PLAN_SAVE_DEBOUNCE_MS = 500ms) -- nothing sent yet.
    expect(planPutRequests()).toHaveLength(0);

    // flushPendingPlanSave() clears the pending debounce timer synchronously,
    // before this handler's first await -- so the PUT that follows can only
    // be the immediate Ctrl+S path, never the (now-cancelled) debounce.
    fireEvent.keyDown(caption, { key: "s", ctrlKey: true });
    await waitFor(() => expect(planPutRequests()).toHaveLength(1));
    expect(planPutRequests()[0].body).toMatchObject({ caption: "new caption text" });

    // Cmd+S (metaKey) works the same way.
    fireEvent.change(caption, { target: { value: "edited again" } });
    fireEvent.keyDown(caption, { key: "s", metaKey: true });
    await waitFor(() => expect(planPutRequests()).toHaveLength(2));
  });

  it("guards g / space / arrow shortcuts while a text field is focused, but not Ctrl+S", async () => {
    app = await renderOpenedProject();
    fireEvent.click(takeRows()[4]); // take-05
    const audio = audioForTake("take-05");

    const caption = screen.getByLabelText("Caption") as HTMLInputElement;
    caption.focus();
    expect(document.activeElement).toBe(caption);

    fireEvent.keyDown(caption, { key: "g" });
    fireEvent.keyDown(caption, { key: " " });
    fireEvent.keyDown(caption, { key: "ArrowDown" });
    fireEvent.keyDown(caption, { key: "ArrowUp" });

    expect(jobsPost("generate")).toHaveLength(0);
    expect(audio.paused).toBe(true); // never played
    expect(takeRows()[4].className).toMatch(/selected/); // selection unchanged

    // Ctrl+S is the one shortcut deliberately exempt from the guard -- it's
    // what users need most while actually typing in caption/lyrics/query.
    fireEvent.change(caption, { target: { value: "typed while focused" } });
    fireEvent.keyDown(caption, { key: "s", ctrlKey: true });
    await waitFor(() => expect(planPutRequests()).toHaveLength(1));
  });
});

describe("Restoring an ancestor take (SPEC.md sec 12 Phase 5)", () => {
  it("follows parent_take_id to an ancestor and activates it via the active-take endpoint, without deleting history", async () => {
    // take-01 (newest, currently active) descends from take-03 three
    // generations back in the visible list.
    const takes = makeTakes(5).map((t, i) => (i === 0 ? { ...t, parent_take_id: "take-03" } : t));
    app = await renderOpenedProject({ takes });
    const server = app.server; // captured as a const so it stays narrowed inside closures below
    installActiveTakeEndpoint(server);

    expect(takeRows()[0].className).toMatch(/active-take/);
    expect(takeRows()[0].className).not.toMatch(/selected/);

    // Follow the "from take-03" link on the active (child) take -- this only
    // selects the ancestor for inspection, it does not activate it yet.
    fireEvent.click(within(takeRows()[0]).getByRole("button", { name: /from take-03/ }));
    expect(takeRows()[2].className).toMatch(/selected/);
    expect(server.requests.some((r) => r.url.endsWith("/active_take"))).toBe(false);

    // Now restore it: Set active on the now-selected ancestor.
    fireEvent.click(within(takeRows()[2]).getByRole("button", { name: "Set active" }));

    await waitFor(() => {
      expect(server.requests.some((r) => r.method === "POST" && r.url.endsWith("/active_take"))).toBe(
        true,
      );
    });
    const activeTakeCalls = server.requests.filter(
      (r) => r.method === "POST" && r.url.endsWith("/active_take"),
    );
    expect(activeTakeCalls).toHaveLength(1);
    expect(activeTakeCalls[0].body).toEqual({ take_id: "take-03" });

    await waitFor(() => expect(takeRows()[2].className).toMatch(/active-take/));
    expect(takeRows()[0].className).not.toMatch(/active-take/);

    // History survives the restore: every take, including the former active
    // (now-superseded) child, is still listed, and nothing was deleted.
    expect(takeRows()).toHaveLength(5);
    expect(screen.getByText("seed 1001")).toBeTruthy(); // take-01's fixture data, still present
    expect(audioForTake("take-01")).toBeTruthy();
    expect(server.requests.some((r) => r.method === "DELETE")).toBe(false);
  });
});
