// Regression tests for the Custom plan editor (SPEC.md sec 9.2 Plan pane):
// every field maps onto the right PUT /plan JSON type (nulls for
// BPM/keyscale when cleared, numbers for BPM/duration, booleans for
// instrumental/caption_rewrite), edits survive a reopen of the project, the
// shared debounce collapses rapid keystrokes into one PUT, and a rejected
// PUT neither wipes the on-screen edit nor overwrites the last plan the
// server actually accepted.
//
// This does not duplicate App.sections.test.tsx (named-section rendering on
// the waveform) or the lyric structure-tag button coverage owned elsewhere
// -- this file only exercises the flat scalar Plan fields and the save
// pipeline itself.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with the
// wavesurfer stack stubbed out like the other App.*.test.tsx files (jsdom
// has no canvas/layout for the real library) even though these tests never
// select a take, since App.tsx statically imports the real package.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import App from "../App";
import type { RecordedRequest } from "../test/mockServer";
import { createMockBardServer, type MockBardServer } from "../test/mockServer";
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

let app: OpenedProject | undefined;

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function planPutRequests(server: MockBardServer): RecordedRequest[] {
  return server.requests.filter((r) => r.method === "PUT" && r.url.endsWith("/plan"));
}

// The Lyrics field's wrapping <label> also contains the structure-tag
// buttons (App.tsx), so it has more than one labelable descendant --
// getByLabelText("Lyrics") resolves to the *first* one (an "[Intro]"
// button), not the textarea, because testing-library's implicit label
// association picks the first labelable descendant. The Plan pane has
// exactly one <textarea> (the other one, "take-notes", lives in the Takes
// pane), so scoping a plain tag query to the Plan section finds it
// unambiguously.
function lyricsTextarea(): HTMLTextAreaElement {
  const pane = screen.getByRole("heading", { name: "Plan" }).closest("section");
  if (!pane) throw new Error("plan pane not found");
  const el = within(pane).getByLabelText("Lyrics");
  if (!el) throw new Error("lyrics textarea not found");
  return el as HTMLTextAreaElement;
}

// jsdom does not implement a real browser's default action of moving focus
// to a pointer-down target, so a plain fireEvent.click here would never
// reproduce the reviewer-flagged bug (a tag button's mousedown stealing
// focus from the textarea before onClick runs, which made insertLyricsTag
// always see activeElement !== textarea and append instead of honoring the
// cursor/selection). This reproduces that default action by hand, honoring
// preventDefault() exactly like a real browser would, so it both exercises
// the original bug and verifies the fix (onMouseDown={preventDefault} on
// the tag buttons).
function clickStealingFocus(el: HTMLElement): void {
  const notPrevented = fireEvent.mouseDown(el);
  if (notPrevented) el.focus();
  fireEvent.mouseUp(el);
  fireEvent.click(el);
}

// Opens the fixture project against an already-installed server, for the
// remount ("reopen") test below -- renderOpenedProject() owns its own
// server per call, which doesn't fit that test's need to unmount and
// re-render <App/> against the *same* server/state.
async function openProjectUI(): Promise<void> {
  fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
  await screen.findByRole("heading", { name: "Plan" });
}

it("saves every Custom plan field through one debounced PUT /plan with correct types, and a reopen renders them", async () => {
  const server = createMockBardServer();
  server.install();
  let rendered = render(<App />);
  await openProjectUI();

  fireEvent.change(screen.getByLabelText("Caption"), {
    target: { value: "brooding wizard folk" },
  });
  fireEvent.change(lyricsTextarea(), {
    target: { value: "[Chorus]\nburn the old maps" },
  });
  fireEvent.click(screen.getByLabelText("Instrumental"));
  fireEvent.click(screen.getByLabelText("Allow caption rewrite (Custom mode LM thinking)"));
  fireEvent.change(screen.getByLabelText("BPM"), { target: { value: "140" } });
  fireEvent.change(screen.getByLabelText("Key"), { target: { value: "D Minor" } });
  fireEvent.change(screen.getByLabelText("Time signature"), { target: { value: "3/4" } });
  fireEvent.change(screen.getByLabelText("Duration (sec)"), { target: { value: "185" } });
  fireEvent.change(screen.getByLabelText("Language"), { target: { value: "fr" } });

  // All nine edits land within the 500ms debounce window and collapse into
  // exactly one PUT carrying every change together.
  await waitFor(() => expect(planPutRequests(server)).toHaveLength(1));
  const body = planPutRequests(server)[0].body as Record<string, unknown>;
  expect(body).toMatchObject({
    caption: "brooding wizard folk",
    lyrics: "[Chorus]\nburn the old maps",
    instrumental: true,
    caption_rewrite: false, // fixture default is true
    bpm: 140,
    keyscale: "D Minor",
    timesignature: "3/4",
    duration_sec: 185,
    vocal_language: "fr",
  });
  expect(typeof body.bpm).toBe("number");
  expect(typeof body.duration_sec).toBe("number");
  expect(typeof body.instrumental).toBe("boolean");
  expect(typeof body.caption_rewrite).toBe("boolean");

  // Reopen: unmount (dropping all in-memory React state) and re-render
  // <App/> against the same mock server, so the Plan pane can only be
  // showing what a fresh GET /api/projects/{id} actually returned.
  rendered.unmount();
  rendered = render(<App />);
  await openProjectUI();

  expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe(
    "brooding wizard folk",
  );
  expect(lyricsTextarea().value).toBe("[Chorus]\nburn the old maps");
  expect((screen.getByLabelText("Instrumental") as HTMLInputElement).checked).toBe(true);
  expect(
    (screen.getByLabelText("Allow caption rewrite (Custom mode LM thinking)") as HTMLInputElement)
      .checked,
  ).toBe(false);
  expect((screen.getByLabelText("BPM") as HTMLInputElement).value).toBe("140");
  expect((screen.getByLabelText("Key") as HTMLInputElement).value).toBe("D Minor");
  expect((screen.getByLabelText("Time signature") as HTMLInputElement).value).toBe("3/4");
  expect((screen.getByLabelText("Duration (sec)") as HTMLInputElement).value).toBe("185");
  expect((screen.getByLabelText("Language") as HTMLInputElement).value).toBe("fr");

  rendered.unmount();
  server.uninstall();
});

it("clears BPM and key to null (not empty string) when the fields are emptied", async () => {
  app = await renderOpenedProject();

  fireEvent.change(screen.getByLabelText("BPM"), { target: { value: "" } });
  fireEvent.change(screen.getByLabelText("Key"), { target: { value: "" } });

  await waitFor(() => expect(planPutRequests(app!.server)).toHaveLength(1));
  const body = planPutRequests(app!.server)[0].body as Record<string, unknown>;
  expect(body.bpm).toBeNull();
  expect(body.keyscale).toBeNull();
});

it("collapses rapid keystrokes on the same field into a single trailing PUT with the latest value", async () => {
  app = await renderOpenedProject();
  const caption = screen.getByLabelText("Caption");

  // Each change resets the shared 500ms debounce timer, so nothing should
  // be sent until keystrokes stop -- and then only once, for the final
  // value.
  fireEvent.change(caption, { target: { value: "b" } });
  fireEvent.change(caption, { target: { value: "br" } });
  fireEvent.change(caption, { target: { value: "bro" } });
  expect(planPutRequests(app.server)).toHaveLength(0);

  await waitFor(() => expect(planPutRequests(app!.server)).toHaveLength(1));
  expect(planPutRequests(app.server)[0].body).toMatchObject({ caption: "bro" });
});

it("keeps a rejected edit visible without letting it clobber the last accepted plan", async () => {
  app = await renderOpenedProject();
  const server = app.server;

  // Accept one edit first, establishing "the last accepted plan" that the
  // rejected edit below must not be able to overwrite.
  fireEvent.change(screen.getByLabelText("Caption"), {
    target: { value: "accepted caption" },
  });
  await waitFor(() => expect(planPutRequests(server)).toHaveLength(1));
  expect(server.state.detail.plan.caption).toBe("accepted caption");

  // Make the *next* PUT /plan fail like a server-side validation rejection
  // (e.g. an out-of-range BPM), without touching mockServer.ts's shared PUT
  // handler used by every other test -- this wrapper only intercepts one
  // call and falls through to the installed mock for everything else.
  const previousFetch = globalThis.fetch;
  let failNext = true;
  vi.stubGlobal("fetch", (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (failNext && method === "PUT" && url.endsWith("/plan")) {
      failNext = false;
      const errorBody = { detail: "bpm must be between 40 and 240" };
      return Promise.resolve({
        ok: false,
        status: 422,
        statusText: "Unprocessable Entity",
        json: async () => errorBody,
        text: async () => JSON.stringify(errorBody),
      } as Response);
    }
    return previousFetch(input, init);
  });

  const bpmInput = screen.getByLabelText("BPM") as HTMLInputElement;
  fireEvent.change(bpmInput, { target: { value: "9999" } });

  // The rejected save surfaces an error but the typed value stays visible
  // in the field -- the UI does not silently roll the input back.
  await screen.findByText(/422|bpm must be between/i);
  expect(bpmInput.value).toBe("9999");

  // Crucially, the server-side plan was never overwritten with the
  // rejected value: the last *accepted* plan (caption from above, and the
  // original bpm) is still what's on disk.
  expect(server.state.detail.plan.bpm).toBe(120);
  expect(server.state.detail.plan.caption).toBe("accepted caption");

  vi.stubGlobal("fetch", previousFetch);
});

it("inserts a structure tag at the lyrics cursor instead of always appending", async () => {
  app = await renderOpenedProject();
  const textarea = lyricsTextarea();

  fireEvent.change(textarea, { target: { value: "la la la\nmore lyrics" } });
  textarea.focus();
  textarea.setSelectionRange(9, 9); // collapsed cursor right after the newline

  // A plain (non-focus-stealing) click on the tag button must not disturb
  // the textarea's cursor -- if it did, the tag would land at the end of
  // the text instead of where the user actually clicked from.
  clickStealingFocus(screen.getByRole("button", { name: "[Chorus]" }));

  expect(textarea.value).toBe("la la la\n[Chorus]\nmore lyrics");
});

it("replaces a selected lyrics range when a structure tag button is clicked", async () => {
  app = await renderOpenedProject();
  const textarea = lyricsTextarea();

  fireEvent.change(textarea, { target: { value: "la la la\nmore lyrics" } });
  textarea.focus();
  textarea.setSelectionRange(0, 8); // select "la la la"

  clickStealingFocus(screen.getByRole("button", { name: "[Bridge]" }));

  expect(textarea.value).toBe("[Bridge]\n\nmore lyrics");
});
