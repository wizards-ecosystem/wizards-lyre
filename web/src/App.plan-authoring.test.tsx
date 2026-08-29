// Regression tests for Plan-authoring polish (SPEC.md sec 4.4/9.2 structure
// tags, sec 7.2 named sections): the [Intro]/[Verse]/[Chorus]/[Bridge]/
// [Outro] lyrics tag buttons, and the Plan pane's section list (add/edit/
// remove, plus "Add section from region").
//
// Deliberately does not duplicate:
// - App.sections.test.tsx, which owns section labels *drawn on the
//   waveform* (addRegion calls, label survival across drag selections,
//   label re-render on edit/delete). This file only asserts the Plan pane's
//   own fields and the PUT /plan bodies they produce.
// - App.plan.test.tsx, which owns the flat scalar Plan fields, the shared
//   debounce/save-chain mechanics, and two structure-tag button cases
//   (Chorus caret-insert, Bridge selection-replace) as part of proving the
//   focus-stealing fix. This file instead sweeps *all five* tag buttons for
//   caret-insert and selection-replace, and covers the two things neither
//   sibling file does: the textarea regaining focus after an insert made
//   while it wasn't focused, and sections going through the same PUT
//   /plan-persists / rejected-edit-keeps-last-accepted-plan pattern that
//   App.plan.test.tsx already established for scalar fields.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with the
// wavesurfer stack stubbed out like App.sections.test.tsx (jsdom has no
// canvas/layout for the real library, and the region mock is needed here too
// for the "add section from region" case).
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { RecordedRequest } from "./test/mockServer";
import { makeTakes } from "./test/mockServer";
import { renderOpenedProject, type OpenedProject } from "./test/renderApp";

// Mirrors App.tsx's STRUCTURE_TAGS (not exported) -- SPEC.md sec 4.4/9.2's
// fixed structure-tag palette.
const STRUCTURE_TAGS = ["Intro", "Verse", "Chorus", "Bridge", "Outro"] as const;

interface FakeRegion {
  id: string;
  start: number;
  end: number;
  removed: boolean;
  remove: () => void;
}

// Same shape as App.sections.test.tsx's mock: tracks every addRegion call so
// the "add section from region" test can drive a drag-created region, while
// section-label addRegion calls (fired by App's own label-render effect)
// pass through harmlessly since App.tsx's region-created handler ignores
// ids that start with the section-label prefix.
const regionHooks = vi.hoisted(() => ({
  handlers: {} as Record<string, (region: FakeRegion) => void>,
  regions: [] as FakeRegion[],
}));

vi.mock("wavesurfer.js", () => ({
  default: {
    create: () => ({ on: () => {}, destroy: () => {} }),
  },
}));

vi.mock("wavesurfer.js/plugins/regions", () => ({
  default: {
    create: () => ({
      on: (event: string, cb: (region: FakeRegion) => void) => {
        regionHooks.handlers[event] = cb;
      },
      getRegions: () => regionHooks.regions,
      enableDragSelection: () => {},
      addRegion: (params: Record<string, unknown>) => {
        const region: FakeRegion = {
          id: String(params.id ?? ""),
          start: Number(params.start ?? 0),
          end: Number(params.end ?? 0),
          removed: false,
          remove: () => {
            region.removed = true;
            regionHooks.regions = regionHooks.regions.filter((r) => r !== region);
          },
        };
        regionHooks.regions.push(region);
        regionHooks.handlers["region-created"]?.(region);
        return region;
      },
      clearRegions: () => {
        regionHooks.regions = [];
      },
    }),
  },
}));

function simulateDragRegion(start: number, end: number): void {
  const region: FakeRegion = {
    id: "repaint-selection",
    start,
    end,
    removed: false,
    remove: () => {
      region.removed = true;
      regionHooks.regions = regionHooks.regions.filter((r) => r !== region);
    },
  };
  regionHooks.regions.push(region);
  act(() => {
    regionHooks.handlers["region-created"](region);
  });
}

let app: OpenedProject | undefined;

beforeEach(() => {
  regionHooks.handlers = {};
  regionHooks.regions = [];
});

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function planPutRequests(server: OpenedProject["server"]): RecordedRequest[] {
  return server.requests.filter((r) => r.method === "PUT" && r.url.endsWith("/plan"));
}

function lastPlanPutBody(server: OpenedProject["server"]): Record<string, unknown> {
  const reqs = planPutRequests(server);
  return reqs[reqs.length - 1].body as Record<string, unknown>;
}

// The Lyrics <label> also wraps the structure-tag buttons, so
// getByLabelText("Lyrics") resolves to the first labelable descendant (a tag
// button), not the textarea -- see the identical note in App.plan.test.tsx.
// The Plan pane has exactly one <textarea>.
function lyricsTextarea(): HTMLTextAreaElement {
  const pane = screen.getByRole("heading", { name: "Plan" }).closest("section");
  if (!pane) throw new Error("plan pane not found");
  const el = within(pane).getByLabelText("Lyrics");
  if (!el) throw new Error("lyrics textarea not found");
  return el as HTMLTextAreaElement;
}

// The wrapping <label> associates implicitly with only the *first*
// labelable descendant (App.plan.test.tsx hits the same quirk via
// getByLabelText("Lyrics")) -- but for a button that is the associated
// control, accessible-name computation then also folds in the whole
// label's text content (including the other four tag buttons and the
// textarea value), so getByRole("button", { name: "[Intro]" }) never
// matches. Querying by literal button text sidesteps that quirk for every
// tag uniformly instead of special-casing the first one.
function tagButton(tag: string): HTMLElement {
  const pane = screen.getByRole("heading", { name: "Plan" }).closest("section");
  if (!pane) throw new Error("plan pane not found");
  const palette = pane.querySelector(".lyrics-tag-palette");
  if (!palette) throw new Error("lyrics tag palette not found");
  const button = Array.from(palette.querySelectorAll("button")).find(
    (b) => b.textContent === `[${tag}]`,
  );
  if (!button) throw new Error(`tag button [${tag}] not found`);
  return button;
}

it.each(STRUCTURE_TAGS)(
  "the [%s] button inserts the bracketed tag at the lyrics caret, preserving surrounding text/newlines, and persists it through PUT /plan",
  async (tag) => {
    app = await renderOpenedProject();
    const textarea = lyricsTextarea();

    fireEvent.change(textarea, { target: { value: "line1\nline2" } });
    textarea.focus();
    textarea.setSelectionRange(6, 6); // collapsed caret right at the start of "line2"

    fireEvent.click(tagButton(tag));

    const expected = `line1\n[${tag}]\nline2`;
    expect(textarea.value).toBe(expected);

    await waitFor(() => expect(planPutRequests(app!.server)).toHaveLength(1));
    expect(planPutRequests(app!.server)[0].body).toMatchObject({ lyrics: expected });
  },
);

it.each(STRUCTURE_TAGS)(
  "the [%s] button replaces a selected lyrics range, keeping the text and newlines on both sides intact",
  async (tag) => {
    app = await renderOpenedProject();
    const textarea = lyricsTextarea();

    fireEvent.change(textarea, { target: { value: "before\nSELECTED\nafter" } });
    textarea.focus();
    textarea.setSelectionRange(7, 15); // selects exactly "SELECTED"

    fireEvent.click(tagButton(tag));

    expect(textarea.value).toBe(`before\n[${tag}]\n\nafter`);
  },
);

it("restores focus to the lyrics textarea after inserting a tag while it wasn't focused, placing the cursor after the insertion", async () => {
  app = await renderOpenedProject();
  const textarea = lyricsTextarea();

  fireEvent.change(textarea, { target: { value: "verse one" } });
  // Nothing has focused the textarea yet -- insertLyricsTag's unfocused
  // branch appends at the end instead of honoring a stale cursor.
  expect(document.activeElement).not.toBe(textarea);

  fireEvent.click(tagButton("Outro"));

  const expected = "verse one\n[Outro]\n";
  expect(textarea.value).toBe(expected);

  await waitFor(() => expect(document.activeElement).toBe(textarea));
  expect(textarea.selectionStart).toBe(expected.length);
  expect(textarea.selectionEnd).toBe(expected.length);
});

it("adds a blank named section, edits its fields, persists each through PUT /plan, and removes it", async () => {
  app = await renderOpenedProject();
  const server = app.server;

  fireEvent.click(screen.getByRole("button", { name: "Add section" }));

  fireEvent.change(screen.getByPlaceholderText("name"), { target: { value: "Intro" } });
  fireEvent.change(screen.getByTitle("start (sec)"), { target: { value: "1.5" } });
  fireEvent.change(screen.getByTitle("end (sec)"), { target: { value: "9" } });
  fireEvent.change(screen.getByPlaceholderText("lyrics snippet"), {
    target: { value: "la la" },
  });

  await waitFor(() => expect(planPutRequests(server).length).toBeGreaterThan(0));
  expect(lastPlanPutBody(server).sections).toEqual([
    { name: "Intro", start_sec: 1.5, end_sec: 9, lyrics: "la la" },
  ]);
  expect(server.state.detail.plan.sections).toEqual([
    { name: "Intro", start_sec: 1.5, end_sec: 9, lyrics: "la la" },
  ]);

  fireEvent.click(screen.getByRole("button", { name: /Delete section/ }));

  await waitFor(() => {
    expect(lastPlanPutBody(server).sections).toEqual([]);
  });
  expect(screen.queryByPlaceholderText("name")).toBeNull();
});

it("creates a section from the current waveform selection, using the selected region's bounds", async () => {
  app = await renderOpenedProject({ takes: makeTakes(1) });
  const server = app.server;

  fireEvent.click(screen.getByRole("listitem", { name: /seed 1001/ }));
  simulateDragRegion(12, 34);
  await screen.findByText(/Region: 12\.0s/);

  fireEvent.click(screen.getByRole("button", { name: "Add section from region" }));

  await waitFor(() => expect(planPutRequests(server)).toHaveLength(1));
  expect(planPutRequests(server)[0].body).toMatchObject({
    sections: [{ name: "", start_sec: 12, end_sec: 34, lyrics: "" }],
  });
  expect((screen.getByTitle("start (sec)") as HTMLInputElement).value).toBe("12");
  expect((screen.getByTitle("end (sec)") as HTMLInputElement).value).toBe("34");
});

it("surfaces the server validation error for an invalid/overlapping section edit while retaining the last accepted plan", async () => {
  app = await renderOpenedProject({
    sections: [{ name: "intro", start_sec: 0, end_sec: 8, lyrics: "" }],
  });
  const server = app.server;

  // Establish "the last accepted plan" with one edit the mock server
  // actually stores, so the rejected edit below has something real to fail
  // to overwrite.
  const endInput = screen.getByTitle("end (sec)") as HTMLInputElement;
  fireEvent.change(endInput, { target: { value: "10" } });
  await waitFor(() => expect(planPutRequests(server)).toHaveLength(1));
  expect(server.state.detail.plan.sections[0]).toMatchObject({ end_sec: 10 });

  // Make the *next* PUT /plan fail like a server-side rejection of an
  // invalid/overlapping section edit -- same one-shot wrapper technique as
  // App.plan.test.tsx's scalar-field version of this test, without touching
  // the shared mockServer.ts handler used by every other test.
  const previousFetch = globalThis.fetch;
  let failNext = true;
  vi.stubGlobal("fetch", (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    if (failNext && method === "PUT" && url.endsWith("/plan")) {
      failNext = false;
      const errorBody = { detail: "sections overlap: intro overlaps verse" };
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

  fireEvent.change(endInput, { target: { value: "999" } });

  await screen.findByText(/422|overlap/i);
  // The rejected edit stays visible -- the UI does not silently roll the
  // field back to the last accepted value.
  expect(endInput.value).toBe("999");
  // But the server-side plan was never overwritten with it.
  expect(server.state.detail.plan.sections[0]).toMatchObject({ end_sec: 10 });

  vi.stubGlobal("fetch", previousFetch);
});
