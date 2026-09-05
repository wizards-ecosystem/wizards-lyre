// Regression tests for plan.sections rendering as labeled waveform regions
// (SPEC.md sec 7.2: sections are "region labels on the waveform"; sec 9.2
// Waveform pane). Covers the reviewer finding that sections were editable in
// the Plan pane but never drawn on the waveform: labels must appear when a
// take is selected, coexist with the single repaint selection instead of
// being deleted by it, and re-render when the section list is edited.
//
// Runs against the mocked fetch backend in src/test/mockServer.ts, with the
// wavesurfer stack stubbed out like App.lora.test.tsx does (jsdom has no
// canvas/layout for the real library).
import { act, cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Section } from "../api";
import { makeTakes } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";

interface FakeRegion {
  id: string;
  start: number;
  end: number;
  removed: boolean;
  remove: () => void;
}

// Richer than the stub in App.lora.test.tsx: tracks every addRegion call and
// the live region list so tests can assert what got drawn, and mirrors the
// real plugin firing region-created synchronously from addRegion (its
// saveRegion path) so App's region-created handler runs for labels too.
const regionHooks = vi.hoisted(() => ({
  handlers: {} as Record<string, (region: FakeRegion) => void>,
  regions: [] as FakeRegion[],
  addRegionCalls: [] as Record<string, unknown>[],
  dragSelectionOptions: null as Record<string, unknown> | null,
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
      enableDragSelection: (opts: Record<string, unknown>) => {
        regionHooks.dragSelectionOptions = opts;
      },
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
        regionHooks.addRegionCalls.push(params);
        regionHooks.handlers["region-created"]?.(region);
        return region;
      },
      clearRegions: () => {
        regionHooks.regions = [];
      },
    }),
  },
}));

const SECTIONS: Section[] = [
  { name: "intro", start_sec: 0, end_sec: 8, lyrics: "" },
  { name: "verse", start_sec: 8, end_sec: 32, lyrics: "la la" },
];

let app: OpenedProject | undefined;

beforeEach(() => {
  regionHooks.handlers = {};
  regionHooks.regions = [];
  regionHooks.addRegionCalls = [];
  regionHooks.dragSelectionOptions = null;
});

afterEach(() => {
  app?.cleanup();
  app = undefined;
  cleanup();
});

function labelRegions(): FakeRegion[] {
  return regionHooks.regions.filter((r) => r.id.startsWith("section-label-"));
}

// Simulates the plugin reporting a drag-created region (it fires
// region-created on pointer-up), acting like the plugin would: the region is
// in the live list before the event lands.
function simulateDragRegion(start: number, end: number): FakeRegion {
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
  return region;
}

it("renders persisted sections as labeled regions that survive the repaint selection", async () => {
  app = await renderOpenedProject({ takes: makeTakes(2), sections: SECTIONS });

  // Selecting a take mounts the waveform, which draws one labeled,
  // non-editable region per persisted section.
  fireEvent.click(screen.getByRole("listitem", { name: /seed 1001/ }));
  await waitFor(() => expect(regionHooks.addRegionCalls).toHaveLength(2));
  expect(regionHooks.addRegionCalls[0]).toMatchObject({
    id: "section-label-0",
    start: 0,
    end: 8,
    content: "intro",
    drag: false,
    resize: false,
  });
  expect(regionHooks.addRegionCalls[1]).toMatchObject({
    id: "section-label-1",
    start: 8,
    end: 32,
    content: "verse",
    drag: false,
    resize: false,
  });
  // The drag-select repaint region carries a fixed id, which is what keeps
  // its cleanup from ever touching the section labels.
  expect(regionHooks.dragSelectionOptions).toEqual({ id: "repaint-selection" });

  // Dragging a repaint region must not remove the section labels.
  simulateDragRegion(5, 10);
  await screen.findByText(/Region: 5\.0s/);
  expect(labelRegions()).toHaveLength(2);
  expect(labelRegions().every((r) => !r.removed)).toBe(true);

  // A second drag replaces the first selection, still leaving labels alone.
  simulateDragRegion(20, 25);
  await screen.findByText(/Region: 20\.0s/);
  expect(regionHooks.regions.filter((r) => r.id === "repaint-selection")).toHaveLength(1);
  expect(labelRegions()).toHaveLength(2);
});

it("re-renders waveform labels when sections are edited in the Plan pane", async () => {
  app = await renderOpenedProject({ takes: makeTakes(2), sections: SECTIONS });
  fireEvent.click(screen.getByRole("listitem", { name: /seed 1001/ }));
  await waitFor(() => expect(regionHooks.addRegionCalls).toHaveLength(2));

  // Adding a section in the Plan pane draws its label on the waveform.
  fireEvent.click(screen.getByRole("button", { name: "Add section" }));
  await waitFor(() =>
    expect(regionHooks.addRegionCalls.some((c) => c.id === "section-label-2")).toBe(true),
  );
  expect(regionHooks.addRegionCalls.find((c) => c.id === "section-label-2")).toMatchObject({
    start: 0,
    end: 0,
    content: "",
    drag: false,
    resize: false,
  });
  expect(labelRegions()).toHaveLength(3);

  // Deleting a section removes its label.
  fireEvent.click(screen.getAllByRole("button", { name: /Delete section/ })[0]);
  await waitFor(() => expect(labelRegions()).toHaveLength(2));
});
