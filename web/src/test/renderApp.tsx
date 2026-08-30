// Shared setup for the App-level regression tests: installs the mocked
// fetch backend, stubs window.confirm (jsdom's built-in confirm is a no-op
// that returns undefined, which the base-model-swap gates would read as
// "declined"), renders the real <App/>, and opens the fixture project so
// tests start inside the three-pane workspace.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, vi } from "vitest";
import App from "../App";
import type { Lora, Section, Take } from "../api";
import { createMockLyreServer, type MockLyreServer } from "./mockServer";

// The per-take checkbox that opts a take into style-pack training (App.tsx
// only gives it a title, so query it by that).
export const LORA_SOURCE_TITLE = "Include in style pack training source";

export interface OpenedProject {
  server: MockLyreServer;
  confirm: ReturnType<typeof vi.fn>;
  cleanup: () => void;
}

export async function renderOpenedProject(
  opts: { takes?: Take[]; loras?: Lora[]; sections?: Section[] } = {},
): Promise<OpenedProject> {
  const server = createMockLyreServer();
  if (opts.sections) {
    server.state.detail = {
      ...server.state.detail,
      plan: { ...server.state.detail.plan, sections: opts.sections },
    };
  }
  if (opts.takes) {
    server.state.detail = {
      ...server.state.detail,
      takes: opts.takes,
      project: {
        ...server.state.detail.project,
        active_take_id: opts.takes[0]?.id ?? null,
      },
    };
  }
  server.state.loras = opts.loras ?? [];
  server.install();

  const originalConfirm = window.confirm;
  const confirm = vi.fn(() => true);
  window.confirm = confirm as unknown as typeof window.confirm;

  const rendered = render(<App />);
  // Wait for the library to load, then open the fixture project.
  fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
  const takeCount = server.state.detail.takes.length;
  await waitFor(() => {
    expect(screen.getAllByTitle(LORA_SOURCE_TITLE)).toHaveLength(takeCount);
  });

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
