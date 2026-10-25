// SPEC.md sec 16.5/16.6: jobs are observed from the queue, not from the
// component that started them, and plan edits survive failures and reloads.
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { makeProjectDetail, makeTakes, PROJECT_ID } from "../test/mockServer";
import { renderOpenedProject, type OpenedProject } from "../test/renderApp";
import { createMockLyreServer } from "../test/mockServer";

// jsdom has no canvas/layout for wavesurfer.js; same stub as the other
// App-level suites.
vi.mock("wavesurfer.js", () => ({
  default: { create: () => ({ on: () => {}, destroy: () => {} }) },
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

let app: OpenedProject | null = null;

afterEach(() => {
  app?.cleanup();
  app = null;
  localStorage.clear();
  vi.restoreAllMocks();
});

function jobPolls(id: string): number {
  return app!.server.requests.filter((r) => r.method === "GET" && r.url === `/api/jobs/${id}`)
    .length;
}

describe("Job observation (SPEC.md sec 16.5)", () => {
  it("follows a slow job through its stages without treating failed polls as failure", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({
      statuses: ["queued", "running", "running", "running", "done"],
      stages: [null, "loading_model", "planning", "rendering", "writing_take"],
    });
    app.server.failNextJobPolls(2);
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(
      await screen.findByText("Waiting for GPU: generate", {}, { timeout: 4000 }),
    ).toBeTruthy();
    expect(await screen.findByText("Generating: generate", {}, { timeout: 6000 })).toBeTruthy();
    expect(await screen.findByText("Complete: generate", {}, { timeout: 4000 })).toBeTruthy();
    // Two lost polls were just lost observations: no error, and the take is there.
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getAllByText("seed 1002").length).toBeGreaterThan(0);
  }, 20000);

  it("resends a lost submission under the same identity and queues one job", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.loseNextJobsPostResponses(1);
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(app!.server.jobRequests("generate")).toHaveLength(2));
    const [first, second] = app.server.jobSubmissionIds();
    expect(first).toBeTruthy();
    expect(second).toBe(first);
    await waitFor(() => expect(screen.getAllByText("seed 1002").length).toBeGreaterThan(0));
    // One job: the resend was answered with the job the first send created.
    const queued = (await (await fetch(`/api/jobs?project_id=${PROJECT_ID}`)).json()) as {
      id: string;
    }[];
    expect(queued.map((job) => job.id)).toEqual(["job-1"]);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("gives every submission its own identity", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Complete: generate");
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => expect(app!.server.jobRequests("generate")).toHaveLength(2));
    const [first, second] = app.server.jobSubmissionIds();
    expect(first).not.toBe(second);
  });

  it("attaches to an ordinary job that was already running when the page loaded", async () => {
    const server = createMockLyreServer();
    server.state.detail = makeProjectDetail(makeTakes(1));
    server.seedJob("job-running", PROJECT_ID, "generate", "running");
    server.install();
    const rendered = render(<App />);
    app = {
      server,
      confirm: vi.fn(),
      cleanup: () => {
        rendered.unmount();
        server.uninstall();
      },
    };
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));

    // Recovered from the queue: the page never submitted this job.
    expect(await screen.findByText("Running: generate")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Generate" })).toHaveProperty("disabled", true);
    expect(server.jobRequests()).toHaveLength(0);

    server.setJobScript("job-running", { statuses: ["done"] });
    expect(await screen.findByText("Complete: generate", {}, { timeout: 4000 })).toBeTruthy();
    // The take another session started is listed, but not auto-selected.
    await waitFor(() => expect(screen.getAllByText("seed 1002").length).toBeGreaterThan(0));
  });

  it("stops observing a job another client cancelled", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["running"], stages: ["rendering"] });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText("Generating: generate")).toBeTruthy();

    app.server.setJobScript("job-1", { statuses: ["cancelled"] });
    expect(await screen.findByText("Cancelled: generate", {}, { timeout: 4000 })).toBeTruthy();
    const pollsAtCancel = jobPolls("job-1");
    await act(() => new Promise((resolve) => setTimeout(resolve, 2500)));
    expect(jobPolls("job-1")).toBe(pollsAtCancel);
    expect(screen.getByRole("button", { name: "Generate" })).toHaveProperty("disabled", false);
  });

  it("cancels a running job and explains that the render may finish first", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["running"], stages: ["rendering"] });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(await screen.findByText("Cancelling: generate")).toBeTruthy();
    expect(screen.getByText(/cannot stop mid-render/)).toBeTruthy();
    expect(app.server.requests.some((r) => r.url === "/api/jobs/job-1/cancel")).toBe(true);
  });

  it("offers a retry and a copyable summary for a failed job", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["error"], error: "CUDA out of memory" });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Interrupted: generate");

    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    fireEvent.click(screen.getByRole("button", { name: "Copy error summary" }));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText.mock.calls[0]).toEqual([expect.stringContaining("job-1")]);

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Complete: generate", {}, { timeout: 4000 })).toBeTruthy();
    expect(app.server.requests.some((r) => r.url === "/api/jobs/job-1/retry")).toBe(true);
  });

  it("never applies a finished job to a different project", async () => {
    const server = createMockLyreServer();
    server.state.detail = makeProjectDetail(makeTakes(1));
    const other = makeProjectDetail([]);
    other.project = { ...other.project, id: "proj-b", title: "Other Song", active_take_id: null };
    server.state.createdDetails.set("proj-b", other);
    server.state.projects = [
      ...server.state.projects,
      { id: "proj-b", title: "Other Song", updated_at: "", active_take_id: null, favorite: false },
    ];
    server.install();
    const rendered = render(<App />);
    app = {
      server,
      confirm: vi.fn(),
      cleanup: () => {
        rendered.unmount();
        server.uninstall();
      },
    };
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));
    await screen.findAllByText("seed 1001");
    server.scriptNextJob({ statuses: ["running"] });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Running: generate");

    fireEvent.click(screen.getByRole("button", { name: "Open Other Song" }));
    await screen.findByRole("button", { name: "Other Song" });
    server.setJobScript("job-1", { statuses: ["done"] });
    await act(() => new Promise((resolve) => setTimeout(resolve, 2500)));

    // Project B shows its own (empty) take list and no strip for A's job.
    expect(screen.getByRole("button", { name: "Other Song" })).toBeTruthy();
    expect(screen.queryByText("Complete: generate")).toBeNull();
    expect(screen.getByText("No takes yet.")).toBeTruthy();
  });

  it("keeps observing past ten minutes instead of declaring the job lost", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.scriptNextJob({ statuses: ["running"], stages: ["rendering"] });
    fireEvent.click(screen.getByRole("button", { name: "Generate" }));
    await screen.findByText("Generating: generate");

    const realNow = Date.now.bind(Date);
    vi.spyOn(Date, "now").mockImplementation(() => realNow() + 11 * 60 * 1000);
    const before = jobPolls("job-1");
    await act(() => new Promise((resolve) => setTimeout(resolve, 6500)));
    expect(jobPolls("job-1")).toBeGreaterThan(before);
    expect(screen.getByText("Generating: generate")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  }, 15000);
});

describe("Plan save continuity (SPEC.md sec 16.6)", () => {
  it("keeps the text and retries visibly when a save fails", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    app.server.failNextPlanPuts(1);
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "keep me" } });

    expect(await screen.findByText("Save failed, retrying")).toBeTruthy();
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("keep me");
    fireEvent.click(screen.getByRole("button", { name: "Retry now" }));
    expect(await screen.findByText("Saved")).toBeTruthy();
    expect(app.server.state.detail.plan.caption).toBe("keep me");
  });

  it("does not overwrite a plan that changed elsewhere", async () => {
    app = await renderOpenedProject({ takes: makeTakes(1) });
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "first" } });
    await screen.findByText("Saved");
    // Another client saves meanwhile.
    app.server.state.detail = {
      ...app.server.state.detail,
      plan: { ...app.server.state.detail.plan, caption: "theirs", revision: 99 },
    };

    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "mine" } });
    expect(await screen.findByText("Changed elsewhere")).toBeTruthy();
    expect(app.server.state.detail.plan.caption).toBe("theirs");
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("mine");

    fireEvent.click(screen.getByRole("button", { name: "Keep my edits" }));
    await screen.findByText("Saved");
    expect(app.server.state.detail.plan.caption).toBe("mine");
  });

  it("resumes the last unsent edit after a reload when the plan has not moved on", async () => {
    const server = createMockLyreServer();
    server.state.detail = {
      ...makeProjectDetail(makeTakes(1)),
      plan: { ...makeProjectDetail([]).plan, revision: 3 },
    };
    localStorage.setItem(
      `lyre.planDraft.${PROJECT_ID}`,
      JSON.stringify({
        projectId: PROJECT_ID,
        plan: { ...server.state.detail.plan, caption: "typed just before the reload" },
        baseRevision: 3,
        savedAt: Date.now(),
      }),
    );
    server.install();
    const rendered = render(<App />);
    app = {
      server,
      confirm: vi.fn(),
      cleanup: () => {
        rendered.unmount();
        server.uninstall();
      },
    };
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));

    await waitFor(() =>
      expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe(
        "typed just before the reload",
      ),
    );
    await waitFor(() =>
      expect(server.state.detail.plan.caption).toBe("typed just before the reload"),
    );
  });

  it("offers, but does not apply, an unsent edit older than the saved plan", async () => {
    const server = createMockLyreServer();
    server.state.detail = {
      ...makeProjectDetail(makeTakes(1)),
      plan: { ...makeProjectDetail([]).plan, caption: "saved later", revision: 5 },
    };
    localStorage.setItem(
      `lyre.planDraft.${PROJECT_ID}`,
      JSON.stringify({
        projectId: PROJECT_ID,
        plan: { ...server.state.detail.plan, caption: "stale draft", revision: 2 },
        baseRevision: 2,
        savedAt: Date.now() - 60_000,
      }),
    );
    server.install();
    const rendered = render(<App />);
    app = {
      server,
      confirm: vi.fn(),
      cleanup: () => {
        rendered.unmount();
        server.uninstall();
      },
    };
    fireEvent.click(await screen.findByRole("button", { name: "Open Test Song" }));

    expect(await screen.findByText(/older than the saved plan/)).toBeTruthy();
    expect((screen.getByLabelText("Caption") as HTMLInputElement).value).toBe("saved later");
    expect(server.requests.filter((r) => r.method === "PUT")).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Restore my edits" }));
    await waitFor(() => expect(server.state.detail.plan.caption).toBe("stale draft"));
  });
});
