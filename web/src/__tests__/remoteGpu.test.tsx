// RemoteGpuControl (docs/remote-gpu.md): the opt-in Remote GPU panel. The
// money rule is the thing worth pinning: Start asks first, with the rate, and
// declining sends nothing.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RemoteGpuStatus } from "../api";
import { RemoteGpuControl } from "../components/RemoteGpuControl";

const OFF: RemoteGpuStatus = {
  provisioner: "runpod",
  label: "Runpod",
  can_provision: true,
  stop_on_exit: true,
  quote: {
    gpu: "NVIDIA RTX A5000",
    cloud: "SECURE",
    disk_gb: 60,
    hourly_usd: 0.27,
    memory_gb: 24,
    availability: "HIGH",
  },
  connection: null,
  session: null,
  host: null,
};

const RUNNING: RemoteGpuStatus = {
  ...OFF,
  quote: null,
  connection: { base_url: "https://pod123-8000.proxy.runpod.net" },
  session: {
    id: "pod123",
    state: "starting",
    detail: "pulling the host image (the slow part)",
    gpu: "NVIDIA RTX A5000",
    elapsed_sec: 125,
    hourly_usd: 0.27,
    cost_estimate_usd: 0.0094,
  },
  host: { connected: false, ready: false, error: "unreachable" },
};

function mockFetch(routes: Record<string, () => { status?: number; body: unknown }>) {
  const calls: { method: string; url: string; body: unknown }[] = [];
  const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : null });
    const route = routes[`${method} ${url}`];
    const { status = 200, body } = route ? route() : { status: 404, body: { detail: "none" } };
    return {
      ok: status < 300,
      status,
      statusText: status < 300 ? "OK" : "Error",
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("RemoteGpuControl", () => {
  it("shows the rate before anything is rented", async () => {
    mockFetch({ "GET /api/remote-gpu": () => ({ body: OFF }) });
    render(<RemoteGpuControl />);
    expect(await screen.findByText(/\$0\.27\/hr, secure cloud/)).toBeTruthy();
    expect(screen.getByText("Start GPU")).toBeTruthy();
  });

  it("asks before starting, and declining sends nothing", async () => {
    const calls = mockFetch({
      "GET /api/remote-gpu": () => ({ body: OFF }),
      "POST /api/remote-gpu/session": () => ({ body: RUNNING }),
    });
    render(<RemoteGpuControl />);
    fireEvent.click(await screen.findByText("Start GPU"));
    expect(screen.getByRole("alertdialog").textContent).toContain("$0.27/hr");
    fireEvent.click(screen.getByText("Cancel"));
    expect(calls.some((c) => c.method === "POST")).toBe(false);

    fireEvent.click(screen.getByText("Start GPU"));
    const dialog = screen.getByRole("alertdialog");
    await act(async () => {
      fireEvent.click(dialog.querySelector("button:last-of-type") as HTMLButtonElement);
    });
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(await screen.findByText(/about \$0\.01 so far/)).toBeTruthy();
    expect(screen.getByText("Stop GPU")).toBeTruthy();
  });

  it("stops a running GPU", async () => {
    const calls = mockFetch({
      "GET /api/remote-gpu": () => ({ body: RUNNING }),
      "DELETE /api/remote-gpu/session": () => ({ body: OFF }),
    });
    render(<RemoteGpuControl />);
    const stop = await screen.findByText("Stop GPU");
    await act(async () => {
      fireEvent.click(stop);
    });
    await waitFor(() =>
      expect(calls.some((c) => c.method === "DELETE" && c.url === "/api/remote-gpu/session")).toBe(
        true,
      ),
    );
    expect(await screen.findByText("Start GPU")).toBeTruthy();
  });

  it("connects a self-hosted host and shows the server's refusal", async () => {
    const calls = mockFetch({
      "GET /api/remote-gpu": () => ({
        body: { ...OFF, provisioner: "manual", label: "Self-hosted", can_provision: false },
      }),
      "PUT /api/remote-gpu/connection": () => ({
        status: 400,
        body: { detail: "the Remote GPU address must use https://" },
      }),
    });
    render(<RemoteGpuControl />);
    fireEvent.change(await screen.findByLabelText("Remote GPU address"), {
      target: { value: "http://gpu.example" },
    });
    fireEvent.change(screen.getByLabelText("Remote GPU secret"), {
      target: { value: "x".repeat(40) },
    });
    await act(async () => {
      fireEvent.click(screen.getByText("Connect"));
    });
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("the Remote GPU address must use https://");
    const put = calls.find((c) => c.method === "PUT");
    expect(put?.body).toEqual({ base_url: "http://gpu.example", secret: "x".repeat(40) });
  });

  it("flags a host running an older build", async () => {
    mockFetch({
      "GET /api/remote-gpu": () => ({
        body: {
          ...OFF,
          can_provision: false,
          quote: null,
          connection: { base_url: "https://gpu.example" },
          host: { connected: true, ready: true, gpu: "RTX A5000", stale_build: true, error: null },
        },
      }),
    });
    render(<RemoteGpuControl />);
    expect(await screen.findByText("Remote GPU ready")).toBeTruthy();
    expect(screen.getByText(/older build/)).toBeTruthy();
  });
});
