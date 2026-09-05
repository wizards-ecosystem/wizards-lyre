import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test/setup.ts"],
    // Keep runs deterministic: a single forked process, so the per-test mock
    // servers (src/test/mockServer.ts installs itself on the module-scope
    // fetch) and App's 5s health-poll interval can't interleave across
    // parallel workers.
    pool: "forks",
    fileParallelism: false,
    maxWorkers: 1,
  },
});
