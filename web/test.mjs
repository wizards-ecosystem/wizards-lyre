// Inline Vitest entry point.
//
// `vitest` normally loads vitest.config.* through Vite's config loader,
// which bundles that file with esbuild, and esbuild walks ancestor
// directories looking for node_modules / package.json. Under the Conclave
// NTFS jail, some ancestor directories (e.g. Documents) exist but deny
// enumeration outright (EACCES, not ENOENT), and esbuild's walk-up does not
// tolerate that -- it aborts instead of continuing past a denied directory.
// Passing the config inline (configFile: false) skips that walk entirely,
// so this script is the supported way to run the frontend tests here --
// exactly mirroring build.mjs / dev.mjs, which do the same for the build.
import { startVitest } from "vitest/node";

const vitest = await startVitest(
  "test",
  [],
  {
    run: true,
    configFile: false,
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    // Keep runs deterministic: a single forked process so per-test mock
    // servers and App's 5s health-poll interval can't interleave across
    // parallel workers.
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
  },
);

if (!vitest) {
  // startVitest bails without an instance when nothing matched -- never a
  // silent pass for a suite that should be there.
  console.error("vitest exited without running any tests");
  process.exit(1);
}

const files = vitest.state.getFiles();
const failed = files.length === 0 || files.some((file) => file.result?.state === "fail");
if (files.length === 0) {
  console.error("no test files matched src/**/*.test.{ts,tsx}");
}
await vitest.close();
process.exit(failed ? 1 : 0);
