// Pre-bundles the handful of CJS npm packages the dev server needs
// (react, react-dom/client, the jsx runtimes) into standalone ESM files
// under .dev-vendor/, using Vite's own (Rollup-based) production build.
//
// Why: Vite's normal dev-time path for CJS interop is its esbuild
// dependency pre-bundler, and esbuild's own module resolver
// unconditionally walks every ancestor directory up to the drive root --
// see the comment in build.mjs. Under the Conclave jail this repo's path
// has an ancestor a few levels up that denies directory listing, so that
// walk aborts with EACCES on *any* bare npm import, before it ever gets
// to react specifically (reproducible with a bare `esbuild.build({
// bundle: true })` on an empty stdin module importing "react", nothing
// Vite- or React-specific about it). Production `vite build` never hits
// this because it bundles node_modules deps via Rollup's own resolver,
// which is plain `fs.stat`-based and has no such walk. Re-running that
// same safe Rollup path once at dev-server startup, ahead of time, sidesteps
// the buggy esbuild path entirely for dev too.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { build } from "vite";

export const VENDOR_SPECIFIERS = {
  react: "react",
  // @vitejs/plugin-react (vite:react-refresh) unconditionally force-adds
  // bare "react-dom" to optimizeDeps.include alongside "react-dom/client",
  // even though nothing in this app imports it directly -- both need an
  // alias or Vite's esbuild optimizer will still reach for it.
  "react-dom": "react-dom",
  "react-dom-client": "react-dom/client",
  "jsx-runtime": "react/jsx-runtime",
  "jsx-dev-runtime": "react/jsx-dev-runtime",
};

export async function buildDevVendor(root, outDir) {
  const nodeRequire = createRequire(import.meta.url);
  // react and the jsx runtimes are pure `module.exports = require(...)`
  // re-exports, which defeats Rollup's static named-export detection for
  // CJS interop (it only sees a dynamic re-export, not a literal exports
  // object) -- every named import (`useEffect`, `jsx`, ...) collapses to a
  // single `default` export otherwise. Writing an explicit ESM wrapper per
  // package -- `export const useEffect = mod.useEffect` for every key we
  // already know the module has (found via Node's own `require`, which
  // hits none of esbuild's walk-up problems) -- sidesteps Rollup's
  // detection heuristics entirely: these are real, statically-visible
  // named exports.
  const srcDir = path.join(root, ".dev-vendor-src");
  fs.rmSync(srcDir, { recursive: true, force: true });
  fs.mkdirSync(srcDir, { recursive: true });

  const entry = {};
  for (const [name, specifier] of Object.entries(VENDOR_SPECIFIERS)) {
    const mod = nodeRequire(specifier);
    const namedKeys = Object.keys(mod).filter((key) => key !== "default");
    const wrapperPath = path.join(srcDir, `${name}.mjs`);
    const lines = [
      `import __mod from ${JSON.stringify(specifier)};`,
      `export default __mod;`,
      ...namedKeys.map((key) => `export const ${key} = __mod[${JSON.stringify(key)}];`),
    ];
    fs.writeFileSync(wrapperPath, lines.join("\n") + "\n");
    entry[name] = wrapperPath;
  }

  try {
    // Library mode (not a normal multi-page app build): each entry keeps
    // its full public export surface. A plain multi-entry
    // `rollupOptions.input` build instead tree-shakes each chunk down to
    // only what the *other* entries in this same build happen to import
    // from it.
    await build({
      root,
      configFile: false,
      logLevel: "warn",
      build: {
        outDir,
        emptyOutDir: true,
        minify: false,
        target: "esnext",
        lib: {
          entry,
          formats: ["es"],
          fileName: (_format, entryName) => `${entryName}.js`,
        },
        rollupOptions: {
          output: {
            chunkFileNames: "chunk-[hash].js",
          },
        },
      },
    });
  } finally {
    fs.rmSync(srcDir, { recursive: true, force: true });
  }
}
