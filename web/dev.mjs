// Inline Vite dev-server entry point. See build.mjs for why this project
// loads its Vite config as inline JS (configFile: false) instead of letting
// the CLI auto-load vite.config.js.
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { buildDevVendor, VENDOR_SPECIFIERS } from "./dev-vendor.mjs";

const root = path.dirname(fileURLToPath(import.meta.url));
const bardPort = process.env.BARD_PORT ?? "8421";

// See dev-vendor.mjs for why react/react-dom/jsx-runtime are pre-bundled
// up front here instead of going through Vite's normal (esbuild-based)
// dev dependency optimizer.
const vendorDir = path.join(root, ".dev-vendor");
await buildDevVendor(root, vendorDir);
// Vite's alias matching treats a string `find` as a prefix match (`id ===
// find || id.startsWith(find + "/")`), so a shorter alias checked first can
// swallow a longer, more specific one -- "react-dom" would otherwise match
// "react-dom/client" as "react-dom" + "/client" and try to resolve "client"
// as a subpath of the react-dom.js file. Longest-specifier-first avoids that.
const vendorAlias = Object.keys(VENDOR_SPECIFIERS)
  .map((name) => ({
    find: VENDOR_SPECIFIERS[name],
    replacement: path.join(vendorDir, `${name}.js`),
  }))
  .sort((a, b) => b.find.length - a.find.length);

// @vitejs/plugin-react (vite:react-refresh) unconditionally adds
// react/react-dom/the jsx runtimes to `optimizeDeps.include` via its own
// `config` hook (see dev-vendor.mjs) -- Vite's config merge concatenates
// array fields like this, so passing `optimizeDeps.include: []` in our own
// config here has no effect; the plugin's entries survive the merge
// regardless. `configResolved` runs after that merge but before the
// dependency optimizer starts, so this plugin strips them back out of the
// final resolved config in place -- the alias above already covers every
// one of them with an already-ESM file, so the optimizer has nothing left
// to do for this app and never needs to touch esbuild's jail-broken
// resolver at all.
const vendoredSpecifiers = new Set(Object.values(VENDOR_SPECIFIERS));
const stripVendoredFromOptimizer = {
  name: "strip-vendored-deps-from-optimizer",
  configResolved(config) {
    config.optimizeDeps.include = (config.optimizeDeps.include ?? []).filter(
      (id) => !vendoredSpecifiers.has(id),
    );
  },
};

const server = await createServer({
  root,
  configFile: false,
  plugins: [react(), stripVendoredFromOptimizer],
  resolve: {
    alias: vendorAlias,
  },
  server: {
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${bardPort}`,
        changeOrigin: true,
      },
    },
  },
});

await server.listen();
server.printUrls();
