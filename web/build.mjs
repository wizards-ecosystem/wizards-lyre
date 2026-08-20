// Inline Vite build entry point.
//
// `vite build` normally bundles vite.config.* with esbuild, which walks
// parent directories looking for node_modules / package.json. Under the
// Conclave NTFS jail, some ancestor directories (e.g. Documents) exist but
// deny enumeration outright (EACCES, not ENOENT), and esbuild's walk-up
// does not tolerate that -- it aborts instead of continuing past a denied
// directory. Loading the config as inline JS (configFile: false) skips
// that walk entirely, so this script is the supported way to build here.
import { fileURLToPath } from "node:url";
import path from "node:path";
import { build } from "vite";
import react from "@vitejs/plugin-react";

const root = path.dirname(fileURLToPath(import.meta.url));
const bardPort = process.env.BARD_PORT ?? "8421";

await build({
  root,
  configFile: false,
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${bardPort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
