// Inline Vite dev-server entry point. See build.mjs for why this project
// loads its Vite config as inline JS (configFile: false) instead of letting
// the CLI auto-load vite.config.js.
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const root = path.dirname(fileURLToPath(import.meta.url));
const bardPort = process.env.BARD_PORT ?? "8421";

const server = await createServer({
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
});

await server.listen();
server.printUrls();
