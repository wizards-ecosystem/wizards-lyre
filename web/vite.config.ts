import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI backend so the SPA can use
// same-origin paths in both dev and prod (SPEC.md sec 5: production serves
// the built bundle from FastAPI itself).
const lyrePort = process.env.LYRE_PORT ?? "8421";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${lyrePort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
