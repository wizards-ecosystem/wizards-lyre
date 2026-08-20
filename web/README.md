# web

Vite + React + TypeScript SPA for Wizard's Bard (SPEC.md §9). Dark, dense,
local-tool aesthetic — a project library plus a three-pane project workspace
(Plan / Takes / Waveform). No accounts, no marketing shell.

## Setup

```
cd web
npm install
```

## Dev server

```
npm run dev
```

Starts a Vite dev server on `http://localhost:5173`. Requests to `/api/*`
are proxied to the FastAPI backend at `http://127.0.0.1:8421` (override with
`BARD_PORT`). The dev server works standalone even if the backend isn't
running yet — API calls will just fail (shown as "server offline" / error
banners in the UI) until `server/` is up.

## Build

```
npm run build
```

Type-checks with `tsc -b` and produces a production bundle in `web/dist/`,
which FastAPI serves directly in prod (SPEC.md §5).

## Notes

- `dev.mjs` / `build.mjs` load the Vite config as inline JS instead of a
  `vite.config.ts` file, and `dev-vendor.mjs` pre-bundles React via Vite's
  Rollup-based production build rather than the esbuild dev optimizer — both
  work around an `EACCES` esbuild hits when it walks up through a
  permission-denied ancestor directory under the Conclave worktree jail.
  Functionally this is the same as a standard `vite.config.ts` with an
  `/api` proxy; see the comments in those files for the full story.
- Waveform integration (wavesurfer.js) is a later phase — the Waveform pane
  is currently a placeholder container plus the action buttons.
