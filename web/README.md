# web

Vite + React + TypeScript SPA for Wizard's Lyre (SPEC.md §9). Dark, dense,
local-tool aesthetic — a project library plus a three-pane project workspace
(Plan / Takes / Waveform). No accounts, no marketing shell.

## Setup

```
./scripts/lyre bootstrap
```

Run this once from the repository root. It installs `web/node_modules` locally
and keeps npm's cache inside the repository.

## Dev server

```
./scripts/lyre web
```

Starts a Vite dev server on `http://localhost:5173`. Requests to `/api/*`
are proxied to the FastAPI backend at `http://127.0.0.1:8421` (override with
`LYRE_PORT`). The dev server works standalone even if the backend isn't
running yet — API calls will just fail (shown as "server offline" / error
banners in the UI) until `server/` is up.

## Build

```
./scripts/lyre build-web
```

Type-checks with `tsc -b` and produces a production bundle in `web/dist/`,
which FastAPI serves directly in prod (SPEC.md §5).

## Tests

```
./scripts/lyre test-web
```

Frontend regression tests (Vitest + React Testing Library + jsdom), covering
the studio's core flows (generate/cover/repaint, extract/lego/complete,
library management, take annotations, plan editing, LoRA train/load, and
more) against a mocked fetch backend in `src/test/mockServer.ts` — no
FastAPI, CUDA, ACE-Step, credentials, or generated audio required. Python-side
tests stay pytest at the repo root
(SPEC.md §11). Like `dev.mjs` / `build.mjs`, the Vitest config lives inline
in `test.mjs` instead of a `vitest.config.ts`; see the Notes section for why.

## Keyboard shortcuts

Active whenever a project is open. `g` / `Space` / `↑` / `↓` are disabled
while focus is in a text field (the query/caption/lyrics/track-name inputs)
so ordinary characters still work there; `Ctrl+S` / `Cmd+S` works everywhere,
including while typing, since that's when saving matters most. There's no
command palette or help screen, so this list — and the `title` tooltip on the
Generate button — is the only place they're documented:

- `g` — Generate (disabled while typing in a text field)
- `Space` — play/pause the selected take (disabled while typing in a text field)
- `↑` / `↓` — select the previous/next take, newest-first (disabled while typing in a text field)
- `Ctrl+S` / `Cmd+S` — save the plan immediately instead of waiting out the debounce (works everywhere, including text fields)

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
