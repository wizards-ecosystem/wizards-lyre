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
tests stay pytest at the repo root (SPEC.md §11).

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

- The Vitest config pins `pool: "forks"` with `singleFork: true`. Each test
  installs its own mock backend on the module-scope `fetch`, and `App` runs a
  5 s health poll, so parallel workers would interleave state across files.
- The waveform is a real wavesurfer.js instance with the regions plugin: it is
  remounted per selected take and owns both the repaint drag-selection and the
  plan's named section labels (SPEC.md §9.2).
