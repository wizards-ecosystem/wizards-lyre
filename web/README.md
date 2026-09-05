# The Wizard's Lyre web app

Lyre's Vite + React + TypeScript SPA implements SPEC.md section 9. Its light,
tactile workspace contains the project library, composition plan, takes, and
waveform. It has no accounts or marketing shell.

## Setup

```
./scripts/lyre install
```

Run this once from the repository root. It installs `web/node_modules` locally
and keeps npm's cache inside the repository. Node.js 20.19+ or 22.12+ is
required. Model weights and a GPU are unnecessary for frontend work.

## Dev server

```
./scripts/lyre web
```

Starts a Vite dev server on `http://localhost:5173`. Requests to `/api/*`
are proxied to the FastAPI backend at `http://127.0.0.1:8421` (override with
`LYRE_PORT`). The dev server works standalone even if the backend isn't
running yet. API calls will fail (shown as "server offline" / error
banners in the UI) until `server/` is up.

## Build

```
./scripts/lyre build-web
```

Type-checks with `tsc -b` and produces a production bundle in `web/dist/`,
which FastAPI serves directly in production (SPEC.md section 5).

## Tests

```
./scripts/lyre test-web
```

Frontend regression tests (Vitest + React Testing Library + jsdom), covering
the studio's core flows (generate/cover/repaint, extract/lego/complete,
library management, take annotations, plan editing, LoRA train/load, and
more) against a mocked fetch backend in `src/test/mockServer.ts`. They require
no FastAPI, CUDA, ACE-Step, credentials, or generated audio. Python-side tests
stay in pytest at the repo root (SPEC.md section 11).

## Keyboard shortcuts

Active whenever a project is open. `g` / `Space` / `Up` / `Down` are disabled
while focus is in a text field (the query/caption/lyrics/track-name inputs)
so ordinary characters still work there; `Ctrl+S` / `Cmd+S` works everywhere,
including while typing, since that's when saving matters most. There's no
command palette, so this list and the `title` tooltip on the Generate button
record every shortcut:

- `g`: Generate (disabled while typing in a text field)
- `Space`: play/pause the selected take (disabled while typing in a text field)
- `Up` / `Down`: select the previous/next take, newest-first (disabled while typing in a text field)
- `Ctrl+S` / `Cmd+S`: save the plan immediately instead of waiting out the debounce (works everywhere, including text fields)

## Notes

- The Vitest config uses the forks pool with one worker and file parallelism
  disabled. Each test installs its own mock backend on the module-scope
  `fetch`, and `App` runs a 5 s health poll, so parallel workers would
  interleave state across files.
- The waveform is a real wavesurfer.js instance with the regions plugin: it is
  remounted per selected take and owns both the repaint drag-selection and the
  plan's named section labels (SPEC.md section 9.2).
