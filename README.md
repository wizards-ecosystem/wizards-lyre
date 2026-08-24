# Wizard's Lyre

Local generative music studio. ACE-Step 1.5 on the GPU. No cloud music APIs.

The product spec is **[SPEC.md](SPEC.md)**. Implement that file. Do not invent extra engines.

## Status

All six phases in SPEC.md sec 12 are implemented:

- **Phase 1 (scaffold + generate):** FastAPI health/projects/plan/takes/jobs API, a SQLite
  `queued -> running -> done|error` job queue, a dedicated worker process (`worker/run_worker.py`)
  that drains it, a React shell, and a mocked worker for tests. The production job backend is
  `worker/acestep_worker.py`, which calls ACE-Step 1.5's `generate_music` -- it requires ACE-Step
  installed and weights downloaded (see below).
- **Phase 2 (studio loop):** project library, the plan editor (simple + custom), waveform + region
  display/select, `cover` (take source select + strength), and `repaint` (take/region source, no
  strength control), reusing generate's plan-save/poll UX.
- **Phase 3 (base swap):** worker unload/load, `extract` / `lego` / `complete` job types, and a
  confirmation UI for the `studio_ops` base-model swap.
- **Phase 4 (polish):** a quality score on takes, LRC output when ACE-Step supplies timestamps,
  and a LoRA train/load path (style-pack UI to train from selected takes, then load a trained
  LoRA into generation).
- **Phase 5 (studio ergonomics):** A/B take compare, project export as a zip (`project.json`,
  `plan.json`, active mix, optional stems), keyboard shortcuts (generate, play/pause, next/prev
  take, save plan), and a UI to walk `parent_take_id` and restore an earlier take.
- **Phase 6 (library and ingest):** project search, favorites for projects/takes, free-text take
  notes, a loudness/peak meter on the player, and drag-drop of a local WAV/MP3 as a cover/repaint
  source.

This describes what the code implements, not verified GPU performance -- generation quality, LoRA
training throughput, etc. depend on ACE-Step and real hardware and are not something this repo's
test suite exercises.

## Machine

WSL Ubuntu (`limb06`), RTX 4070 Ti SUPER 16 GB. Checkout: `/home/limb06/wizards-lyre`.
Default: ACE-Step 2B turbo + 1.7B LM. Bind `127.0.0.1:8421`.

## Layout

| Path | Role |
|---|---|
| `SPEC.md` | Sole product spec |
| `server/` | FastAPI (HTTP, jobs, files) |
| `worker/` | `run_worker.py` (dedicated process entry point), `acestep_worker.py` (real backend), test-only `mock_worker.py` |
| `web/` | Vite + React studio |
| `tests/` | pytest, mocked worker, no GPU |
| `scripts/smoke-gpu.py` | Real ACE-Step turbo smoke (manual, not part of pytest) |

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

`pip install -e ".[dev]"` above only installs this repo's own dependencies (FastAPI, uvicorn,
pytest) -- it does **not** install ACE-Step itself. ACE-Step 1.5 is a separate package (SPEC.md
sec 4, sec 13) that `worker/acestep_worker.py` imports lazily, so it's only required on the
machine that runs `worker.run_worker`, not on a server-only box. Install it before downloading
weights:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5 ../ACE-Step-1.5
( cd ../ACE-Step-1.5 && uv pip install -e . )  # follow docs/en/INSTALL.md there if this has drifted
```

Once `acestep` is importable and its download entry point is on PATH, pull the turbo checkpoint +
the default 1.7B 5Hz LM once. The exact command name comes from
`ACE-Step-1.5/docs/en/INSTALL.md`; if you installed ACE-Step with `uv` instead of `pip`, run it as
`uv run acestep-download` there instead. Run it from **this repo's root** (not the `ACE-Step-1.5`
checkout) -- `worker/acestep_worker.py`'s `CHECKPOINTS_ROOT` resolves the relative default
`checkpoints/` from wherever `worker.run_worker` is launched, and `acestep-download` writes to a
`checkpoints/` under its own current directory the same way, so the two must be run from the same
place or the worker won't find what was downloaded:

```bash
acestep-download
```

Weights land under `checkpoints/` by default, relative to the directory `acestep-download` above
was run from (override the worker's read side with `BARD_CHECKPOINTS_DIR` if you'd rather point it
at weights downloaded elsewhere, e.g. the `ACE-Step-1.5` checkout's own `checkpoints/`) -- see
`worker/acestep_worker.py`'s `CHECKPOINTS_ROOT`.

Without ACE-Step installed, the server still runs; job types that need it (`generate`, `cover`,
`repaint`, `extract`, `lego`, `complete` -- see Status above) will fail with a clear "acestep is
not installed" error instead of crashing. Set `BARD_WORKER=mock` to force the mocked worker
(silent WAV, no GPU) for local UI/API poking without a GPU.

## Run the server + worker

Two separate processes (SPEC.md sec 5): the FastAPI server only ever reads/writes SQLite and
disk; the worker is where CUDA and ACE-Step actually load, so a GPU crash can't take HTTP down
and a long generation never blocks a request.

```bash
python -m server.app          # terminal 1: HTTP API + (if built) the SPA
python -m worker.run_worker   # terminal 2: claims `queued` jobs, runs them one at a time
```

`server.app` binds `127.0.0.1:8421` by default; override with `BARD_PORT`. If `web/dist` exists,
it serves the built SPA at `/`; otherwise `/` returns a hint to build or run the frontend dev
server. Jobs posted to `/api/projects/{id}/jobs` sit as `queued` until `worker.run_worker` (or
`BARD_WORKER=mock` for a GPU-free worker) picks them up.

## Frontend

```bash
cd web
npm install
npm run build     # writes web/dist, served by the FastAPI app above
npm run dev        # Vite dev server with a /api proxy to BARD_PORT (default 8421)
```

## Tests

```bash
pytest
```

Default pytest must not load CUDA or ACE-Step weights (it pins `BARD_WORKER=mock`).

Frontend regression tests: `cd web && npm test` (Vitest + React Testing
Library against a mocked backend, no FastAPI/CUDA/ACE-Step required) — see
web/README.md's Tests section for details.

## GPU smoke (manual)

```bash
python scripts/smoke-gpu.py
```

Loads ACE-Step turbo, generates ~10s of instrumental `text2music`, prints the output path, exits
0. Not part of `pytest`; requires a real GPU and installed weights.

## Location

Canonical checkout: `/home/limb06/wizards-lyre` (WSL Ubuntu, next to the other limb06 projects).
Studio data (`projects/`) lives in this tree and is gitignored.

Conclave on Windows cannot jail a WSL path, so the legacy
`config/projects/wizards-bard.yaml` remains `active: false`. Do not recreate a Windows clone
under `.projects/` unless you want Conclave to drive Lyre again.
