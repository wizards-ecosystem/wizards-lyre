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
| `scripts/lyre` | portable setup, launch, and test commands |
| `scripts/smoke-gpu.py` | Real ACE-Step turbo smoke (manual, not part of pytest) |

## One-folder setup

Install the small system prerequisites (`git`, `uv`, and Node.js/npm), clone this repository, then
run one command from its root:

```bash
./scripts/lyre bootstrap
```

It pins and downloads ACE-Step 1.5 into `vendor/ACE-Step-1.5`, creates the Python environment in
`.venv`, installs frontend dependencies in `web/node_modules`, builds the SPA, and downloads every
ACE-Step profile Lyre exposes into `checkpoints/` (Turbo, SFT, base, and XL Turbo). It is resumable:
re-running it reuses files already present in this checkout.

If you only need the web/API shell or want to defer the large model downloads, use
`./scripts/lyre install` first and `./scripts/lyre models` when ready.

The launcher redirects package-manager, Hugging Face, ModelScope, PyTorch/CUDA, compiler, and
temporary-file caches to this project. Runtime data remains under `projects/` and `output/`; no
Lyre-owned state needs to be stored in a home-directory cache. The NVIDIA driver and CUDA runtime
remain normal system prerequisites.

Use `./scripts/lyre paths` to show the exact locations. `vendor/`, models, environments, caches,
and generated music are intentionally ignored by Git. The `ACE_STEP_REVISION` file records the
pinned upstream source revision used by the installer.

For a GPU-free UI/API development session, set `LYRE_WORKER=mock` before starting
`./scripts/lyre worker`; it writes silent WAVs and never loads CUDA.

## Run the server + worker

Two separate processes (SPEC.md sec 5): the FastAPI server only ever reads/writes SQLite and
disk; the worker is where CUDA and ACE-Step actually load, so a GPU crash can't take HTTP down
and a long generation never blocks a request.

```bash
./scripts/lyre server  # terminal 1: HTTP API + built SPA
./scripts/lyre worker  # terminal 2: claims queued jobs one at a time
```

`server.app` binds `127.0.0.1:8421` by default; override with `LYRE_PORT`. If `web/dist` exists,
it serves the built SPA at `/`; otherwise `/` returns a hint to build or run the frontend dev
server. Jobs posted to `/api/projects/{id}/jobs` sit as `queued` until `worker.run_worker` (or
`LYRE_WORKER=mock` for a GPU-free worker) picks them up.

## Frontend

```bash
./scripts/lyre build-web  # writes web/dist, served by FastAPI
./scripts/lyre web        # Vite dev server with a /api proxy to LYRE_PORT
```

## Tests

```bash
./scripts/lyre test
```

Default pytest must not load CUDA or ACE-Step weights (it pins `LYRE_WORKER=mock`).

Frontend regression tests: `./scripts/lyre test-web` (Vitest + React Testing
Library against a mocked backend, no FastAPI/CUDA/ACE-Step required) — see
web/README.md's Tests section for details.

## GPU smoke (manual)

```bash
./scripts/lyre smoke-gpu
```

Loads ACE-Step turbo, generates ~10s of instrumental `text2music`, prints the output path, exits
0. Not part of `pytest`; requires a real GPU and installed weights.

## Location

Canonical checkout: `/home/limb06/wizards-lyre` (WSL Ubuntu, next to the other limb06 projects).
Studio data (`projects/`) lives in this tree and is gitignored.

Conclave on Windows cannot jail a WSL path, so the legacy
`config/projects/wizards-bard.yaml` remains `active: false`. Do not recreate a Windows clone
under `.projects/` unless you want Conclave to drive Lyre again.
