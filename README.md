# Wizard's Bard

Local generative music studio. ACE-Step 1.5 on the GPU. No cloud music APIs.

The product spec is **[SPEC.md](SPEC.md)**. Implement that file. Do not invent extra engines.

## Status

Phase 1 (SPEC.md sec 12): FastAPI health/projects/plan/takes/jobs API, a SQLite
`queued -> running -> done|error` job queue, a dedicated worker process (`worker/run_worker.py`)
that drains it, a minimal React shell (library, plan, takes, generate), and a mocked worker for
tests. The production job backend is `worker/acestep_worker.py`, which calls ACE-Step 1.5's
`generate_music` -- it requires ACE-Step installed and weights downloaded (see below). Phases
2-4 (studio loop, base-model swap, LoRA/polish) are not implemented yet.

## Machine

Windows, RTX 4070 Ti SUPER 16 GB. Default: ACE-Step 2B turbo + 1.7B LM. Bind `127.0.0.1:8421`.

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

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Download ACE-Step 1.5 (turbo checkpoint + the default 1.7B 5Hz LM) once, per upstream's install
docs (SPEC.md sec 4, sec 13):

```powershell
uv run acestep-download   # or the equivalent from ACE-Step-1.5/docs/en/INSTALL.md
```

Without ACE-Step installed, the server still runs; `generate`/`cover`/... jobs will fail with a
clear "acestep is not installed" error instead of crashing. Set `BARD_WORKER=mock` to force the
mocked worker (silent WAV, no GPU) for local UI/API poking without a GPU.

## Run the server + worker

Two separate processes (SPEC.md sec 5): the FastAPI server only ever reads/writes SQLite and
disk; the worker is where CUDA and ACE-Step actually load, so a GPU crash can't take HTTP down
and a long generation never blocks a request.

```powershell
python -m server.app          # terminal 1: HTTP API + (if built) the SPA
python -m worker.run_worker   # terminal 2: claims `queued` jobs, runs them one at a time
```

`server.app` binds `127.0.0.1:8421` by default; override with `BARD_PORT`. If `web/dist` exists,
it serves the built SPA at `/`; otherwise `/` returns a hint to build or run the frontend dev
server. Jobs posted to `/api/projects/{id}/jobs` sit as `queued` until `worker.run_worker` (or
`BARD_WORKER=mock` for a GPU-free worker) picks them up.

## Frontend

```powershell
cd web
npm install
npm run build     # writes web/dist, served by the FastAPI app above
npm run dev        # Vite dev server with a /api proxy to BARD_PORT (default 8421)
```

## Tests

```powershell
pytest
```

Default pytest must not load CUDA or ACE-Step weights (it pins `BARD_WORKER=mock`).

## GPU smoke (manual)

```powershell
python scripts/smoke-gpu.py
```

Loads ACE-Step turbo, generates ~10s of instrumental `text2music`, prints the output path, exits
0. Not part of `pytest`; requires a real GPU and installed weights.

## Conclave / jail

The canonical clone lives at `wizards-conclave/.projects/wizards-bard`. That `.projects/` folder
is the NTFS jail (`jail.root`). Conclave jobs work in their own worktree under
`.projects/.conclave-wt/<job>` and merge back into that clone. Agents may write anything in
`.projects`; they must not write Conclave source, `.env`, or `.conclave`.

Do not recreate `C:/Users/isaac/Documents/wizards-bard`. Conclave skips any `repo:` path outside
`.projects/`.

Jail setup (once, elevated, from the Conclave repo):

```powershell
cd C:\Users\isaac\Documents\wizards-conclave
.\scripts\setup-windows-jail.cmd
.\scripts\wz.ps1 doctor
```

Doctor must show PASS for `jail .projects`. Re-run setup after `pnpm install`. `config/projects/bard.yaml` is enabled; goals are implement SPEC.md in phase order.
