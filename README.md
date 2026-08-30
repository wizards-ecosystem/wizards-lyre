# Wizard's Lyre

A local generative music **studio** built on [ACE-Step 1.5][acestep]. Runs
entirely on your own GPU: no accounts, no API keys, no cloud music services,
and no network needed once the weights are on disk.

[![CI](https://github.com/wizards-ecosystem/wizards-lyre/actions/workflows/ci.yml/badge.svg)](https://github.com/wizards-ecosystem/wizards-lyre/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Lyre is deliberately **not** a one-click "type a prompt, get a song" page. It is
a studio you iterate in: a library of song projects, each with a plan, a rail of
takes, and a waveform you can select regions on. You throw a seed, listen, and
push it around with Cover, Repaint, Extract, Lego, and Complete — each of which
maps 1:1 onto an ACE-Step task. Every take is immutable and remembers its
parent, so the history is always walkable.

If you want a finished track from one prompt, this is the wrong tool.

## Requirements

- **An NVIDIA GPU.** 16 GB VRAM is the baseline the defaults are tuned for
  (2B turbo DiT + 1.7B LM). Less will work for the smaller profiles; the
  optional 4B `quality` profile needs CPU offload at 16 GB. LoRA training is
  sized for 24 GB.
- **Linux or WSL2**, with a working NVIDIA driver and CUDA runtime.
- **`git`, [`uv`][uv], and Node.js 20+.** Everything else installs into the
  checkout.
- **~25 GB of disk** for model weights, plus room for the audio you make.

You do **not** need a GPU to develop against the UI or API — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

```bash
git clone https://github.com/wizards-ecosystem/wizards-lyre.git
cd wizards-lyre
./scripts/lyre bootstrap
```

`bootstrap` pins and downloads ACE-Step 1.5 into `vendor/`, creates the Python
environment in `.venv`, installs and builds the frontend, and downloads every
model profile Lyre exposes into `checkpoints/`. It is resumable — re-running it
reuses whatever is already there.

To defer the large downloads, run `./scripts/lyre install` now and
`./scripts/lyre models` when you are ready.

Everything writable — weights, caches, environments, generated audio — stays
inside the checkout. `./scripts/lyre paths` prints exactly where. Nothing but
the NVIDIA driver itself is installed system-wide.

## Running

Two processes, in two terminals:

```bash
./scripts/lyre server   # HTTP API + the built SPA on 127.0.0.1:8421
./scripts/lyre worker   # claims queued jobs one at a time; this is where CUDA loads
```

Then open <http://127.0.0.1:8421>.

They are separate on purpose: the server only ever touches SQLite and disk, so
a GPU crash fails the job rather than taking the API down, and a long
generation never blocks an HTTP request. Jobs posted while no worker is running
simply wait as `queued`.

For a GPU-free session, start the worker as `LYRE_WORKER=mock ./scripts/lyre
worker` — it writes silent WAVs and never loads CUDA.

## What it does

All six phases of [SPEC.md](SPEC.md) §12 are implemented:

- **Generate** — Simple mode seeds a plan from a natural-language query using
  ACE-Step's 5 Hz LM; Custom mode is you writing caption, lyrics, BPM, key,
  time signature, and duration.
- **Cover / Repaint** — remix a take, or drag a region on the waveform and
  rewrite just that stretch.
- **Extract / Lego / Complete** — isolate a track, add or replace one, or fill
  out an arrangement. These swap in the base checkpoint, which the UI confirms
  first.
- **Style packs** — train a LoRA from 8+ of your own takes and apply it to
  later generations.
- **Studio ergonomics** — A/B compare, keyboard shortcuts, a live loudness
  meter, restoring an earlier take, and zip export you can drop into a DAW.
- **Library** — search, favorites, take notes, and drag-drop of a local
  WAV/MP3 as a source.

This describes what the code implements. Generation quality and training
throughput depend on ACE-Step and your hardware, and are not something this
repository's tests can speak to.

## Documentation

| | |
|---|---|
| [SPEC.md](SPEC.md) | The product spec. Authoritative on scope. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the three processes fit together, and why. |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every `LYRE_*` variable and the DiT profiles. |
| [docs/API.md](docs/API.md) | The HTTP surface. Live schema at `/docs` while running. |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Stuck jobs, VRAM, worker startup. |
| [docs/LIVE_STACK_TEST.md](docs/LIVE_STACK_TEST.md) | The end-to-end check against real weights. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the test loop, and what is out of scope. |
| [SECURITY.md](SECURITY.md) | The threat model, stated plainly. |

## Layout

| Path | Role |
|---|---|
| `server/` | FastAPI: HTTP, storage, job queue. Never imports ACE-Step. |
| `worker/` | The GPU process. `acestep_worker/` is real, `mock_worker.py` is for tests. |
| `web/` | Vite + React studio UI. |
| `tests/` | pytest against a mocked worker. No GPU. |
| `scripts/lyre` | Setup, run, test, lint. Run everything through it. |

## Development

```bash
./scripts/lyre test        # pytest, mocked worker, no GPU
./scripts/lyre test-web    # vitest + React Testing Library, mocked backend
./scripts/lyre lint        # ruff, mypy, tsc, eslint, prettier, shellcheck
./scripts/lyre smoke-gpu   # manual: one real ACE-Step generation
./scripts/lyre live-check  # manual: full end-to-end pass against a live server + worker
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## Security

Lyre binds `127.0.0.1` and has **no authentication by design**. Do not expose
it to a network or put it behind a reverse proxy. See
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE).

Built on [ACE-Step 1.5][acestep], which is also MIT-licensed. ACE-Step is not
vendored in this repository — `scripts/lyre` clones it at the revision pinned
in `ACE_STEP_REVISION`. Model weights are downloaded from their upstream
sources and carry their own terms.

[acestep]: https://github.com/ace-step/ACE-Step-1.5
[uv]: https://docs.astral.sh/uv/
