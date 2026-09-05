# Installation

Wizard's Lyre installs into a source checkout. It does not need Docker, an
account, an API key, or a system-wide Python environment.

## Supported environment

| Component | Requirement |
|---|---|
| Operating system | Linux x86-64, or Windows 11 through WSL2 |
| GPU | NVIDIA CUDA GPU; 16 GB VRAM is the tuned baseline |
| Driver | A working NVIDIA driver visible to `nvidia-smi` |
| Python | 3.11 or 3.12; `uv` installs 3.12 locally when needed |
| Node.js | 20.19+ or 22.12+ |
| Tools | `git`, [`uv`](https://docs.astral.sh/uv/), Node.js and npm |
| Optional | FFmpeg, required only for training LoRA style packs |

Native Windows is not currently supported by Lyre's Bash launcher. WSL2 is the
recommended Windows path and uses the same commands as Linux. macOS and
non-NVIDIA accelerators are outside the product scope.

Before setup, confirm that `nvidia-smi` sees the GPU. Under WSL2, install the
Windows NVIDIA driver with WSL support; do not install a second Linux display
driver inside the distribution.

## Choose a model footprint

The Python/CUDA environment uses roughly 8 GB. Model sizes are approximate and
downloads are resumable.

| Setup | Approx. models | Capabilities |
|---|---:|---|
| `models-core` | 9 GB | Generate, Cover, Repaint with the default `iterate` profile |
| `models-standard` | 19 GB | Core + `polish` + Extract, Lego, Complete, and style packs |
| `models` | 28 GB | Standard + optional 4B `quality` profile |
| `bootstrap` | 28 GB | Full install and every model in one command |

Allow about 40 GB free for a full environment plus all models, and additional
space for projects and exported audio.

## Install

Clone the repository, then choose one path:

```bash
git clone https://github.com/wizards-ecosystem/wizards-lyre.git
cd wizards-lyre

# One command: dependencies, production UI, and every model.
./scripts/lyre bootstrap
```

For a smaller first install:

```bash
./scripts/lyre install       # dependencies + production UI, no weights
./scripts/lyre models-core   # default generation workflow only
```

You can add the remaining profiles later with `models-standard` or `models`.
Re-running any setup command is safe: downloads and package caches are reused.

Check the result:

```bash
./scripts/lyre doctor
```

Warnings about models only disable the profiles they name. A missing FFmpeg
warning affects style-pack training, not generation. Developers without a GPU
can use the mock worker described in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Run

Start the server and worker in separate terminals:

```bash
./scripts/lyre server
```

```bash
./scripts/lyre worker
```

Open <http://127.0.0.1:8421>. Jobs wait safely in SQLite if the worker is not
running. Stop each process with `Ctrl+C`.

## Files, network, and backups

`./scripts/lyre paths` prints every writable location. Environments, package
caches, upstream source, weights, generated projects, and exports stay inside
the checkout and are ignored by Git.

Setup requires network access to clone ACE-Step and download packages and
weights. Generation and editing make no network calls after those files exist.

Back up `projects/` to preserve songs, takes, uploads, style packs, and the job
database. Removing the checkout also removes all of that local data unless you
back it up first.

## Updating

Read [CHANGELOG.md](../CHANGELOG.md), stop both processes, back up `projects/`,
then update the checkout and reconcile the locked dependencies:

```bash
git pull --ff-only
./scripts/lyre install
```

Run `./scripts/lyre models` only when the changelog or upstream revision notes
say new weights are needed. See [Troubleshooting](TROUBLESHOOTING.md) if any
readiness check fails.
