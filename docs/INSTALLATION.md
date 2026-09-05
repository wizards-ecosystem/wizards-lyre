# Installation

The Wizard's Lyre installs into either a slim release directory or a source
checkout. It does not need Docker, an account, an API key, or a system-wide
Python environment.

## Supported environment

| Component | Requirement |
|---|---|
| Operating system | Linux x86-64, or Windows 11 through WSL2 |
| GPU | NVIDIA CUDA GPU; 16 GB VRAM is the tuned baseline |
| Driver | A working NVIDIA driver visible to `nvidia-smi` |
| Python | 3.11 or 3.12; `uv` installs 3.12 locally when needed |
| Node.js | Not needed for a release bundle; 20.19+ or 22.12+ for a source checkout |
| Tools | `git` and [`uv`](https://docs.astral.sh/uv/); Node.js and npm only for source |
| Optional | FFmpeg, required only for training LoRA style packs; generation is unaffected |

Lyre's Bash launcher supports Windows through WSL2, using the same commands as
Linux. macOS and
non-NVIDIA accelerators are outside the product scope.

Before setup, confirm that `nvidia-smi` sees the GPU. Under WSL2, install the
Windows NVIDIA driver with WSL support; do not install a second Linux display
driver inside the distribution.

## Choose a model footprint

The Python/CUDA environment and package cache use roughly 10 GB of unique disk
space. Model sizes are approximate and downloads are resumable.

| Setup | Approx. models | Capabilities |
|---|---:|---|
| `models-core` | 10 GB | Generate, Cover, Repaint with the default `iterate` profile |
| `models-standard` | 20 GB | Core + `polish` + Extract, Lego, Complete, and style packs |
| `models` | 40 GB | Standard + optional 4B `quality` profile |
| `bootstrap` | 40 GB | Full install and every model in one command |

Allow at least 55 GB free for a full environment, package cache, and all models,
plus additional space for projects and exported audio. These figures were
measured against the pinned ACE-Step revision; upstream artifact layouts can
change when that pin is updated.

## Install a release bundle (recommended)

Open the [latest release](https://github.com/wizards-ecosystem/wizards-lyre/releases/latest)
and download either `wizards-lyre-*-linux-x86_64.tar.gz` or the equivalent
`.zip`. Download `SHA256SUMS` alongside it, then verify from the download
directory:

```bash
sha256sum --ignore-missing --check SHA256SUMS
```

The checksum file covers both archive formats, so a missing format may be
reported; the archive you downloaded must report `OK`. You can additionally
verify its signed build provenance with the
[GitHub CLI](https://cli.github.com/):

```bash
gh attestation verify wizards-lyre-*-linux-x86_64.tar.gz \
  --repo wizards-ecosystem/wizards-lyre
```

Extract the archive into a durable location, enter its top-level directory,
and choose one setup path:

```bash
# One command: dependencies and every model profile.
./scripts/lyre bootstrap

# Or start smaller with only the default generation profile.
./scripts/lyre install
./scripts/lyre models-core
```

The bundle includes a compiled web app, runtime code, license/notices, locked
install metadata, and a file-hash manifest. It excludes source UI
files, tests, contributor tooling, model weights, caches, and user data. If an
unzip tool loses the executable bit, restore it with `chmod +x scripts/lyre`.

## Install from source

Install Node.js 20.19+ or 22.12+, clone the repository, then choose one path:

```bash
git clone https://github.com/wizards-ecosystem/wizards-lyre.git
cd wizards-lyre

# One command: dependencies, production UI, and every model.
./scripts/lyre bootstrap
```

For a smaller source install:

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

Warnings about models only disable the profiles they name. Missing FFmpeg stops
style-pack training; generation remains available. Developers without a GPU can
use the mock worker described in [CONTRIBUTING.md](../CONTRIBUTING.md).

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

Always stop both processes and back up `projects/` first.

For a release bundle, download and verify the new archive, extract it into a
new directory, run `./scripts/lyre install`, then copy the backed-up `projects/`
directory into it. Keep the old directory until the updated install opens your
library correctly.

For a source checkout, read [CHANGELOG.md](../CHANGELOG.md), update the checkout,
and reconcile the locked dependencies:

```bash
git pull --ff-only
./scripts/lyre install
```

Run `./scripts/lyre models` only when the changelog or upstream revision notes
say new weights are needed. See [Troubleshooting](TROUBLESHOOTING.md) if any
readiness check fails.
