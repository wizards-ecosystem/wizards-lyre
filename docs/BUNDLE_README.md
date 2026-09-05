# Wizard's Lyre @VERSION@

This is the slim, runnable Linux/WSL2 release of Wizard's Lyre. It includes the
prebuilt studio interface, so Node.js and npm are **not** required.

## What you need

- Linux x86-64, or Windows 11 using WSL2
- an NVIDIA CUDA GPU; 16 GB VRAM is the tuned baseline
- `git` and [`uv`](https://docs.astral.sh/uv/)
- about 20 GB free for the environment and default models, or at least 55 GB
  for every model profile
- optionally, FFmpeg for style-pack training

Confirm that `nvidia-smi` sees your GPU before setup. Native Windows, macOS, and
non-NVIDIA accelerators are not currently supported.

## Install

Extract this archive into a durable location. Everything—including projects,
models, package caches, and the Python environment—stays inside that directory.
Then run:

```bash
./scripts/lyre install
./scripts/lyre models-core
./scripts/lyre doctor
```

`models-core` downloads roughly 10 GB and enables Generate, Cover, and Repaint.
Use `models-standard` for all profiles except optional XL quality, or `models`
for every profile. `bootstrap` is the one-command equivalent of a full install
plus `models`.

Setup downloads the pinned ACE-Step source, Python packages, and model weights.
Downloads are resumable. Runtime generation makes no cloud music API calls.

## Run

Start these in separate terminals from the extracted directory:

```bash
./scripts/lyre server
```

```bash
./scripts/lyre worker
```

Open <http://127.0.0.1:8421>. Stop each process with `Ctrl+C`.

Lyre intentionally binds only to `127.0.0.1` and has no authentication. Do not
expose it through a reverse proxy, public bind, or port forwarding.

## Your files and updates

`./scripts/lyre paths` shows every writable location. Back up `projects/` to
preserve songs, uploads, takes, style packs, and the job database. Removing this
directory removes that data unless you backed it up first.

To update, download the new release, stop Lyre, and copy your backed-up
`projects/` directory into the new extracted directory. Do not replace a
working install in place before taking a backup.

For help, diagnostics, and full documentation, visit the
[project repository](https://github.com/wizards-ecosystem/wizards-lyre). Include
the output of `./scripts/lyre doctor` when asking for setup support.

## Build identity

- Wizard's Lyre: `@VERSION@`
- Source revision: `@SOURCE_REVISION@`
- ACE-Step revision: `@ACE_REVISION@`

`LYRE_RELEASE.json` contains the same identity plus hashes for every runtime
file. Verify the archive itself with the accompanying `SHA256SUMS` file and the
GitHub artifact attestation on the release page.

Wizard's Lyre is provided under the [MIT License](LICENSE). ACE-Step source,
model weights, and dependencies are fetched separately and retain their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
