# Configuration

The Wizard's Lyre is configured by environment variables. There is no config file
and no settings UI.

You normally do not need to set any of these: `scripts/lyre` exports every one
of them to keep all writable state inside the checkout. Run
`./scripts/lyre paths` to see exactly where things land.

## The Wizard's Lyre variables

| Variable | Default | What it does |
|---|---|---|
| `LYRE_PORT` | `8421` | Port for the HTTP server. The host is fixed at `127.0.0.1`; see [SECURITY.md](../SECURITY.md). |
| `LYRE_PROJECTS_DIR` | `<repo>/projects` | Where project directories and their takes live. |
| `LYRE_OUTPUT_DIR` | `<repo>/output` | Scratch output root (the GPU smoke test writes here). |
| `LYRE_DB_PATH` | `<projects dir>/lyre.db` | The SQLite job queue. |
| `LYRE_CHECKPOINTS_DIR` | `<repo>/checkpoints` | ACE-Step model weights. Must be a directory named `checkpoints`. Upstream resolves weights as `<parent>/checkpoints/<name>`, and the worker reports a mismatched path. |
| `LYRE_WORKER` | `acestep` | Job backend. Set to `mock` for a GPU-free worker that writes silent WAVs; the whole UI and API work against it. |
| `LYRE_DEVICE` | `cuda` | Torch device string for the worker. |

## Variables the launcher sets for you

`scripts/lyre` also redirects every third-party cache into the checkout, so
running Lyre does not scatter gigabytes through your home directory:
`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `npm_config_cache`, `XDG_CACHE_HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`,
`MODELSCOPE_CACHE`, `TORCH_HOME`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR`,
`PYTHONPYCACHEPREFIX`, and `TMPDIR`.

The NVIDIA driver and CUDA runtime remain ordinary system prerequisites.

If you invoke `python -m server.app` or `python -m worker.run_worker` directly
instead of through the launcher, none of this redirection happens and caches go
to their usual system locations. That works fine; it is just less tidy.

## DiT profiles

Not an environment variable, but the other main knob. Each project has a
`dit_profile`, and a job may override it (SPEC.md section 4.1):

| Profile | Checkpoint | Steps | Use |
|---|---|---|---|
| `iterate` | `acestep-v15-turbo` (2B) | 8 | The default. Daily generate, cover, repaint. |
| `polish` | `acestep-v15-sft` (2B) | 50 | More prompt adherence and detail. |
| `quality` | `acestep-v15-xl-turbo` (4B) | 8 | Optional; needs CPU offload on a 16 GB card. |
| `studio_ops` | `acestep-v15-base` (2B) | 50 | Required for extract/lego/complete, and used for LoRA training. |

Switching between profiles unloads the previous DiT before loading the next:
one GPU occupant at a time, and jobs serialize.
