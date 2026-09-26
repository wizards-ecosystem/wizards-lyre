# Configuration

The Wizard's Lyre is configured by environment variables. There is no config file.
The one exception is the optional Remote GPU connection, which the UI saves
beside the jobs database because it changes while Lyre runs.

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
| `LYRE_WORKER` | `acestep` | Job backend. Set to `mock` for a GPU-free worker that writes silent WAVs; the whole UI and API work against it. Set to `remote` to render on a [Remote GPU](remote-gpu.md). |
| `LYRE_DEVICE` | `cuda` | Torch device string for the worker. |

## Remote GPU variables

Only read when you opt in to the [Remote GPU lane](remote-gpu.md). Set them in
the environment of `./scripts/lyre server`; the worker needs only
`LYRE_WORKER=remote`. `scripts/lyre` also loads plain `NAME=value` lines from a
gitignored `.env` in the checkout (never executed; the environment wins), which
is the place to keep `LYRE_RUNPOD_API_KEY` out of your shell history.

| Variable | Default | What it does |
|---|---|---|
| `LYRE_REMOTE_GPU_PROVISIONER` | `manual` | Who makes the GPU exist. `manual`: you run the host and enter its address. `runpod`: Lyre can start and stop a Runpod pod on your account. |
| `LYRE_REMOTE_GPU_STOP_ON_EXIT` | `true` | Terminate a rented pod when the server shuts down cleanly. A crash runs no shutdown code; see [runpod.md](runpod.md#stopping-it-is-your-job). |
| `LYRE_RUNPOD_API_KEY` | unset | Runpod API key with pod read/write. Stays on this machine; the pod never receives it. |
| `LYRE_RUNPOD_IMAGE` | published image for this checkout | Host image to run. Unset means `ghcr.io/wizards-ecosystem/wizards-lyre-remote-gpu:<build id>`. |
| `LYRE_RUNPOD_GPU_TYPE` | `NVIDIA GeForce RTX 4090` | Runpod GPU type id. 24 GB holds every profile. |
| `LYRE_RUNPOD_CLOUD` | `SECURE` | `SECURE` or `COMMUNITY`. |
| `LYRE_RUNPOD_DISK_GB` | `60` | Container disk. Weights are ~10 GB core, ~40 GB with every profile. |
| `LYRE_RUNPOD_NETWORK_VOLUME_ID` | unset | Keep weights on a network volume across pods. Pins pods to its data center. |
| `LYRE_RUNPOD_DATA_CENTER_IDS` | unset | Comma-separated data centers to allow. |

`HF_TOKEN`, if set, is passed to the pod for model downloads. ACE-Step's
weights are public, so it is optional.

These configure the host itself (`python -m worker.remote_host`, or the
container image, which sets sensible values):

| Variable | Default | What it does |
|---|---|---|
| `LYRE_REMOTE_SECRET` | none; required | Shared secret every request must carry. At least 24 random characters. |
| `LYRE_REMOTE_HOST` | `127.0.0.1` | Bind address. The container image sets it so the provider's proxy can reach the host. |
| `LYRE_REMOTE_PORT` | `8000` | Bind port. |
| `LYRE_REMOTE_ROOT` | `./.lyre-remote-gpu` | Weights, caches, and job scratch. Point it at a volume to keep weights. |
| `LYRE_REMOTE_BACKEND` | `acestep` | `mock` renders silence with no GPU, for smoke-testing a deployment. |

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
