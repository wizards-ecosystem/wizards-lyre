# Troubleshooting

Start with the readiness check; it catches missing tools, an incomplete local
install, an unbuilt UI, absent model profiles, GPU visibility, and FFmpeg:

```bash
./scripts/lyre doctor
```

For setup and disk requirements, see [Installation](INSTALLATION.md).

## Jobs sit at `queued` and never run

The server only *enqueues* jobs; a separate worker process runs them
(see [ARCHITECTURE.md](ARCHITECTURE.md)). Almost always, the worker is not
running.

```bash
curl -s http://127.0.0.1:8421/api/health
```

- `"worker backend: acestep (not reported yet -- is worker.run_worker running?)"`:
  no worker has ever checked in. Start one: `./scripts/lyre worker`.
- `"unavailable: worker heartbeat stale since ..."`: a worker ran and then
  died or hung. Check its terminal for a traceback and restart it.
- `"unavailable: ..."` with an ACE-Step message: the worker started but could
  not load models. See the next section.

Jobs left `running` by a crashed worker are requeued automatically when a
worker next starts, and marked `error` after three attempts rather than
retrying forever.

## The worker will not start

**`WorkerUnavailable: ACE-Step is not installed`:** run
`./scripts/lyre install` (or `bootstrap`). The launcher installs ACE-Step into
the same `.venv` used by the server.

**Weights are missing:** run `./scripts/lyre models`. It is resumable; re-run
it after an interrupted download. If you only need default Generate, Cover,
and Repaint, `./scripts/lyre models-core` is the smaller download.
`./scripts/lyre paths` shows where the files land.

**`LYRE_CHECKPOINTS_DIR ... must be a directory named 'checkpoints'`:** upstream
ACE-Step resolves weights as
`<project_root>/checkpoints/<name>`, so a directory named anything else can
never satisfy that convention. The worker reports the mismatch. Rename the
directory or leave the default.

**No GPU at all?** You do not need one to work on the UI or API:

```bash
LYRE_WORKER=mock ./scripts/lyre worker
```

It writes silent WAVs, never imports CUDA, and exercises the same job contract.

## Out of memory during generation

One DiT plus one LM run at a time, but the profiles differ a
lot in appetite (see [CONFIGURATION.md](CONFIGURATION.md)):

- `quality` (XL, 4B) needs CPU offload on a 16 GB card. If the worker reports
  it cannot load XL with offload, the server rejects `quality` jobs up front.
- `polish` and `studio_ops` run 50 steps and are much slower than `iterate`'s 8.
- LoRA training targets the base checkpoint and is the app's heaviest task.
  SPEC.md sizes it at "3090-class" (24 GB).

Drop to `iterate` and confirm generation works at all before chasing a
configuration problem.

## Style-pack training fails with "labeled 0/8 staged source files"

FFmpeg is missing. ACE-Step's dataset builder decodes training audio through
torchcodec, which loads FFmpeg's shared libraries at import; the worker log
will show something like `libavutil.so.56: cannot open shared object file`.

```bash
sudo apt install ffmpeg      # or your platform's equivalent
```

Missing FFmpeg stops style-pack LoRA training; generation is unaffected because
it falls back to soundfile. A machine can therefore generate takes while the
training dataset builder labels zero files. If the count is zero, install
FFmpeg before investigating the takes.

## Switching to Extract / Lego / Complete is slow

Expected. Those three require the `studio_ops` base checkpoint, so the worker
unloads the current DiT and loads another. The UI asks you to confirm the swap
and shows "loading base model" while it happens.

## The browser shows a hint instead of the app

`http://127.0.0.1:8421/` returns a JSON hint when `web/dist` does not exist.
Build it:

```bash
./scripts/lyre build-web
```

Or run the dev server with hot reload, which proxies `/api` to the backend:

```bash
./scripts/lyre web     # http://localhost:5173
```

**Node.js is too old:** the current Vite toolchain needs Node.js 20.19+ or
22.12+. Upgrade Node, then re-run `./scripts/lyre install`.

## The UI says "server offline"

The SPA is reachable but `/api/health` is not. Either the FastAPI server is not
running (`./scripts/lyre server`), or you changed `LYRE_PORT` for the server
but left the dev server unchanged. The Vite proxy reads the same variable, so
export it for both.

## `./scripts/lyre: Permission denied`

`chmod +x scripts/lyre`. This affected clones made before 0.1.0, where the
launcher was recorded in git without its executable bit.

## Tests

**`pytest` tries to import `acestep` or CUDA`:** it should never do that. Tests
pin `LYRE_WORKER=mock`. If you hit this, something imported `acestep` at module
scope; the rule is that only `worker/acestep_worker/` may touch it, and only
lazily inside functions.

**A test passes but does not seem to test anything:** if it monkeypatches
something in `server.storage`, `server.jobs`, or `worker.acestep_worker`, check
that it patches the *defining module* rather than the package re-export. See
[CONTRIBUTING.md](../CONTRIBUTING.md#things-worth-knowing-before-you-edit).

## Something else

Open an issue with the output of `./scripts/lyre paths`, your GPU and VRAM,
your OS, and the worker's terminal output.
