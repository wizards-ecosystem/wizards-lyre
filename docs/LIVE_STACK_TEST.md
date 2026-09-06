# Live stack test

Everything in `pytest` runs against a mocked worker and never loads a GPU
(SPEC.md section 11). This keeps CI fast and contribution GPU-free, but leaves
the integration between the server, worker, and real weights unverified.

`./scripts/lyre live-check` is that layer. It drives the actual HTTP API
against a running server and a running worker, and asserts on what comes back.

It complements `./scripts/lyre smoke-gpu`, which calls `run_job` directly and
answers only "can this machine generate audio at all".

## Running it

Three terminals, from the repository root:

```bash
./scripts/lyre server        # 1
./scripts/lyre worker        # 2: real backend, without LYRE_WORKER=mock
./scripts/lyre live-check    # 3
```

Expect **10-25 minutes** for the default stages on a 16 GB card. Most of it is
the `studio-ops` stage, which unloads and reloads a checkpoint three times.

```bash
./scripts/lyre live-check --list                    # what each stage proves
./scripts/lyre live-check --stages preflight,generate,cover
./scripts/lyre live-check --include-lora            # adds roughly an hour
./scripts/lyre live-check --keep                    # keep the takes to listen to
./scripts/lyre live-check --base-url http://127.0.0.1:9000
```

The check creates its own project, and deletes it on success unless `--keep`
is passed. **On failure the project is always kept**, so the takes that
exposed the problem can be listened to. Your other projects are untouched.

Exit code is 0 only if every selected stage passed. A failure stops the run:
later stages reuse earlier takes, so continuing produces cascades rather than
information.

## What each stage proves

| Stage | What a failure means |
|---|---|
| `preflight` | The server is up but no worker has reported in, or the worker failed to load models. |
| `generate` | Simple mode is broken: either the 5 Hz LM did not fill a caption, or the server did not persist the filled plan that makes simple mode reusable. Also checks that the worker replaces the `-1` seed with the value it used. |
| `custom` | An explicit plan was not honored: a fixed seed was not recorded, or the LM rewrote a caption the user locked with `caption_rewrite: false`. |
| `profiles` | The 50-step `polish` profile or the 4B `quality` profile failed to load and generate. On the 16 GB target, a successful `quality` run also proves the required CPU-offload API is available. |
| `cover` | A cover overwrote its source instead of creating a new take, or did not set `parent_take_id`. Takes are immutable (SPEC.md section 7.3); the history depends on this. |
| `repaint` | The requested region was not recorded, which is what the waveform drag-selection feeds. |
| `upload` | Drag-drop ingest is broken, or the file did not land under `uploads/` inside the project. |
| `studio-ops` | The stage requires real models. `extract`/`lego`/`complete` must swap the base checkpoint in and restore the previous checkpoint afterwards. A failure means the swap did not happen or was one-way, and the UI or later jobs will use the wrong model. This is also where VRAM pressure shows up first. |
| `guard` | The server accepted `extract` on a non-`studio_ops` profile, i.e. it would run structural editing on the wrong checkpoint. |
| `annotations` | Favorite, notes, or the active take did not reach `meta.json` on disk. The API and disk layout disagree. |
| `export` | The zip a user drops into a DAW is missing `project.json`, `plan.json`, or the active mix; is corrupt; contains a member escaping the archive root; or the mix is silent. |
| `jail` | A traversing `upload_path` was accepted by the *running server*. This verifies the unit-tested guard in the live process. |
| `lora` | Opt-in. Style-pack training failed, produced no `lora_id`, or the pack did not attach to a later generation. This stage requires FFmpeg; generation does not. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The stage generates its own source takes, so it can run on its own. On a 16 GB card, applying the pack can trigger ACE-Step's much slower CPU VAE fallback; the verifier gives that generation the same extended timeout as training. |

## Audio checks

Every stage that produces a take reads the WAV back through the API and
asserts it is **not digital silence**, and that its duration is within a few
seconds of what was requested.

This matters because a job reporting `done` is not evidence of anything: a
broken checkpoint or a misrouted VAE decode produces a perfectly well-formed
silent file, and a status-only check sails straight past it.

The check refuses to run against `LYRE_WORKER=mock`, which writes half-second
silent WAVs at 8 kHz. Every audio, duration, and model-swap assertion would
be meaningless. It recognizes the mock by that fingerprint rather than by
asking `/api/health`, which does not name the backend once a worker has
reported. `--allow-mock` relaxes the audio assertions and exists only for
checking the script itself.

## Reading a failure

Each stage prints its own notes as it goes: seeds, durations, sample rates,
RMS, file sizes, and the loaded profile. It then prints `PASS` or `FAIL` with a
message saying what the failure implies. The summary table at
the end shows every stage with its timing.

If a job hangs, the worker's terminal is the place to look; a stuck job is
almost always a crash or a model still downloading.

## Before a release

Run it against real weights on the target hardware. `pytest` and CI verify the
surrounding plumbing but cannot prove that generation works.
