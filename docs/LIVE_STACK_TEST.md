# Live stack test

Everything in `pytest` runs against a mocked worker and never loads a GPU
(SPEC.md §11). That is deliberate — it keeps CI fast and contribution
GPU-free — but it means an entire layer goes unverified: what the server and
the worker do *together*, on real weights.

`./scripts/lyre live-check` is that layer. It drives the actual HTTP API
against a running server and a running worker, and asserts on what comes back.

It complements, rather than repeats, `./scripts/lyre smoke-gpu`, which calls
`run_job` directly and answers only "can this machine generate audio at all".

## Running it

Three terminals, from the repository root:

```bash
./scripts/lyre server        # 1
./scripts/lyre worker        # 2 — the real backend, NOT LYRE_WORKER=mock
./scripts/lyre live-check    # 3
```

Expect **10–25 minutes** for the default stages on a 16 GB card. Most of it is
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
| `generate` | Simple mode is broken: either the 5 Hz LM did not fill a caption, or the server did not persist the filled plan — which is what makes simple mode reusable. Also checks the `-1` seed is replaced by the seed actually used. |
| `custom` | An explicit plan was not honored: a fixed seed was not recorded, or the LM rewrote a caption the user locked with `caption_rewrite: false`. |
| `cover` | A cover overwrote its source instead of creating a new take, or did not set `parent_take_id`. Takes are immutable (SPEC.md §7.3); the history depends on this. |
| `repaint` | The requested region was not recorded, which is what the waveform drag-selection feeds. |
| `upload` | Drag-drop ingest is broken, or the file did not land under `uploads/` inside the project. |
| `studio-ops` | **The stage no mocked test can stand in for.** `extract`/`lego`/`complete` must swap the base checkpoint in — and back out again afterwards. A failure here means either the swap did not happen (so the UI's "loading base model" reports something untrue) or it was one-way (so `studio_ops` silently serves every later job). This is also where VRAM pressure shows up first. |
| `guard` | The server accepted `extract` on a non-`studio_ops` profile, i.e. it would run structural editing on the wrong checkpoint. |
| `annotations` | Favorite, notes, or the active take did not reach `meta.json` on disk — the API and the disk layout disagree. |
| `export` | The zip a user drops into a DAW is missing `project.json`, `plan.json`, or the active mix; is corrupt; contains a member escaping the archive root; or the mix is silent. |
| `jail` | A traversing `upload_path` was accepted by the *running server*, not just rejected in a unit test. |
| `lora` | Opt-in. Style-pack training failed, produced no `lora_id`, or the pack did not actually attach to a later generation. |

## Audio is checked, not assumed

Every stage that produces a take reads the WAV back through the API and
asserts it is **not digital silence**, and that its duration is within a few
seconds of what was requested.

This matters because a job reporting `done` is not evidence of anything: a
broken checkpoint or a misrouted VAE decode produces a perfectly well-formed
silent file, and a status-only check sails straight past it.

The check refuses to run against `LYRE_WORKER=mock`, which writes half-second
silent WAVs at 8 kHz — every audio, duration, and model-swap assertion would
be meaningless. It recognizes the mock by that fingerprint rather than by
asking `/api/health`, which does not name the backend once a worker has
reported. `--allow-mock` relaxes the audio assertions and exists only for
checking the script itself.

## Reading a failure

Each stage prints its own notes as it goes — seeds, durations, sample rates,
RMS, file sizes, the loaded profile — then `PASS` or `FAIL` with a message
saying what the failure implies, not just what differed. The summary table at
the end shows every stage with its timing.

If a job hangs, the worker's terminal is the place to look; a stuck job is
almost always a crash or a model still downloading.

## Before a release

Run it against real weights on the target hardware. `pytest` and CI cannot
tell you that generation works — only that the plumbing around it does.
