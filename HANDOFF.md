# Handoff — Wizard's Lyre FOSS release prep

State as of 2026-08-30. Branch `foss-release-prep`, 31 commits, open as
**[PR #60](https://github.com/wizards-ecosystem/wizards-lyre/pull/60)** —
mergeable, CI green on all four jobs.

Everything below is what is *left* or *unverified*. It is not a list of
problems with the branch; the branch is ready to merge. It is the list of
things the next person should know rather than rediscover.

## Blocked on this machine

### 1. Style-pack (LoRA) training is unverified — needs FFmpeg

The only feature never run end to end. ACE-Step decodes training audio through
torchcodec, which loads FFmpeg's shared libraries; FFmpeg is not installed here
and installing it needs root.

```bash
sudo apt install ffmpeg
./scripts/lyre live-check --stages preflight,lora    # ~1 hour, self-contained
```

Symptom without it: training stages its 8 files, ACE-Step counts them, labels
**zero**. Generation is unaffected (it falls back to soundfile), so the machine
generates perfectly well and trains nothing. The error now names FFmpeg as the
likely cause; that message has been confirmed against the real worker, but the
success path has not.

### 2. No real-browser verification of the CSS change

1024 provably-dead CSS declarations were removed. This was verified twice with
an independently written cascade-equivalence checker — on the source files
(2236 selector/property pairs) and on the shipped minified bundles (2256) —
and every effective value is identical.

A computed-style diff in a real browser was the intended third check. The
bundled Chromium is missing `libnspr4`, and installing it needs root:

```bash
sudo npx playwright install-deps chromium
```

The two cascade checks are strong, but nobody has looked at the app.

## Verified, for the record

`./scripts/lyre live-check` — **12/12 stages against the RTX 4070 Ti SUPER**,
~1.1 min: simple and custom generate, seed reproducibility, cover, repaint,
upload ingest, extract/lego/complete with the base-model swap observed both
ways, the studio_ops guard, annotations reaching `meta.json`, the export
archive, and the path jail.

Also: 224 unit tests, 102 frontend tests, 91% coverage, ruff + mypy + tsc +
ESLint (zero warnings) + Prettier + shellcheck clean, all re-run from a clean
clone.

## Unverified paths

### DiT profiles

Only two of four have been exercised live: `iterate` (turbo) and `studio_ops`
(base). `polish` (sft) and `quality` (xl-turbo) have not.

`quality` is the one to watch: SPEC.md §4.1 says XL needs CPU offload on a
16 GB card, and that path has never run. Its weights were absent here.

A `polish` job did complete during a probe, but the run was not inspected
closely enough to call it verified — treat both as open.

### Not covered by the live check

Frontend behaviour (A/B compare, keyboard shortcuts, the waveform UI, library
search, project delete) is covered by the 102 Vitest tests against a mocked
backend, and by nothing against the real stack. That is a reasonable division,
but it means no automated check has ever driven the actual UI.

`batch_size` is forced to 1 server-side, per SPEC.md §8.1. Not a bug; just
never exercised above 1.

## Decisions deliberately left to you

### Quote-wrapped captions

The 5 Hz LM returns simple-mode captions wrapped in single quotes:
`'A somber and introspective instrumental piece...'`. Lyre stores them
verbatim, so the quotes show up in the Plan editor and get re-fed to the model
on the next generate.

Not changed, because stripping quotes is a heuristic that would corrupt a
caption legitimately starting or ending with one. Your call whether Lyre
should normalize upstream's formatting.

### CSS: further flattening

`styles.css` is still eight layers imported in cascade order — that ordering is
the design. The dead declarations are gone, so what remains in each layer is
what it actually contributes, but `02-workstation.css` is still 1574 lines.

If you prune further, the safe rule is deletion-only where a *later* rule has
the identical selector in the identical at-rule context. Anything beyond that
needs the browser verification above.

### App.tsx is still 2300 lines

Down from 2626. Its constants, types, leaf components, the library and
style-pack panes, and five hooks have been extracted. The remaining panes
(Plan, Takes, Waveform, Operations) share state heavily with each other and
with the job-running logic; splitting them would produce wide prop interfaces
without buying real separation, so it was stopped deliberately rather than
run out of steam.

### Coverage floor

Set to 88% in `pyproject.toml` against an actual 90.7%, so rounding does not
fail CI. Worth raising once the number has been stable across a few runs.

### Large test files

`tests/test_acestep_worker_adapter.py` is 1973 lines and
`tests/test_phase1_api.py` is 1159. Both are coherent — the first is entirely
"does the adapter match upstream's API" — but they would split cleanly if the
fake `acestep.*` modules moved into a shared `tests/fakes/`.

## Three bugs the live check found, and the guards added

All fixed and pushed. Recorded here because the *pattern* matters: each was
invisible to the mocked suite, because the fakes encoded the same wrong belief
the adapter did.

1. **Simple-mode generation never worked.** The adapter called
   `create_sample(lm_handler=...)`; upstream spells it `llm_handler`. Every
   real simple-mode generate raised TypeError, while the suite stayed green.
2. **Pinned seeds were recorded but never applied.** Upstream resolves the seed
   from `GenerationConfig.seeds`, not `GenerationParams.seed`, despite its own
   docstring. Every take reported a seed that had had no effect on it. The seed
   control shipped in SPEC phase 2 had never worked.
3. **FFmpeg missing** — item 1 above.

Guards, so the class cannot recur:

- `tests/test_acestep_signature_conformance.py` introspects the *installed*
  ACE-Step and asserts the adapter's keywords and dataclass fields exist.
  Skipped when ACE-Step is absent, so CI is unaffected — it runs on the GPU
  machine, which is the only place the question can be answered. **Run it after
  every ACE-Step version bump**; it is the cheapest possible upstream-drift
  detector.
- `FakeGenerationConfig.seeds` is keyword-only with no default, so an adapter
  that stops sending it fails loudly.
- The `reproducible` live stage compares audio byte-for-byte. Asserting on
  returned metadata cannot catch a seed that is recorded but unused.

## If you change the worker

The worker process holds ACE-Step in memory. **Restart it after editing
anything under `worker/`** — this cost two confusing live-check runs where a
fix appeared not to work because the old code was still loaded.

## Next steps, in order

1. Merge PR #60.
2. `sudo apt install ffmpeg`, then run the LoRA stage.
3. Download the `quality` (XL) weights and exercise CPU offload on 16 GB.
4. Install the Chromium system libs and do a visual pass over the CSS change.
5. Decide the caption-quoting question.
6. Tag 0.1.0. `CHANGELOG.md` is written and current.
