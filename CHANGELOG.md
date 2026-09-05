# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Release candidate for the first public release. Everything below describes the
current candidate, not an incremental change from a prior published version.

### Added

- **Generate.** FastAPI server, SQLite job queue (`queued → running →
  done | error`), and a dedicated worker process that loads ACE-Step 1.5 and
  drains the queue. A React SPA with a project library and a three-pane
  workspace. Simple mode seeds a plan from a natural-language query via the
  5 Hz LM; Custom mode is the human editing caption, lyrics, and metadata.
- **Studio loop.** Cover and repaint against a source take, waveform with
  drag-to-select regions, and a `parent_take_id` chain so every take is
  immutable and traceable.
- **Base swap.** `extract`, `lego`, and `complete` on the base checkpoint, with
  worker unload/load and an explicit confirmation before the swap.
- **Polish.** A quality score on takes, `.lrc` output when ACE-Step supplies
  timestamps, and LoRA style packs trained from 8+ selected takes.
- **Studio ergonomics.** A/B take compare, project export as a zip, keyboard
  shortcuts, and restoring an earlier take as active without losing history.
- **Library and ingest.** Project search, favorites, free-text take notes, a
  live loudness/peak meter, and drag-drop of a local WAV/MP3 as a source.
- MIT license, contributor and security documentation, GitHub Actions CI
  covering both stacks on Python 3.11/3.12 and Node 20, and Dependabot.
- `./scripts/lyre lint` and `./scripts/lyre format`, running ruff, mypy, tsc,
  ESLint (at zero warnings), Prettier, and shellcheck. Coverage is gated at
  88%.
- Tests for behavior that previously had none: the one-GPU-occupant guard
  (OS-level lock plus SQLite lease), the cross-process project lock's
  contention paths, and the accuracy of each package's re-export façade.
- Tests that keep `docs/API.md` and `docs/CONFIGURATION.md` in step with the
  routes the app serves and the variables it reads, so neither can drift.
- `./scripts/lyre live-check`, an end-to-end pass against a running server,
  worker, and GPU. It covers what the mocked suite cannot: the base-model
  swap, plan persistence through the real LM, and whether generated takes are
  actually audible rather than well-formed silence.
- A release-ready public project surface: illustrated README, full installation
  and release guides, support routing, third-party/model notices, machine-readable
  REUSE licensing, and an explicit inbound-contribution license policy.
- `./scripts/lyre doctor` plus core, standard, and full model-download choices,
  so a user can validate an install and avoid downloading profiles they do not
  need yet.
- Pull-request dependency review and scheduled CodeQL scanning, alongside
  frontend dependency auditing in CI.
- Reproducible `.tar.gz` and `.zip` Linux/WSL2 runtime bundles with a prebuilt
  UI, minimal bundled documentation, a per-file manifest, SHA-256 checksums,
  and a Node-free install path.
- A tag-driven release workflow that repeats the GPU-free quality gates,
  attests the runtime archives, and creates a draft GitHub release for manual
  hardware-note review before publication.

### Changed

Nothing here is a change users can observe: 0.1.0 is the first release. These
are recorded because they shaped the code a contributor will read.

- `server/storage.py`, `server/jobs.py`, and `worker/acestep_worker.py` are now
  packages split by concern. Each package's `__init__` re-exports the previous
  surface, so `storage.<name>` and `jobs.<name>` call sites are unchanged.
- `web/` uses standard `vite.config.ts` and `vitest.config.ts` instead of four
  hand-rolled `.mjs` entry points that worked around an esbuild bug in a
  filesystem layout this project no longer uses.
- `web/src/styles.css` is split into eight numbered layer files imported in
  cascade order; the bundled CSS is byte-identical.
- Test fixtures are shared through `tests/conftest.py` rather than copy-pasted
  across 20 files.
- `web/src/App.tsx` sheds its constants, types, leaf components, the library
  and style-pack panes, and five hooks -- including the plan-autosave and
  take-notes debounce machinery, whose ordering and failure semantics are
  subtle enough to deserve isolating.
- 1024 CSS declarations and 284 whole rules that later layers already
  overrode are gone; the bundled stylesheet drops from 81.8 kB to 53.0 kB
  with a verified-identical effective cascade.
- The Python package configuration now discovers every nested `server.*` and
  `worker.*` package in built source artifacts.
- Vite, Vitest, and the React plugin were upgraded to maintained releases; the
  documented Node.js floor now matches their actual runtime requirement.

### Fixed

- **Simple-mode generation never worked against real ACE-Step.** The adapter
  passed `create_sample(lm_handler=...)`; upstream spells it `llm_handler`.
- **Pinned seeds were recorded but never applied.** Upstream resolves the seed
  from `GenerationConfig.seeds`, not `GenerationParams.seed`, so every "fixed"
  seed silently produced different audio each run.
- Style-pack training now explains itself when it labels zero source files,
  which means FFmpeg is missing rather than anything being wrong with the
  takes.
- `storage.touch_project` had been defined and exported since the first commit
  without ever being called.
- `scripts/lyre` was recorded in git as non-executable, so the README's first
  instruction failed on a fresh clone with "Permission denied".
- The project-lock timeout chained the wrong exception, reporting a transient
  Windows sharing violation as the cause of a lock timeout.
- `WorkerFn` was typed as returning two values while every worker returns
  three, so the annotation misdescribed the worker contract.
- `enqueue_job` built a style pack's adapter path from the raw client-supplied
  `lora_id` instead of the resolved, validated record.
- The frontend lockfile no longer contains the five known advisories reported
  by `npm audit`, including the critical Vitest file-read issue and Vite path
  traversal issues.
- Source builds no longer combine a PEP 639 license expression with the
  superseded license classifier, which recent Setuptools rejects.
- Model-footprint guidance now reflects the pinned XL checkpoint's sharded
  20 GB layout, and `doctor` recognizes both sharded and single-file weights.
- Direct worker and GPU-smoke invocations now resolve the checkout-level model
  directory instead of incorrectly looking under `worker/checkpoints/`.

### Security

- Setup removes ACE-Step's unused Gradio UI packages and applies a small exact
  overlay for patched versions of shared runtime dependencies.
- `./scripts/lyre audit` checks the complete installed ACE-Step environment and
  the frontend lockfile. The remaining upstream constraints have explicit,
  reachability-reviewed exceptions in `docs/SECURITY_AUDIT.md`; new advisories
  still fail the audit.
- GitHub Actions dependencies are pinned to verified full commit SHAs and kept
  current by Dependabot.

[Unreleased]: https://github.com/wizards-ecosystem/wizards-lyre/commits/main
