# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-29

First public release. Everything below is the state of the project at the point
it was prepared to be shared, not an incremental change from a prior release.

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
- `./scripts/lyre lint` and `./scripts/lyre format`, running ruff, tsc, ESLint,
  Prettier, and shellcheck.

### Changed

- **BREAKING: every environment variable is now `LYRE_*`.** `BARD_PORT`,
  `BARD_WORKER`, `BARD_PROJECTS_DIR`, `BARD_OUTPUT_DIR`, `BARD_DB_PATH`,
  `BARD_CHECKPOINTS_DIR`, and `BARD_DEVICE` were left over from the project's
  pre-rename name and are gone, with no fallback. The queue database is
  `projects/lyre.db`; `scripts/lyre` migrates an existing `projects/bard.db`
  (and its sidecars and worker lock) once, on its next run.
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

### Fixed

- `scripts/lyre` was recorded in git as non-executable, so the README's first
  instruction failed on a fresh clone with "Permission denied".
- The project-lock timeout chained the wrong exception, reporting a transient
  Windows sharing violation as the cause of a lock timeout.

[Unreleased]: https://github.com/wizards-ecosystem/wizards-lyre/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wizards-ecosystem/wizards-lyre/releases/tag/v0.1.0
