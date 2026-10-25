# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Bounded completion (SPEC.md section 16).** One creation contract for the
  studio, `/v1`, and MCP: `creation_mode` `from_idea` or `from_brief`,
  per-field origins (user, planner, preset, default), and explicit fields
  always win over the planner. Captions are not rewritten unless asked.
  Supplied lyrics are kept verbatim and planned with `format_sample`.
- **Frozen jobs and provenance.** Every job freezes what it will render
  when it is queued and records the resolved brief before the first render.
  Each take records its job, brief, actual parameters, stage timestamps, and
  runtime. A retry is a new job (`POST /v1/jobs/{id}/retry`).
- **Safe recovery and edits.** Renders are staged and committed under the
  project lock, so a crash never leaves a half-written take. Plan saves send
  `expected_revision`; a stale save is `409`, and the studio offers "Keep my
  edits" or "Use saved plan" and keeps a local draft. A finished take never
  replaces the kept take.
- **Studio jobs survive a reload.** The studio reattaches to every running
  job in the open project, and only that project's jobs appear in it.
- **Readiness and presets.** `GET /v1/capabilities` reports what this
  machine can render now. Six curated presets (songs and game cues) preview
  before they apply.
- **Audition.** One player owns playback: A/B switching keeps the position,
  and Space plays and pauses outside text fields.
- **Export manifest.** The export ZIP carries `manifest.json`
  (`lyre.export-manifest/1`). New MCP tools: `get_capabilities`,
  `generate_take`, `set_active_take`, `get_export`.
- **Backup and restore.** `./scripts/lyre backup` and `restore`. Jobs queued
  before the upgrade, with no snapshot, wait as "Needs confirmation" instead
  of rendering from a guessed plan. See `docs/COMPLETION_MIGRATION.md`.
- **Benchmark.** `./scripts/lyre live-check --benchmark` times each stage
  from take provenance; results in `docs/COMPLETION_BENCHMARK.md`.
- **Declared playback ffmpeg.** The worker runtime depends on `imageio-ffmpeg`
  0.6.0 (no published PyPI, OSV, or GitHub advisories; wheels for Linux x86_64
  and Windows, including `libmp3lame`). `./scripts/lyre install` installs it
  with the other locked runtime dependencies, so a CUDA worker venv gets a
  playback encoder without a system `ffmpeg` and without the CUDA stack being
  part of the CUDA-less server install. `ffmpeg` on `PATH` still takes
  precedence. Style-pack training still needs a system FFmpeg with shared
  libraries; the bundled executable does not provide those.
- **MCP server.** `./scripts/lyre mcp` speaks MCP 2026-07-28 on stdin/stdout,
  and the HTTP server accepts stateless `POST /mcp` on the same port. Both
  are a thin client of `/v1` (`LYRE_URL`, optional `LYRE_API_KEY` on stdio).
  Tools: `create_song`, `transform_take`, `get_job`, `cancel_job`,
  `list_projects`, `get_project`, `update_plan`, `list_takes`,
  `get_artifact`. Resources: `lyre://project/{id}` and
  `lyre://take/{project_id}/{take_id}`. Generation returns a task whose
  `taskId` is the job id, or polls when the caller passes
  `wait_timeout_sec`. See `docs/MCP.md`.
- **One-shot songs.** `POST /v1/songs` creates a project, writes the plan, and
  queues `generate` in one call (`202`, `Location: /v1/jobs/{id}`, the job
  object). Scope `generate`. Prompt-only is simple mode; caption or lyrics is
  custom mode and `caption_rewrite` defaults to false. A null duration is
  stored as null. `Idempotency-Key` replays the queued job as `200`. A
  non-null `webhook_url` is rejected until webhook delivery is served. `/api`
  is unchanged.
- **Agent API writes.** `/v1` can now create and update projects, replace a
  plan, upload a WAV/MP3, annotate takes, set the active take, delete a
  project, list style packs, and download the export zip. `POST
  /v1/projects/{id}/jobs` queues a job (`202`, `Location`) through the same
  enqueue function as `/api`. `/api` is unchanged.
- **Idempotency-Key** on every `/v1` POST. The same key and the same raw body
  replay the original response for 24 hours; a queued job replays as `200`.
  The same key and a different body is `409`. Keys are stored in the jobs
  SQLite database and expire.
- **Agent API, first slice.** Typed `GET /v1` reads (health, projects, takes,
  jobs) with response schemas in OpenAPI, one job object for `queued`,
  `running`, `done`, `error`, and `cancelled`, and `POST /v1/jobs/{id}/cancel`.
  Cancel is cooperative: a queued job is never claimed, and a running job
  finishes the current ACE-Step call, then drops the audio. Stable artifact
  GETs serve `mix.wav`, `mix.mp3`, and `lyrics.lrc`. The studio and `/api`
  are unchanged, and `/api` does not yet send deprecation headers.
- **Opt-in API keys.** `./scripts/lyre api-key create` / `revoke`. Secrets are
  stored as sha256 in an owner-only file. Loopback with auth off stays the
  default. A non-loopback `LYRE_BIND`, or `LYRE_REQUIRE_AUTH=1`, requires a
  key and serves `/v1` only.

### Security

- `POST /mcp` rejects a present `Origin` that is not loopback (`localhost`,
  `127.0.0.1`, or `[::1]`, any port) and not listed in `LYRE_ALLOWED_ORIGINS`.
  The response is `403` in the `/v1` error shape. A missing `Origin` stays
  allowed, so a web page cannot queue jobs on the default loopback server
  while auth is off.
- A bad `LYRE_BIND` / `LYRE_REQUIRE_AUTH` no longer returns its exception text
  on `/v1`. The response is a generic misconfigured error plus a `request_id`;
  the detail is logged on the server. `api-key create` still shows the secret
  once on stdout and logs only the key id.
- Bump the web devDependency `vite` to 7.3.7 so `npm audit` clears
  GHSA-698v-4r3x-3vcv (`launch-editor` percent-escaping on Windows). The
  studio bundle does not ship Vite.
- Bump the web devDependency `vitest` to 4.1.12 so `npm audit` clears
  GHSA-qxg6-89mg-rqv8 and GHSA-xp26-3454-mjgp in `@vitest/mocker`. Vitest
  is a test runner only; the studio bundle does not ship it.
- The ACE-Step overlay now pins `werkzeug==3.1.9` (CVE-2026-102598) and
  `urllib3==2.8.0` (GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw,
  GHSA-gh4c-6fx4-qh6g). `click` 8.5.0, `idna` 3.20, and `starlette` 1.7.0
  stay aligned with Lyre's lock. FastAPI 0.141.1 accepts Starlette >=0.46.0,
  and the overlay's `anyio==4.14.2` satisfies Starlette 1.7.

### Fixed

- A hand-written brief with one planner-filled field rendered with LM codes
  on, which broke same-seed reproducibility.
- Instrumental takes ran lyric alignment on the `[Instrumental]`
  placeholder: about 11 s per take on the `quality` profile.
- At window widths of 1360 px and up, the takes drawer had no button to
  open it.
- A planned brief now merges into the draft before its job reads `done`.
- A VRAM pre-flight refusal says another program is probably using the GPU.
- **Quality profile on a 16 GB card.** A `quality` job that followed
  `iterate` or `polish` kept the previous DiT and the 1.7B LM on the GPU,
  and passed `offload_to_cpu` without `offload_dit_to_cpu`. Upstream then
  leaves the 4B DiT resident, so VAE decode saw ~0 GB free, fell back to
  CPU, and sat on one core. Lyre now moves the retired DiT off the device
  (`gc` + `empty_cache`) before the swap, parks the LM on CPU for that
  profile, passes both offload flags, and sizes the CPU thread pool to
  `cpu_count` (or `LYRE_CPU_THREADS`). `generate_music`'s arguments are
  unchanged. Upstream still defaults the DiT to stay resident under
  `offload_to_cpu` alone, and its CPU VAE fallback is single-threaded
  unless the caller sets the pool; that should be filed against
  ACE-Step 1.5 (same family as upstream issue #248).
- **Hung generates.** The job heartbeat kept a stuck decode `running`
  forever, because the lease is renewed while the process is alive. The
  same loop now marks the job `error` and exits the worker when generation
  exceeds `LYRE_JOB_TIMEOUT_SEC` (default 20 minutes) or LoRA training
  exceeds `LYRE_LORA_TIMEOUT_SEC` (default 3 hours). The deadline lifts at
  `writing_take`, after `generate_music` returns, so the MP3 encode and the
  take write are not killed mid-file. Restart the worker to continue the
  queue. The row is already `error`, so stale-job reclaim does not requeue it.
- **`mix.mp3`.** SPEC.md section 7's playback copy was never written, so
  `GET /v1/artifacts/{project}/{take}/mix.mp3` was 404 after a successful
  song while `mix.wav` and `lyrics.lrc` were 200. The worker transcodes the
  WAV with ffmpeg. The output path is `mix.mp3.partial`, so the command
  passes `-f mp3` (without it ffmpeg exits 234, "Unable to choose an output
  format"). The binary is `ffmpeg` on `PATH`, which takes precedence, then
  `imageio_ffmpeg.get_ffmpeg_exe()` from the declared `imageio-ffmpeg`
  dependency. The job's `artifacts` list still names only files that exist,
  so a machine with neither binary does not advertise `mix.mp3`.

- `POST /v1/songs` and `POST /v1/projects/{id}/jobs` return the accepted
  job as `queued`. A worker that claims the row before the response is
  built no longer changes that `202` body. Poll `GET /v1/jobs/{id}` for
  the live status.

### Changed

- SPEC.md section 15 still specifies SSE, webhooks, and signed artifact
  tokens. Those are not served yet. The studio default remains loopback
  with auth off, and ACE-Step 1.5 is still the only engine. See
  `docs/adr/0001-agent-api.md`.

## [0.2.0] - 2026-09-26

### Added

- **Remote GPU lane (opt-in).** `LYRE_WORKER=remote` sends render jobs to an
  ACE-Step 1.5 host on a GPU you rent or own, while the queue, the GPU lease,
  and every take stay local. The host (`worker/remote_host`) wraps the same
  `worker/acestep_worker` adapter, speaks an authenticated submit/poll/ack
  protocol, fetches weights on first use, and reports a build id so a stale
  host is flagged. Generate, cover, repaint, extract, lego, and complete work
  remotely; style packs stay local for now. Mirrors The Wizard's Brush's
  Remote GPU lane. See `docs/remote-gpu.md`.
- **Runpod provisioner.** With `LYRE_REMOTE_GPU_PROVISIONER=runpod`, a Remote
  GPU panel shows the rate before you start, starts one pod on your click,
  shows elapsed time and estimated spend, and terminates it on Stop or a clean
  shutdown. Nothing rents hardware on its own. See `docs/runpod.md`.
- `./scripts/lyre remote-host` and `./scripts/lyre remote-image`, and a
  workflow that builds, boots, and (on release) publishes the host image to
  GHCR tagged by build id.
- `scripts/lyre` loads plain `NAME=value` lines from a gitignored `.env`
  (parsed, never executed; the environment wins), for keeping
  `LYRE_RUNPOD_API_KEY` out of shell history.

### Security

- The ACE-Step overlay now pins `anyio==4.14.2` (CVE-2026-63374,
  CVE-2026-64847) and aligns `idna` with Lyre's own lock (3.19) instead of
  downgrading it.
- `accelerate` `PYSEC-2026-3804` has no fixed release; it is a reviewed
  exception in `docs/SECURITY_AUDIT.md` because Lyre never loads a checkpoint
  index it did not download itself.

### Changed

- SPEC.md section 3.1 amends "fully local" to "local by default" to admit the
  opt-in Remote GPU. ACE-Step 1.5 remains the only engine, and the Lyre server
  still binds `127.0.0.1` with no auth.

## [0.1.0] - 2026-09-05

First public release. Everything below describes the combined public baseline.

### Added

- **Generate.** FastAPI server, SQLite job queue (`queued -> running ->
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
  audible rather than well-formed silence.
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
- The public repository launch, with protected `main` and `v*` refs, private
  vulnerability reporting, secret push protection, least-privilege Actions,
  and a contribution path backed by required review and CI.

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
  from `GenerationConfig.seeds`; the adapter had populated
  `GenerationParams.seed`, so every "fixed" seed produced different audio each
  run.
- Style-pack training now explains itself when it labels zero source files,
  identifying missing FFmpeg as the gate for style-pack LoRA training while
  leaving generation unaffected.
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
- Style-pack training now writes the caption sidecar name ACE-Step actually
  reads and normalizes ACE-Step's nested PEFT output to Lyre's stable adapter
  path, so a trained pack can be loaded and applied to generation.
- The live-stack verifier gives LoRA-applied generation the same extended
  timeout as training, accommodating ACE-Step's low-VRAM CPU decode fallback.

### Security

- Setup removes ACE-Step's unused Gradio UI packages and applies a small exact
  overlay for patched versions of shared runtime dependencies.
- `./scripts/lyre audit` checks the complete installed ACE-Step environment and
  the frontend lockfile. The remaining upstream constraints have explicit,
  reachability-reviewed exceptions in `docs/SECURITY_AUDIT.md`; new advisories
  still fail the audit.
- GitHub Actions dependencies are pinned to verified full commit SHAs and kept
  current by Dependabot.

[Unreleased]: https://github.com/wizards-ecosystem/wizards-lyre/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/wizards-ecosystem/wizards-lyre/releases/tag/v0.2.0
[0.1.0]: https://github.com/wizards-ecosystem/wizards-lyre/releases/tag/v0.1.0
