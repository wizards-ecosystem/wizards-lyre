# Contributing to The Wizard's Lyre

Thanks for taking a look. This is a small, opinionated project. Read
[Scope](#scope) before starting a large change so your proposal stays within
the product boundary.

## Getting set up

You need `git`, [`uv`](https://docs.astral.sh/uv/), and Node.js 20.19+ or
22.12+. Everything else is installed inside the checkout. FFmpeg is required
only to *train* style packs; generation works without it.
The [installation guide](docs/INSTALLATION.md) has the full platform and disk
requirements.

```bash
git clone https://github.com/wizards-ecosystem/wizards-lyre.git
cd wizards-lyre
./scripts/lyre install     # dependencies + frontend build, no model weights
./scripts/lyre doctor
```

`install` is the contributor setup: it clones the pinned ACE-Step source and
builds the SPA, but skips the ~20 GB of model downloads. Use
`./scripts/lyre bootstrap` when you want to generate
audio locally.

**You do not need a GPU to work on Lyre.** The entire test suite runs against
a mocked worker, and you can drive the real UI against it too:

```bash
LYRE_WORKER=mock ./scripts/lyre worker   # terminal 1: writes silent WAVs, never loads CUDA
./scripts/lyre server                    # terminal 2
./scripts/lyre web                       # terminal 3, if you want hot reload
```

## The loop

```bash
./scripts/lyre test        # pytest, mocked worker, no GPU
./scripts/lyre test-web    # vitest + React Testing Library against a mocked backend
./scripts/lyre lint        # ruff, mypy, tsc, eslint, prettier, shellcheck
./scripts/lyre audit       # known advisories in installed Python and locked JS deps
./scripts/lyre format      # fix what is auto-fixable
```

CI enforces these gates: ESLint runs at `--max-warnings 0`, pytest fails below
88% coverage, and REUSE verifies the file-level license declarations.
`shellcheck` is the one optional local piece. Install it from your package
manager, or let CI catch launcher issues. CI also builds the Python source
artifacts, audits the frontend lockfile, runs dependency review on pull
requests, and scans Python and TypeScript with CodeQL.

If you need to suppress a lint rule, do it at the site with the reason written
above it, rather than weakening the rule for the whole project. The existing
`eslint-disable-next-line` comments in `web/src/` are the pattern to follow.

Two checks need a real GPU and are never part of `pytest`:

```bash
./scripts/lyre smoke-gpu     # one ACE-Step generation: does this machine work at all
./scripts/lyre live-check    # the full stack end to end, ~10-25 min
```

`live-check` drives the real HTTP API against a running server and worker, and
covers the layer the mocked suite cannot reach, including the base-model swap
and whether generated files contain audible audio. See
[docs/LIVE_STACK_TEST.md](docs/LIVE_STACK_TEST.md).
Run it before a release.

## Scope

`SPEC.md` is the product spec and takes precedence over any other document
here, including this one. The constraints that matter most:

- **ACE-Step 1.5 on the local GPU is the only engine.** Suno/Udio wrappers,
  Lyria, ElevenLabs, Stability, Magenta, LeVo, YuE, and optional adapters or
  stubs are outside scope. `tests/test_spec_lock.py` enforces this by
  scanning the source, and it will fail your build.
- **No Gradio.** ACE-Step's own demo UI is upstream's; Lyre owns its product UI.
- **Localhost only, no auth.** See [SECURITY.md](SECURITY.md).
- **`pytest` must never need a GPU** and must never import `acestep` or `torch`.
  If you are testing worker behavior, follow the pattern in
  `tests/test_acestep_worker_adapter.py`, which installs fake `acestep.*`
  modules matching upstream's real signatures.
- **No mixer, MIDI, plugins, Docker, or cloud deploy.** Lyre focuses on
  generating and iterating on takes; DAW features are outside scope.

Forks may pursue features outside these boundaries under the MIT license.

## Things worth knowing before you edit

### Package exports

The package `__init__` files re-export by design. `server/storage`,
`server/jobs`, and `worker/acestep_worker` are packages whose `__init__.py`
re-exports their modules' surface, including some private helpers, so that
`storage.<name>` keeps working. If you patch one of these in a test, **patch
the module that defines it** (`server.storage.jsonio._write_json`) instead of
the package re-export. Patching the re-export does nothing to internal
callers, and your test will pass while exercising the unpatched code. This has
bitten before; see the comments in `server/storage/jsonio.py`.

### Stylesheet layers

The stylesheet uses cascade layers. `web/src/styles.css` imports eight
numbered files, and the order is the design: 03-08 are successive refinements
that override each other by cascade position. Adding a rule at the end of
`08-lyre-instrument.css` is usually what you want.

Declarations a later layer provably overrode have already been removed, so
what is left in each file is what that layer currently contributes. If you
prune further, the safe rule is deletion-only where a *later* rule has the
identical selector in the identical at-rule context. The same selector means
the same specificity and matched elements, so document order alone decides.
Anything beyond that needs real-browser verification, which the DOM-level test
suite does not provide.

### Environment variables

Environment variables are all `LYRE_*`. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). Nothing reads any other prefix,
and a test fails if one appears.

## Pull requests

- One concern per PR. Keep formatting-only changes in their own commit.
- Both suites and the linters green.
- Explain *why* in the commit message. This codebase comments the reasoning
  behind non-obvious decisions heavily. Match that approach;
  it is the main reason the tricky concurrency and adapter code is
  maintainable.
- New behavior needs a test. Bug fixes need a test that fails without the fix.
  Confirm the regression test fails before applying the fix.

## Licensing contributions

Unless agreed otherwise in the pull request, contributions are licensed under
the license already declared for the file: MIT for Lyre's original source,
documentation, and artwork; CC BY 4.0 for the adapted Code of Conduct. By
submitting a contribution, you confirm that you have the right to do so under
those terms. The project does not require a separate contributor license
agreement.

Do not commit generated audio, third-party model weights, training data, or
copied code whose license is incompatible or unclear. If a change adds a new
dependency or incorporates third-party material, update the relevant lockfile,
attribution, and [third-party notices](THIRD_PARTY_NOTICES.md).

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
Questions that are not code changes belong in the paths described by
[SUPPORT.md](SUPPORT.md).
