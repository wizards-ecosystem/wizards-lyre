# Contributing to Wizard's Lyre

Thanks for taking a look. This is a small, opinionated project, so the fastest
way to have a change accepted is to understand what it is deliberately *not*
trying to become — see [Scope](#scope) below before starting anything large.

## Getting set up

You need `git`, [`uv`](https://docs.astral.sh/uv/), and Node.js 20 or newer.
Everything else is installed inside the checkout.

```bash
git clone https://github.com/wizards-ecosystem/wizards-lyre.git
cd wizards-lyre
./scripts/lyre install     # dependencies + frontend build, no model weights
```

`install` is the contributor setup: it clones the pinned ACE-Step source and
builds the SPA, but skips the ~20 GB of model downloads. Use
`./scripts/lyre bootstrap` instead only when you actually want to generate
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
./scripts/lyre lint        # ruff, tsc, eslint, prettier, shellcheck
./scripts/lyre format      # fix what is auto-fixable
```

CI runs exactly these. `shellcheck` is the one optional piece — install it
from your package manager, or let CI catch launcher issues.

`./scripts/lyre smoke-gpu` runs a real ACE-Step generation. It is manual, needs
a real GPU and downloaded weights, and is never part of `pytest`.

## Scope

`SPEC.md` is the product spec and takes precedence over any other document
here, including this one. The constraints that matter most:

- **ACE-Step 1.5 on the local GPU is the only engine.** No Suno/Udio wrappers,
  no Lyria, ElevenLabs, Stability, Magenta, LeVo, or YuE — not even as an
  optional adapter or a stub. `tests/test_spec_lock.py` enforces this by
  scanning the source, and it will fail your build.
- **No Gradio.** ACE-Step's own demo UI is upstream's; Lyre owns its product UI.
- **Localhost only, no auth.** See [SECURITY.md](SECURITY.md).
- **`pytest` must never need a GPU** and must never import `acestep` or `torch`.
  If you are testing worker behavior, follow the pattern in
  `tests/test_acestep_worker_adapter.py`, which installs fake `acestep.*`
  modules matching upstream's real signatures.
- **No mixer, MIDI, plugins, Docker, or cloud deploy.** Lyre is a studio for
  iterating on takes, not a DAW.

If you want something outside this, a fork is a completely reasonable answer
and no hard feelings.

## Things worth knowing before you edit

**The package `__init__` files re-export deliberately.** `server/storage`,
`server/jobs`, and `worker/acestep_worker` are packages whose `__init__.py`
re-exports their modules' surface, including some private helpers, so that
`storage.<name>` keeps working. If you patch one of these in a test, **patch
the module that defines it** (`server.storage.jsonio._write_json`), not the
package re-export — patching the re-export silently does nothing to internal
callers, and your test will pass while exercising the unpatched code. This has
bitten before; see the comments in `server/storage/jsonio.py`.

**The stylesheet is layered, not modular.** `web/src/styles.css` imports eight
numbered files, and the order is the design: 03–08 are successive redesigns
that override each other by cascade position. Adding a rule at the end of
`08-lyre-instrument.css` is usually what you want. Flattening the stack to a
single effective stylesheet would be a genuine improvement, but it needs
visual verification the DOM-level tests cannot provide, so it has not been
done.

**Environment variables are all `LYRE_*`.** See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). Nothing reads any other prefix,
and a test fails if one appears.

## Pull requests

- One concern per PR. Keep formatting-only changes in their own commit.
- Both suites and the linters green.
- Explain *why* in the commit message. This codebase comments the reasoning
  behind non-obvious decisions heavily and deliberately — please match that;
  it is the main reason the tricky concurrency and adapter code is
  maintainable.
- New behavior needs a test. Bug fixes need a test that fails without the fix —
  and it is worth confirming it actually fails, not assuming it would.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
