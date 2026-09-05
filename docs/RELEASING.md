# Release guide

Each release has two layers: GitHub's tag-based source archive for contributors,
and reproducible, slim Linux/WSL2 runtime archives for users. The package
version may be a release candidate before a tag exists; [CHANGELOG.md](../CHANGELOG.md)
is the authority on whether a version has actually shipped.

## 1. Prepare the candidate

- Confirm the change is within [SPEC.md](../SPEC.md) and all work for the target
  phase is complete.
- Choose the SemVer version and update it in `pyproject.toml` and
  `web/package.json`.
- Move the relevant changelog entries from `Unreleased` to a dated version.
- Confirm [ACE_STEP_REVISION](../ACE_STEP_REVISION) is intentional. After any
  revision change, run the upstream signature conformance test on the GPU
  machine.
- Review user-facing instructions, third-party notices, supported versions,
  and model size estimates for accuracy.
- Review every exception and exact override in
  [SECURITY_AUDIT.md](SECURITY_AUDIT.md); remove any exception that now has a
  compatible fix.

## 2. Run the reproducible gates

From a clean checkout:

```bash
./scripts/lyre install
./scripts/lyre doctor
./scripts/lyre test
./scripts/lyre test-web
./scripts/lyre lint
./scripts/lyre audit
uv build
./scripts/lyre release --version X.Y.Z
```

`release` rebuilds the web app and writes a `.tar.gz`, `.zip`, and `SHA256SUMS`
under `dist/release/`. Repeating it from the same commit and toolchain must
produce byte-identical archives. Inspect both formats: they must contain the
prebuilt UI, all `server.*` and `worker.*` runtime packages, launcher, lockfiles,
MIT license, notices, and `LYRE_RELEASE.json`. They must not contain tests,
source UI, community documents, model weights, generated audio, runtime
databases, caches, or `vendor/`.

Also inspect the wheel and source archive written directly under `dist/`; they
must contain every Python subpackage and must not contain runtime state.

## 3. Run the hardware gates

These are manual by design and never run in ordinary CI:

```bash
./scripts/lyre smoke-gpu
./scripts/lyre live-check
```

Before declaring a release stable, also verify:

- `iterate`, `polish`, `quality` with CPU offload, and `studio_ops` on the
  documented 16 GB target;
- LoRA training and application with FFmpeg installed;
- a fresh install followed only by the public instructions;
- the production bundle in a real browser, including project creation,
  generation, playback, waveform selection, A/B, export, and deletion;
- recovery after stopping the worker during a running job.

Record the GPU, driver, operating system, ACE-Step revision, and results in the
release notes. A skipped hardware gate must be called out as unverified; a
release candidate must not silently present it as tested.

## 4. Publish

1. Confirm CI, dependency review, and CodeQL are green on the exact commit.
2. Create a signed or annotated tag: `git tag -s vX.Y.Z` (use `-a` if signing
   is unavailable), then push the tag.
3. The release workflow repeats the GPU-free gates, validates the tag against
   both package versions, creates the runtime archives and checksums, records a
   GitHub build-provenance attestation, and opens a **draft** GitHub release.
4. Add the matching changelog section and hardware verification notes to the
   draft. Download an archive, verify `SHA256SUMS` and `gh attestation verify`,
   then follow only its bundled README on a fresh machine or directory.
5. Publish the reviewed draft. Verify its links and installation commands.
6. Restore an empty `Unreleased` section and begin the next development cycle.

Do not publish model weights or generated user data as release assets.
