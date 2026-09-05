# Dependency security audit

Last reviewed: 2026-09-05 against ACE-Step revision
`14c0211d5a0653b0f63e27686f4c3f151b4d8629`.

Lyre installs ACE-Step into the same local Python environment as its API. That
keeps GPU integration reproducible, but upstream's lock describes a complete
demo and training application rather than Lyre's narrower runtime. The release
process therefore audits both the installed environment and the frontend
lockfile with `./scripts/lyre audit`.

## Patched overlay

[requirements/ace-step-security.txt](../requirements/ace-step-security.txt)
pins patched versions of shared packages after the upstream lock is installed.
The file is intentionally small and exact. It is not an alternate engine or a
fork of ACE-Step; it is the compatibility layer for dependency versions, just
as `worker/acestep_worker/` is the compatibility layer for Python parameters.

ACE-Step's Gradio packages are removed after installation. Lyre neither imports
nor serves that UI, and removing it eliminates an unused network surface and
its transitive advisories. ACE-Step's headless downloader and Python generation
API remain installed.

Every overlay change requires the worker signature conformance test and a real
GPU smoke test. A release requires the full live-stack test.

## Reviewed exceptions

`scripts/lyre audit` ignores only the advisory identifiers below. They remain
visible here because an explicit, reviewed exception is safer than presenting a
misleading zero without context.

| Package / advisories | Why the vulnerable path is unreachable in Lyre |
|---|---|
| `diskcache` — `PYSEC-2026-2447` | Exploitation requires another actor to write a malicious pickle into the local cache. Lyre is single-user, keeps caches inside the checkout, accepts no cache uploads, and grants no remote filesystem access. An actor able to alter that directory already has code execution as the Lyre user. |
| `lightning` — `PYSEC-2026-3624` | The issue loads attacker-selected Python modules from a malicious training checkpoint. Lyre accepts audio for a new style pack; it has no checkpoint upload or arbitrary resume path. Style-pack records and paths are created inside the project jail. |
| `transformers` — `PYSEC-2025-217`, `PYSEC-2026-2288`, `PYSEC-2026-2289`, `PYSEC-2026-2290`, `CVE-2026-9856` | The affected paths load attacker-controlled X-CLIP, LightGlue, Trainer, model/config, or chat-template data. Lyre loads only its fixed ACE-Step model set, accepts no model repository or model configuration from HTTP, and performs no runtime Hub lookup after setup. ACE-Step currently requires Transformers `<4.58`; this exception must be removed when upstream supports a fixed major version. |
| `setuptools` — `PYSEC-2025-49`, `PYSEC-2026-3447` | These issues affect deprecated package-index download behavior and macOS source-distribution manifest exclusion. Lyre does not invoke either path at runtime, does not support macOS, and builds its own artifacts in an isolated environment. ACE-Step currently pins Setuptools `<72`; review when that upstream constraint changes. |

The audit still reports unauditable local ACE-Step/nano-vLLM packages and CUDA
wheel variants. Their source revision and PyTorch versions are pinned upstream;
they require manual review during an ACE-Step revision bump.

## Review policy

- Any advisory not listed above fails the audit.
- An exception is allowed only with a source-to-sink reachability explanation,
  not merely because a package is transitive.
- Review every exception at each release and each ACE-Step revision bump.
- Remove an exception as soon as a compatible patched version is available.
- If Lyre ever accepts models, checkpoints, cache contents, or training state
  from an untrusted source, these conclusions are invalid and release must stop.
