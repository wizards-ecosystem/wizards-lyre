# Security

## Supported versions

Security fixes are made on `main` and included in the next release. Only the
latest published release is supported; older releases should be upgraded
before reporting a version-specific issue.

## Threat model, stated plainly

Wizard's Lyre is a **single-user tool for one machine**. It is not hardened for
any other deployment, and the design deliberately rules one out (SPEC.md §2):

- **There is no authentication, and there never will be.** Anyone who can reach
  the HTTP port has full control of every project, take, and file operation the
  server can perform.
- **The server binds `127.0.0.1` only.** This is enforced in `server/config.py`
  and asserted by `tests/test_spec_lock.py`, which fails the build if a public
  bind host appears anywhere in Lyre's own source.
- **Do not expose it to a network.** Do not put it behind a reverse proxy, do
  not port-forward it, do not bind it to `0.0.0.0`, and do not run it on a
  shared machine you do not trust. Treat the port as equivalent to a shell on
  that machine.

If you need multi-user or remote access, Lyre is the wrong tool. Building that
is explicitly out of scope.

## What is defended anyway

Even for a local tool, some inputs are worth constraining, and these are
covered by tests:

- **Path jail.** Every filesystem path is resolved through
  `server/storage/paths.py`, which rejects any path escaping `projects/` or
  `output/` — including `..` traversal and symlinks. Job payloads,
  `upload_path` values, and export archive member names all pass through it.
- **Upload limits.** Request bodies are counted as they arrive by an ASGI
  middleware above routing (`server/app.py`), so an oversized upload is
  rejected before `python-multipart` can spool it to disk. Uploads are capped
  at `MAX_UPLOAD_BYTES` and restricted to `.wav` / `.mp3`; the client's
  filename is discarded entirely in favour of a generated id.
- **Zip-slip.** Export archive member names are sanitized and then re-checked
  for absolute paths and `..` segments before being written.

## Reporting a vulnerability

Please report anything security-relevant privately rather than in a public
issue: open a [security advisory][advisory] on the repository.

Include the affected revision, impact, reproduction steps or a proof of
concept, and any suggested remediation. Do not include private audio or model
weights. Maintainers will acknowledge reports on a best-effort basis, keep the
report private while it is assessed, and coordinate disclosure after a fix is
available. This volunteer project cannot promise a response or remediation
deadline.

Because of the threat model above, "the API has no authentication" and
"anyone on localhost can call it" are known and intended, not vulnerabilities.
Reports about the path jail, the upload limits, the archive builder, or
anything that lets Lyre read or write outside its own directories are very
much wanted.

Dependency vulnerabilities without a demonstrated Lyre impact are still
useful, but please check the lockfiles and existing Dependabot alerts before
reporting. Pull requests are scanned by dependency review, and the repository
is scanned with CodeQL. The installed ACE-Step environment, patched dependency
overlay, and narrowly reviewed audit exceptions are documented in
[docs/SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md).

[advisory]: https://github.com/wizards-ecosystem/wizards-lyre/security/advisories/new
