# Security

## Supported versions

Security fixes are made on `main` and included in the next release. Only the
latest published release is supported; older releases should be upgraded
before reporting a version-specific issue.

## Threat model, stated plainly

The Wizard's Lyre is a **single-operator tool**. The running release's
security boundary is the loopback interface:

- **The default has no authentication.** Anyone who can reach the HTTP port
  has full control of every project, take, and file operation the server can
  perform. On the default bind that person is a local user of the machine.
- **The server binds `127.0.0.1` unless a future setting says otherwise.**
  Today that host is fixed in `server/config.py`. `tests/test_spec_lock.py`
  fails the build if a public bind host appears as a literal in Lyre's own
  source. The default constant stays `127.0.0.1` even after the opt-in below
  is built.
- **Do not expose this release to a network.** Do not put it behind a reverse
  proxy, do not port-forward it, and do not run it on a shared machine you do
  not trust. Treat the port as equivalent to a shell on that machine.

SPEC.md section 15 specifies an opt-in after this release: the operator may
set a non-loopback bind, and the process must then refuse to start unless an
API key exists. A tunnel or reverse proxy onto loopback must set
`LYRE_REQUIRE_AUTH=1`, because the server cannot see the client in front of
the proxy. That opt-in is not implemented. Until the phase 11 PR lands, the
paragraphs above are the whole story, and exposing the port is still unsafe.

Lyre does not grow accounts or a hosted multi-tenant service. Keys, when they
exist, name one operator's agents. Remote GPU session start stays a human
action on the loopback studio; an agent job must not rent a machine.

## The optional Remote GPU

The opt-in [Remote GPU lane](docs/remote-gpu.md) sends render jobs to an
ACE-Step host on another machine. It opens nothing on yours. The host is the
one Lyre component that serves on a network, so it is built for that:

- It authenticates every request with a shared secret before reading the body,
  caps request size, and refuses to start with a missing, short, or placeholder
  secret.
- Lyre sends the secret only over HTTPS (plain HTTP only to loopback), never
  follows a redirect, stores it in an owner-only file, and never returns it
  from its API.
- A provisioner's API key stays in the server's environment. A Runpod pod gets
  only a secret generated for that session.

Anyone who has the host's address and secret can use that GPU, and read the
lyrics, captions, and source audio of jobs sent to it. Treat both as
credentials, and terminate a rented host when you are done.

## What is defended anyway

Even for a local tool, some inputs are worth constraining, and these are
covered by tests:

- **Path jail.** Every filesystem path is resolved through
  `server/storage/paths.py`, which rejects any path escaping `projects/` or
  `output/`, including `..` traversal and symlinks. Job payloads,
  `upload_path` values, and export archive member names all pass through it.
- **Upload limits.** Request bodies are counted as they arrive by an ASGI
  middleware above routing (`server/app.py`), so an oversized upload is
  rejected before `python-multipart` can spool it to disk. Uploads are capped
  at `MAX_UPLOAD_BYTES` and restricted to `.wav` / `.mp3`; the client's
  filename is discarded in favour of a generated id.
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

Because of the threat model above, "the default API has no authentication"
and "anyone on localhost can call it" describe expected behavior rather than
vulnerabilities. Once section 15 is implemented, a network bind that starts
without a key, a leaked key file, a webhook that can reach a metadata
address, and an agent path that rents a GPU are in scope.
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
