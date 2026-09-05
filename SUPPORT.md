# Support

Wizard's Lyre is a volunteer-maintained local tool. Help is offered on a
best-effort basis; there is no guaranteed response time or commercial support.

## Start here

1. Run `./scripts/lyre doctor` and keep its output.
2. Check [Troubleshooting](docs/TROUBLESHOOTING.md) for the common setup, worker,
   model, and VRAM failures.
3. Search [existing issues](https://github.com/wizards-ecosystem/wizards-lyre/issues)
   before opening a new one.

If the problem remains, use the
[setup and usage question](https://github.com/wizards-ecosystem/wizards-lyre/issues/new?template=help.yml)
template. For reproducible defects, use the
[bug report](https://github.com/wizards-ecosystem/wizards-lyre/issues/new?template=bug_report.yml)
template instead.

Include your operating system, GPU and VRAM, worker backend, the output of
`./scripts/lyre doctor`, and the relevant server or worker log. Remove personal
paths or project names if they are sensitive. Never attach model weights,
private audio, generated projects, or secrets unless you deliberately intend to
publish them.

## Scope of support

The supported product and hardware boundaries are defined by [SPEC.md](SPEC.md).
In particular, Lyre is localhost-only, single-user, ACE-Step-only, and not a
cloud service or full DAW. Requests outside those boundaries may be closed with
an explanation; forks remain welcome under the MIT license.

Security vulnerabilities must be reported privately as described in
[SECURITY.md](SECURITY.md). Conduct concerns use the private path in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), not a public issue.
