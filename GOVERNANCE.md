# Governance

The Wizard's Lyre is a small, maintainer-led open-source project. This document
explains who makes decisions and how contributors can participate without
creating process heavier than the project needs.

## Roles

- **Users** run the software, ask questions, and report problems.
- **Contributors** open issues, improve documentation, test changes, and submit
  pull requests.
- **Maintainers** review and merge changes, publish releases, handle security
  reports, moderate project spaces, and administer the repository.

The current maintainer is [@limbwizard](https://github.com/limbwizard). The
project may invite additional maintainers after sustained, constructive work
that demonstrates sound judgment across code, review, security, and community
interactions.

## Decisions

Routine changes are discussed and reviewed in public issues and pull requests.
Maintainers decide whether to merge after considering correctness, project
scope, maintenance cost, security, accessibility, and contributor feedback.
Decisions should include enough reasoning that a future contributor can
understand the tradeoff.

[SPEC.md](SPEC.md) is the sole product specification. A proposal cannot change
the permanent scope boundaries by editing another document or adding an
optional adapter. Forks may pursue a different product direction under the MIT
license.

Security reports and conduct complaints use the private paths in
[SECURITY.md](SECURITY.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Details
remain private until disclosure is safe or the reporter agrees otherwise.

## Repository authority

Only maintainers may merge pull requests, change repository protections, or
publish releases. CODEOWNERS requests maintainer review, automated checks cover
the GPU-free test surface, and the hardware release gates remain manual. The
controls and recovery exception are recorded in
[docs/REPOSITORY_SETTINGS.md](docs/REPOSITORY_SETTINGS.md).

A maintainer may use an administrative bypass only for repository recovery or
an urgent security response. The reason and resulting change should be
documented afterward when doing so does not expose a vulnerability.

## Changing governance

Governance changes use the same public pull-request process as other
documentation. The project does not promise that contribution will lead to a
maintainer role, and volunteer maintainers do not promise response times. If
maintenance stops, the MIT license preserves the community's right to fork and
continue the work.
