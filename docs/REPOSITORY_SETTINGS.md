# Repository settings

This checklist records the GitHub controls for The Wizard's Lyre. It keeps the
contribution path open while limiting accidental changes to source, workflows,
and releases.

## Repository features

- Visibility: public
- Default branch: `main`
- Issues: enabled
- Projects, wiki, and discussions: disabled until the project has a named use
  for them
- Merge method: squash only, using the pull request title and body
- Head branches: deleted after merge
- Update branch button: enabled
- Web commit signoff: disabled because the project uses the inbound license
  statement in [CONTRIBUTING.md](../CONTRIBUTING.md) instead of a DCO

Before the public launch on 2026-09-05, the complete Git history and tracked
tree were scanned for credentials, every non-`main` branch was reconciled and
removed, and the remote was verified to contain only `main`.

## Main branch protection

The following settings are active:

- Require a pull request before merging.
- Require one approval, dismiss stale approvals, require review of the latest
  push, and request the owner in `.github/CODEOWNERS`.
- Require branches to be current before merging.
- Require the `python 3.11`, `python 3.12`, `web`, `launcher`,
  `javascript-typescript`, `python`, and `dependency-review` checks.
- Require conversation resolution and linear history.
- Block force pushes and branch deletion.
- Leave signed commits optional so contributors can use GitHub's web editor or
  an unsigned local Git setup.
- Allow the repository administrator to bypass protection only for recovery.

Check names must remain unique across workflows. Update the protection rule if
a workflow intentionally renames a required check.

## Actions

- Default workflow token permission: read-only.
- Workflow tokens cannot approve pull requests.
- Allowed actions: GitHub-owned actions plus
  `astral-sh/setup-uv`; other third-party actions are blocked.
- Workflow approval is required for first-time contributors to limit
  fork-based compute abuse.
- Every action reference uses a full 40-character commit SHA.
- Checkout credentials are not persisted in the working tree.
- Every job has a timeout and the workflows cancel superseded branch runs.
- `pull_request_target` is prohibited.

Workflow files declare the smallest permissions each job needs. The release
workflow is the only workflow with `contents: write`; it runs only for `v*`
tags, requires an annotated tag whose commit is on `main`, creates a draft
release, and attests its archives.

## Dependencies and security

- Dependabot version updates cover Python, npm, and GitHub Actions.
- Dependabot alerts and security updates are enabled.
- CodeQL covers Python and JavaScript/TypeScript.
- Dependency review rejects moderate-or-higher findings on pull requests.
- Secret scanning and push protection are enabled.
- Vulnerabilities use GitHub's private advisory channel described in
  [SECURITY.md](../SECURITY.md).
- Private vulnerability reporting is enabled.
- Immutable releases are enabled, so a published release locks its tag and
  assets.

## Release tags

The active `Protect release tags` ruleset restricts creation, updates, deletion,
and force-pushes for `v*`. Organization administrators can bypass it for
recovery. Only a maintainer should create a release tag. Follow
[RELEASING.md](RELEASING.md) and publish the workflow-created draft only after
the hardware notes and downloadable archives have been reviewed.

## Periodic audit

At each release:

1. Review repository access, deploy keys, webhooks, environments, Actions
   variables and secrets, and remote branches.
2. Confirm branch and tag rules still apply.
3. Confirm every workflow action remains pinned and allowed.
4. Review open Dependabot, CodeQL, dependency-review, and secret-scanning
   alerts.
5. Run the repository test, lint, audit, and release commands in
   [RELEASING.md](RELEASING.md).
