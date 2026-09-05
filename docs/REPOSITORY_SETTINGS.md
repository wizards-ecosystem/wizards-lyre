# Repository settings

This checklist records the GitHub controls for The Wizard's Lyre. It keeps the
contribution path open while limiting accidental changes to source, workflows,
and releases.

## Repository features

- Default branch: `main`
- Issues: enabled
- Projects, wiki, and discussions: disabled until the project has a named use
  for them
- Merge method: squash only, using the pull request title and body
- Head branches: deleted after merge
- Update branch button: enabled
- Web commit signoff: disabled because the project uses the inbound license
  statement in [CONTRIBUTING.md](../CONTRIBUTING.md) instead of a DCO

Changing visibility to public is a release decision. Before changing it, scan
the complete Git history and every remote branch for credentials and private
material. Visibility must not be used as a shortcut around unavailable private
repository features.

## Main branch protection

Protect `main` as soon as the repository is public. GitHub Free provides branch
protection for public organization repositories, but this private repository
does not have that feature.

Use these settings:

- Require a pull request before merging.
- Require one approval, dismiss stale approvals, require review of the latest
  push, and request the owner in `.github/CODEOWNERS`.
- Require branches to be current before merging.
- Require all CI, CodeQL, and dependency-review checks.
- Require conversation resolution and linear history.
- Block force pushes and branch deletion.
- Leave signed commits optional so contributors can use GitHub's web editor or
  an unsigned local Git setup.
- Allow the repository administrator to bypass protection only for recovery.

Check names must remain unique across workflows. Confirm their exact names on
the first pull request before marking them required.

## Actions

- Default workflow token permission: read-only.
- Workflow tokens cannot approve pull requests.
- Allowed actions: GitHub-owned actions plus
  `astral-sh/setup-uv`; other third-party actions are blocked.
- After public visibility is enabled, require workflow approval for first-time
  contributors to limit fork-based compute abuse.
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
- Vulnerabilities use GitHub's private advisory channel described in
  [SECURITY.md](../SECURITY.md).
- Immutable releases are enabled, so a published release locks its tag and
  assets.

When the repository becomes public, verify that secret scanning, push
protection, code scanning, dependency review, and private vulnerability
reporting are active. Also set Actions fork approval to first-time contributors.
These controls are unavailable or limited on the current private GitHub Free
repository.

## Release tags

After public visibility enables rulesets, protect `v*` tags from updates and
deletion. Only the maintainer should create a release tag. Follow
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
