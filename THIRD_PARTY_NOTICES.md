# Third-party software and model notices

This file explains the boundaries between The Wizard's Lyre, the software it
installs, and the model artifacts it downloads. It aids redistributors and
supplements the license texts supplied by each upstream project.

## The Wizard's Lyre

Lyre's original source code, documentation, and artwork are licensed under the
[MIT License](LICENSE). The adapted [Code of Conduct](CODE_OF_CONDUCT.md) is
licensed separately under CC BY 4.0 and includes its attribution in the file.
Machine-readable file-level declarations live in [REUSE.toml](REUSE.toml), with
canonical license texts under `LICENSES/`.

Contributors submit changes under the license already declared for the file
unless a pull request explicitly says otherwise and the maintainers agree.

## ACE-Step 1.5

Lyre does not vendor or redistribute ACE-Step. During setup,
`./scripts/lyre install` clones
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) into the ignored
`vendor/` directory and checks out the exact commit recorded in
[ACE_STEP_REVISION](ACE_STEP_REVISION). That upstream source declares the MIT
License and retains its own copyright notices.

`./scripts/lyre models*` downloads ACE-Step model artifacts from the upstream
sources selected by ACE-Step's downloader. The ACE-Step Hugging Face model
cards currently declare those artifacts as MIT-licensed. Model artifacts are
ignored by Git and are not part of Lyre's source distribution. Review the model
card and files delivered with each artifact before redistributing it; upstream
terms can change independently of this repository.

## Package dependencies

Python and JavaScript dependencies are resolved by [uv.lock](uv.lock) and
[web/package-lock.json](web/package-lock.json). They remain under their own
licenses and copyright notices. The lockfiles identify exact resolved versions;
package metadata and installed license files are the authoritative notices for
those packages.

Runnable release archives redistribute a compiled frontend. The copyright
notices and license terms for every resolved production JavaScript dependency
used to build it are preserved in
[licenses for the compiled web runtime](docs/WEB_THIRD_PARTY_LICENSES.md) and
included beside the release README.

ACE-Step's installed environment receives a small, exact security overlay and
does not retain the unused upstream Gradio packages. The versions and reviewed
exceptions are documented in
[docs/SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md).

Generated music, source audio uploaded by a user, project data, and trained
style packs are user data and are never included in this repository. You are
responsible for having the rights needed for audio and lyrics you upload or use
for training, and for evaluating the rights and suitability of generated output
for your intended use.
