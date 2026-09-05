# Third-party notices

The Wizard's Lyre @VERSION@ is distributed under the [MIT License](LICENSE).

The release archive does not redistribute ACE-Step source or model weights.
During setup it fetches ACE-Step 1.5 source at revision
`@ACE_REVISION@` from the
[official repository](https://github.com/ace-step/ACE-Step-1.5), and downloads
the model artifacts selected by the user. Those materials retain their own
license files, notices, model cards, and use conditions.

Python dependencies are installed according to the lockfile used for source
revision `@SOURCE_REVISION@`; their terms remain with their respective authors.
The compiled web dependencies' copyright notices and license terms are
included in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Corresponding
source, lockfiles, full notices, and build instructions are available from the
[Lyre repository](https://github.com/wizards-ecosystem/wizards-lyre).

User projects, uploaded audio, generated audio, and trained style packs are not
part of this release. Users remain responsible for the rights and permissions
needed for their own inputs and uses.
