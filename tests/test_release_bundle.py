"""The downloadable release is reproducible, runnable, and intentionally slim."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import build_release

VERSION = "1.2.3"
ACE_REVISION = "a" * 40
SOURCE_REVISION = "b" * 40
SOURCE_EPOCH = "1700000000"


def _write(path: Path, content: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _release_source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _write(
        root / "pyproject.toml",
        f'[project]\nname = "wizards-lyre"\nversion = "{VERSION}"\n',
    )
    _write(root / "web/package.json", json.dumps({"version": VERSION}))
    _write(root / "ACE_STEP_REVISION", f"{ACE_REVISION}\n")
    _write(root / "LICENSE", "MIT fixture\n")
    _write(root / "uv.lock")
    _write(root / "requirements/ace-step-security.txt")
    _write(root / "scripts/lyre", "#!/usr/bin/env bash\n")
    _write(root / "scripts/smoke-gpu.py")
    _write(root / "scripts/live_stack_check.py")
    _write(root / "server/__init__.py")
    _write(root / "server/app.py")
    _write(root / "server/__pycache__/app.cpython-312.pyc")
    _write(root / "worker/__init__.py")
    _write(root / "worker/run_worker.py")
    _write(root / "web/dist/index.html", '<script src="/assets/app.js"></script>\n')
    _write(root / "web/dist/assets/app.js", "console.log('lyre')\n")
    _write(root / "web/dist/assets/app.js.map", "must not ship\n")
    _write(root / "web/src/App.tsx", "must not ship\n")
    _write(root / "tests/test_private.py", "must not ship\n")
    _write(root / ".github/workflows/ci.yml", "must not ship\n")
    _write(
        root / "docs/BUNDLE_README.md",
        "# Lyre @VERSION@\n@SOURCE_REVISION@\n@ACE_REVISION@\n[License](LICENSE)\n",
    )
    _write(
        root / "docs/BUNDLE_NOTICES.md",
        "# Notices @VERSION@\n@SOURCE_REVISION@\n@ACE_REVISION@\n"
        "[Licenses](THIRD_PARTY_LICENSES.md)\n",
    )
    _write(root / "docs/WEB_THIRD_PARTY_LICENSES.md", "# Compiled dependency licenses\n")
    return root


def _build(root: Path, output: Path) -> build_release.ReleaseResult:
    return build_release.build_release(
        root=root,
        output_dir=output,
        requested_version=f"v{VERSION}",
        allow_dirty=True,
    )


def _file_members(names: list[str]) -> set[str]:
    prefix = f"wizards-lyre-{VERSION}-linux-x86_64/"
    return {
        name.removeprefix(prefix)
        for name in names
        if name.startswith(prefix) and not name.endswith("/")
    }


def test_release_archives_are_reproducible_and_contain_only_runtime_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _release_source(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", SOURCE_REVISION)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", SOURCE_EPOCH)

    first = _build(root, tmp_path / "first")
    second = _build(root, tmp_path / "second")

    assert first.tar_path.read_bytes() == second.tar_path.read_bytes()
    assert first.zip_path.read_bytes() == second.zip_path.read_bytes()
    assert first.checksums_path.read_bytes() == second.checksums_path.read_bytes()

    with tarfile.open(first.tar_path, "r:gz") as archive:
        tar_files = _file_members(
            [member.name for member in archive.getmembers() if member.isfile()]
        )
    with zipfile.ZipFile(first.zip_path) as archive:
        zip_files = _file_members(archive.namelist())
        launcher = archive.getinfo(f"{first.archive_root}/scripts/lyre")
        manifest = json.loads(archive.read(f"{first.archive_root}/LYRE_RELEASE.json"))
        readme = archive.read(f"{first.archive_root}/README.md").decode()

    assert tar_files == zip_files
    assert {
        "ACE_STEP_REVISION",
        "LICENSE",
        "LYRE_RELEASE.json",
        "README.md",
        "THIRD_PARTY_LICENSES.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "requirements/ace-step-security.txt",
        "scripts/live_stack_check.py",
        "scripts/lyre",
        "scripts/smoke-gpu.py",
        "server/__init__.py",
        "server/app.py",
        "uv.lock",
        "web/dist/assets/app.js",
        "web/dist/index.html",
        "worker/__init__.py",
        "worker/run_worker.py",
    } == zip_files
    assert stat.S_IMODE(launcher.external_attr >> 16) == 0o755
    assert "@VERSION@" not in readme
    assert VERSION in readme
    assert manifest["version"] == VERSION
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["source_dirty"] is True
    assert manifest["ace_step_revision"] == ACE_REVISION
    assert set(manifest["files"]) == zip_files - {"LYRE_RELEASE.json"}

    with zipfile.ZipFile(first.zip_path) as archive:
        for name, record in manifest["files"].items():
            content = archive.read(f"{first.archive_root}/{name}")
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            assert len(content) == record["size"]
        for document in ("README.md", "THIRD_PARTY_NOTICES.md"):
            content = archive.read(f"{first.archive_root}/{document}").decode()
            for target in re.findall(r"\[[^]]*]\(([^)]+)\)", content):
                if not target.startswith(("https://", "http://")):
                    assert target in zip_files

    checksum_lines = first.checksums_path.read_text(encoding="ascii").splitlines()
    expected = {
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in (first.tar_path, first.zip_path)
    }
    assert set(checksum_lines) == expected


def test_release_version_must_match_both_package_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _release_source(tmp_path)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", SOURCE_EPOCH)
    _write(root / "web/package.json", json.dumps({"version": "9.9.9"}))

    with pytest.raises(RuntimeError, match=r"web/package.json=9\.9\.9"):
        _build(root, tmp_path / "release")
