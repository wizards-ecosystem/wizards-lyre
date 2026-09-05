#!/usr/bin/env python3
"""Build reproducible, runnable The Wizard's Lyre release archives.

The archives intentionally contain only the local runtime surface. ACE-Step
source and model weights remain separate downloads performed by scripts/lyre.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import time
import tomllib
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_NAME = "The Wizard's Lyre"
PROJECT_SLUG = "wizards-lyre"
RELEASE_TARGET = "linux-x86_64"
SOURCE_URL = "https://github.com/wizards-ecosystem/wizards-lyre"
MINIMUM_ZIP_EPOCH = 315_532_800  # 1980-01-01, the beginning of ZIP time.
VERSION_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
)


@dataclass(frozen=True)
class PayloadFile:
    data: bytes
    mode: int = 0o644


@dataclass(frozen=True)
class SourceState:
    revision: str
    epoch: int
    dirty: bool


@dataclass(frozen=True)
class ReleaseResult:
    tar_path: Path
    zip_path: Path
    checksums_path: Path
    archive_root: str


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=False,
        text=True,
    )


def _source_state(root: Path, *, allow_dirty: bool) -> SourceState:
    revision_result = _run_git(root, "rev-parse", "HEAD")
    top_level_result = _run_git(root, "rev-parse", "--show-toplevel")
    is_checkout_root = (
        revision_result.returncode == 0
        and top_level_result.returncode == 0
        and Path(top_level_result.stdout.strip()).resolve() == root.resolve()
    )
    if is_checkout_root:
        revision = revision_result.stdout.strip()
        status_result = _run_git(root, "status", "--porcelain", "--untracked-files=all")
        if status_result.returncode != 0:
            raise RuntimeError(status_result.stderr.strip() or "could not inspect Git status")
        dirty = bool(status_result.stdout.strip())
        if dirty and not allow_dirty:
            raise RuntimeError(
                "the source tree has uncommitted changes; commit them or pass --allow-dirty "
                "for a non-publishable local build"
            )
        epoch_result = _run_git(root, "show", "-s", "--format=%ct", "HEAD")
        if epoch_result.returncode != 0:
            raise RuntimeError(epoch_result.stderr.strip() or "could not read commit timestamp")
        epoch = int(epoch_result.stdout.strip())
    elif allow_dirty:
        revision = os.environ.get("GITHUB_SHA", "unknown")
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", str(MINIMUM_ZIP_EPOCH)))
        dirty = True
    else:
        raise RuntimeError("release archives must be built from a Git checkout")

    if source_date_epoch := os.environ.get("SOURCE_DATE_EPOCH"):
        epoch = int(source_date_epoch)
    return SourceState(revision=revision, epoch=epoch, dirty=dirty)


def _project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    project = document.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise RuntimeError("pyproject.toml is missing project.version")
    return project["version"]


def _web_version(root: Path) -> str:
    document: Any = json.loads((root / "web" / "package.json").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("version"), str):
        raise RuntimeError("web/package.json is missing version")
    return document["version"]


def _normalized_version(requested: str | None, root: Path) -> str:
    project_version = _project_version(root)
    web_version = _web_version(root)
    version = (requested or project_version).removeprefix("v")
    if VERSION_PATTERN.fullmatch(version) is None:
        raise RuntimeError(f"release version is not valid SemVer: {version!r}")
    mismatches = [
        f"pyproject.toml={project_version}" if project_version != version else "",
        f"web/package.json={web_version}" if web_version != version else "",
    ]
    mismatches = [mismatch for mismatch in mismatches if mismatch]
    if mismatches:
        raise RuntimeError(
            f"requested release {version} does not match " + " and ".join(mismatches)
        )
    return version


def _read_file(path: Path) -> bytes:
    if path.is_symlink():
        raise RuntimeError(f"release input must not be a symbolic link: {path}")
    if not path.is_file():
        raise RuntimeError(f"required release input is missing: {path}")
    return path.read_bytes()


def _add_tree(payload: dict[str, PayloadFile], root: Path, source: str) -> None:
    source_root = root / source
    if not source_root.is_dir():
        raise RuntimeError(f"required release directory is missing: {source_root}")
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"release input must not be a symbolic link: {path}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".map"}:
            continue
        payload[relative] = PayloadFile(_read_file(path))


def _render_template(path: Path, replacements: dict[str, str]) -> bytes:
    text = _read_file(path).decode("utf-8")
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    leftover = re.findall(r"@[A-Z_]+@", text)
    if leftover:
        raise RuntimeError(f"unresolved release template marker(s) in {path}: {leftover}")
    return text.encode("utf-8")


def _payload(root: Path, version: str, state: SourceState) -> dict[str, PayloadFile]:
    ace_revision = _read_file(root / "ACE_STEP_REVISION").decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", ace_revision) is None:
        raise RuntimeError("ACE_STEP_REVISION must contain one 40-character Git commit ID")

    payload: dict[str, PayloadFile] = {}
    for relative in (
        "ACE_STEP_REVISION",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "requirements/ace-step-security.txt",
    ):
        payload[relative] = PayloadFile(_read_file(root / relative))
    for relative in (
        "scripts/lyre",
        "scripts/smoke-gpu.py",
        "scripts/live_stack_check.py",
    ):
        payload[relative] = PayloadFile(_read_file(root / relative), 0o755)

    for tree in ("server", "worker", "web/dist"):
        _add_tree(payload, root, tree)
    if "web/dist/index.html" not in payload:
        raise RuntimeError("web/dist/index.html is missing; build the production UI first")

    replacements = {
        "@VERSION@": version,
        "@SOURCE_REVISION@": state.revision,
        "@ACE_REVISION@": ace_revision,
    }
    payload["README.md"] = PayloadFile(
        _render_template(root / "docs" / "BUNDLE_README.md", replacements)
    )
    payload["THIRD_PARTY_NOTICES.md"] = PayloadFile(
        _render_template(root / "docs" / "BUNDLE_NOTICES.md", replacements)
    )
    payload["THIRD_PARTY_LICENSES.md"] = PayloadFile(
        _read_file(root / "docs" / "WEB_THIRD_PARTY_LICENSES.md")
    )

    file_manifest = {
        name: {"sha256": hashlib.sha256(item.data).hexdigest(), "size": len(item.data)}
        for name, item in sorted(payload.items())
    }
    created_at = datetime.fromtimestamp(state.epoch, UTC).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "name": PROJECT_NAME,
        "version": version,
        "target": RELEASE_TARGET,
        "source_url": SOURCE_URL,
        "source_revision": state.revision,
        "source_dirty": state.dirty,
        "source_date_epoch": state.epoch,
        "created_at": created_at,
        "ace_step_revision": ace_revision,
        "files": file_manifest,
    }
    payload["LYRE_RELEASE.json"] = PayloadFile(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return payload


def _archive_directories(archive_root: str, names: list[str]) -> list[str]:
    directories = {archive_root}
    for name in names:
        parent = PurePosixPath(archive_root, name).parent
        while str(parent) != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return sorted(directories, key=lambda value: (value.count("/"), value))


def _write_tar_gz(
    destination: Path,
    archive_root: str,
    payload: dict[str, PayloadFile],
    epoch: int,
) -> None:
    with (
        destination.open("wb") as raw_handle,
        gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=epoch) as gzip_handle,
        tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for directory in _archive_directories(archive_root, list(payload)):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = epoch
            archive.addfile(info)
        for name, item in sorted(payload.items()):
            info = tarfile.TarInfo(PurePosixPath(archive_root, name).as_posix())
            info.size = len(item.data)
            info.mode = item.mode
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = epoch
            archive.addfile(info, fileobj=_BytesReader(item.data))


class _BytesReader:
    """Minimal file object accepted by tarfile without another data copy."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        result = self._data[self._offset : self._offset + size]
        self._offset += len(result)
        return result


def _write_zip(
    destination: Path,
    archive_root: str,
    payload: dict[str, PayloadFile],
    epoch: int,
) -> None:
    zip_time = time.gmtime(max(epoch, MINIMUM_ZIP_EPOCH))[:6]
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for directory in _archive_directories(archive_root, list(payload)):
            info = zipfile.ZipInfo(f"{directory}/", date_time=zip_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFDIR | 0o755) << 16
            archive.writestr(info, b"")
        for name, item in sorted(payload.items()):
            info = zipfile.ZipInfo(PurePosixPath(archive_root, name).as_posix(), date_time=zip_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | item.mode) << 16
            archive.writestr(info, item.data, compresslevel=9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release(
    *,
    root: Path,
    output_dir: Path,
    requested_version: str | None = None,
    allow_dirty: bool = False,
) -> ReleaseResult:
    root = root.resolve()
    version = _normalized_version(requested_version, root)
    state = _source_state(root, allow_dirty=allow_dirty)
    payload = _payload(root, version, state)
    archive_root = f"{PROJECT_SLUG}-{version}-{RELEASE_TARGET}"

    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{archive_root}.tar.gz"
    zip_path = output_dir / f"{archive_root}.zip"
    checksums_path = output_dir / "SHA256SUMS"
    _write_tar_gz(tar_path, archive_root, payload, state.epoch)
    _write_zip(zip_path, archive_root, payload, state.epoch)
    checksum_lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted((tar_path, zip_path), key=lambda p: p.name)
    ]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    return ReleaseResult(tar_path, zip_path, checksums_path, archive_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="release version or v-prefixed tag; defaults to pyproject.toml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/release"),
        help="artifact directory (default: dist/release)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit an uncommitted/non-Git source tree for local testing only",
    )
    return parser


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    args = _parser().parse_args()
    try:
        result = build_release(
            root=root,
            output_dir=(
                root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
            ),
            requested_version=args.version,
            allow_dirty=args.allow_dirty,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"release build failed: {exc}") from exc

    print(f"Built {result.tar_path}")
    print(f"Built {result.zip_path}")
    print(f"Wrote {result.checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
