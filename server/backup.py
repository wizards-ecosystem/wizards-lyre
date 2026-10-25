"""Consistent backup and restore of a Lyre install (SPEC.md sec 16.7).

    ./scripts/lyre backup <empty-or-new-directory>
    ./scripts/lyre restore <backup-directory>

A backup is the jobs database, the API key file, and the project tree, taken
together while nothing writes to them. The database is copied with SQLite's
online backup API, never as a raw file (the WAL may hold committed pages the
main file does not). Staged renders and lock files are not copied: they are
in-flight state, not data. `backup.json` records what was copied, with sizes
and sha256, so a restore can be checked against it.

Both commands refuse to run while a worker holds its lease or the server is
listening on its port: a copy taken around a live writer is not a matching
set. Restore only writes into an empty install, so it can never mix a backup
with newer files.

The export ZIP is a delivery bundle for one take; this is the backup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from server import config, jobs
from server.api_keys import keys_path
from server.generation import artifacts

BACKUP_SCHEMA = "lyre.backup/1"
# In-flight or derived state under projects/, never part of a backup.
_SKIP_NAMES = {".staging", ".project-locks"}
_DB_SUFFIXES = ("", "-wal", "-shm", "-journal")


class BackupError(RuntimeError):
    pass


def _writers_running() -> str | None:
    """Why it is unsafe to copy now, or None."""
    db = config.db_path()
    if db.exists():
        with closing(sqlite3.connect(db, timeout=5.0)) as conn:
            conn.row_factory = sqlite3.Row
            try:
                lease = conn.execute(
                    "SELECT heartbeat_at FROM worker_lease WHERE id = 1"
                ).fetchone()
                running = conn.execute(
                    "SELECT COUNT(*) AS n FROM jobs WHERE status = 'running'"
                ).fetchone()
            except sqlite3.OperationalError:
                lease, running = None, None
        if lease is not None and not jobs._is_stale(
            lease["heartbeat_at"], jobs.WORKER_LEASE_STALE_AFTER_SEC
        ):
            return "a worker is running (stop ./scripts/lyre worker first)"
        if running is not None and running["n"]:
            return (
                f"{running['n']} job(s) are marked running; stop the worker and let them "
                "settle (or wait a minute for stale-job recovery) first"
            )
    try:
        with socket.create_connection(("127.0.0.1", config.lyre_port()), timeout=0.5):
            return f"the server is listening on 127.0.0.1:{config.lyre_port()} (stop it first)"
    except OSError:
        return None


def _db_files(db: Path) -> list[Path]:
    return [db.with_name(db.name + suffix) for suffix in _DB_SUFFIXES]


def _copy_projects(source: Path, target: Path, db: Path, keys: Path) -> list[dict[str, Any]]:
    skip = {path.resolve() for path in _db_files(db)} | {keys.resolve()}
    copied: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in _SKIP_NAMES for part in relative.parts):
            continue
        if path.resolve() in skip or path.is_dir():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": artifacts.sha256_file(destination),
            }
        )
    return copied


def _sqlite_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(source, timeout=10.0)) as src,
        closing(sqlite3.connect(target)) as dst,
    ):
        src.backup(dst)
        integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise BackupError(f"the copied database failed its integrity check: {integrity}")


def create_backup(destination: Path) -> dict[str, Any]:
    reason = _writers_running()
    if reason:
        raise BackupError(f"not backing up while {reason}")
    if destination.exists() and any(destination.iterdir()):
        raise BackupError(f"{destination} is not empty; choose a new directory")
    db = config.db_path()
    projects = config.projects_dir()
    keys = keys_path()
    destination.mkdir(parents=True, exist_ok=True)

    database: dict[str, Any] | None = None
    if db.exists():
        _sqlite_copy(db, destination / "lyre.db")
        copy = destination / "lyre.db"
        database = {"bytes": copy.stat().st_size, "sha256": artifacts.sha256_file(copy)}
    key_file: dict[str, Any] | None = None
    if keys.exists():
        shutil.copy2(keys, destination / "api-keys.json")
        os.chmod(destination / "api-keys.json", 0o600)
        key_file = {"sha256": artifacts.sha256_file(destination / "api-keys.json")}
    files = (
        _copy_projects(projects, destination / "projects", db, keys) if projects.exists() else []
    )

    record = {
        "schema": BACKUP_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "app": {"version": artifacts.app_version(), "revision": artifacts.app_revision()},
        "database": database,
        "api_keys": key_file,
        "projects": sum(1 for f in files if f["path"].endswith("/project.json")),
        "files": files,
    }
    (destination / "backup.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _verify(source: Path, record: dict[str, Any]) -> None:
    for entry in record["files"]:
        path = source / "projects" / entry["path"]
        if not path.is_file() or artifacts.sha256_file(path) != entry["sha256"]:
            raise BackupError(f"backup file {entry['path']} is missing or changed")
    database = record.get("database")
    if database and artifacts.sha256_file(source / "lyre.db") != database["sha256"]:
        raise BackupError("the backed-up database does not match backup.json")


def restore_backup(source: Path) -> dict[str, Any]:
    reason = _writers_running()
    if reason:
        raise BackupError(f"not restoring while {reason}")
    manifest = source / "backup.json"
    if not manifest.is_file():
        raise BackupError(f"{source} has no backup.json; it is not a Lyre backup")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") != BACKUP_SCHEMA:
        raise BackupError(f"unsupported backup schema: {record.get('schema')!r}")
    _verify(source, record)

    db = config.db_path()
    projects = config.projects_dir()
    keys = keys_path()
    existing = [
        p
        for p in (projects.iterdir() if projects.exists() else [])
        if p.name not in _SKIP_NAMES and p.resolve() not in {f.resolve() for f in _db_files(db)}
    ]
    if existing or db.exists() or keys.exists():
        raise BackupError(
            "the install is not empty; restore into a fresh checkout or move "
            f"{projects} and {db} aside first"
        )
    projects.mkdir(parents=True, exist_ok=True)
    for entry in record["files"]:
        target = projects / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "projects" / entry["path"], target)
    if record.get("database"):
        _sqlite_copy(source / "lyre.db", db)
    if record.get("api_keys"):
        shutil.copy2(source / "api-keys.json", keys)
        os.chmod(keys, 0o600)
    # Bring the restored database up to this build's schema.
    jobs.init_db()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyre", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backup").add_argument("destination", type=Path)
    commands.add_parser("restore").add_argument("source", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            record = create_backup(args.destination)
            print(
                f"backed up {record['projects']} project(s), {len(record['files'])} file(s) "
                f"to {args.destination}"
            )
        else:
            record = restore_backup(args.source)
            print(f"restored {record['projects']} project(s) from {args.source}")
    except BackupError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
