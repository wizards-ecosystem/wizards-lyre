"""Cross-process per-project lock guarding every metadata mutation."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from server.storage.paths import jailed_path

# project.json and plan.json are both read-modify-written by the HTTP
# server (save_plan, patch_project) and by the worker process (merging a
# simple-mode plan_patch, and set_active_take, after a generation
# completes) -- see SPEC.md sec 5/10. Without serialization, one process
# can read a stale copy and overwrite the other's concurrent change (e.g. a
# plan PUT landing between the worker's read and write of a plan_patch
# merge silently disappears; a plan save racing generation completion can
# clobber the newly assigned active_take_id). One lock per project_id
# guards *all* metadata mutations for that project, project.json and
# plan.json alike -- they're small, infrequent writes, so sharing a single
# lock is simpler than two independent ones and still lets different
# projects proceed fully in parallel. This is a plain lock-file mutex
# (atomic exclusive create, `os.O_CREAT | os.O_EXCL`) rather than an
# in-process lock, since it must work *across* the server/worker process
# boundary, not just across threads within one of them.
_PROJECT_LOCK_TIMEOUT_SEC = 24 * 60 * 60.0
_PROJECT_LOCK_POLL_SEC = 0.05
# A lock file older than this is assumed abandoned by a crashed process
# (e.g. the worker was killed mid-write) and is reclaimed rather than
# blocking every future update to this project forever.
_PROJECT_LOCK_STALE_SEC = 30.0
_PROJECT_LOCK_STATE = threading.local()


@contextmanager
def _project_lock(project_id: str) -> Iterator[None]:
    # Keep lifecycle locks outside the project directory.  A lock inside
    # projects/<id> disappears during rmtree, allowing a waiting writer to
    # create a fresh lock (and project directory) while deletion still owns
    # the old, unlinked lock file.
    lock_path = jailed_path(".project-locks", f"{project_id}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = getattr(_PROJECT_LOCK_STATE, "held", {})
    if held.get(project_id, 0):
        held[project_id] += 1
        try:
            yield
        finally:
            held[project_id] -= 1
        return
    deadline = time.monotonic() + _PROJECT_LOCK_TIMEOUT_SEC
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            # Someone else holds the lock. On Windows, a concurrent
            # O_CREAT|O_EXCL attempt against a file another handle
            # currently has open can raise PermissionError instead of
            # FileExistsError (a sharing-violation quirk, not a real
            # permissions problem) -- treat both as "still locked".
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between our open() and stat(); retry immediately
            except PermissionError:
                time.sleep(_PROJECT_LOCK_POLL_SEC)  # transient Windows sharing violation
                continue
            if age > _PROJECT_LOCK_STALE_SEC:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for project lock: {project_id}") from None
            time.sleep(_PROJECT_LOCK_POLL_SEC)
    try:
        held[project_id] = 1
        _PROJECT_LOCK_STATE.held = held
        stop_refresh = threading.Event()

        def _refresh_lock() -> None:
            while not stop_refresh.wait(_PROJECT_LOCK_STALE_SEC / 3):
                try:
                    lock_path.touch()
                except OSError:
                    return

        refresh_thread = threading.Thread(target=_refresh_lock, daemon=True)
        refresh_thread.start()
        yield
    finally:
        stop_refresh.set()
        refresh_thread.join()
        del held[project_id]
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


# Public lifecycle guard for operations (worker execution and streamed
# uploads) that perform several filesystem writes over an extended period.
# Deletion takes this same cross-process lock and therefore waits until the
# complete operation, not merely one metadata write, has finished.
project_lifecycle_lock = _project_lock
