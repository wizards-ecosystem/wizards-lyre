"""Shared fixtures for Lyre's test suite.

Every fixture here is GPU-free: runtime paths are redirected into pytest's
`tmp_path` and the job backend is pinned to `worker.mock_worker`, so the
default `pytest` run never loads CUDA, ACE-Step, or real weights (SPEC.md
sec 11).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from server.app import app
from worker.run_worker import run_loop


@pytest.fixture(autouse=True)
def _reset_mock_worker_state() -> Iterator[None]:
    """`worker.mock_worker` tracks a simulated "loaded" flag at module scope
    (mirroring `worker.acestep_worker`'s real `_STATE`) so tests can exercise
    `worker/run_worker.py`'s republish-after-recovery behavior. Reset it
    around every test, or an earlier run's job would leak into a later test
    expecting a fresh "nothing loaded yet" state.
    """
    import worker.mock_worker as mock_worker_module

    mock_worker_module._simulated_loaded_dit_profile = None
    yield
    mock_worker_module._simulated_loaded_dit_profile = None


@pytest.fixture()
def lyre_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Lyre's runtime paths at a throwaway directory and force the
    mocked worker backend. Returns the tmp root, for tests that need to reach
    the same paths directly.
    """
    monkeypatch.setenv("BARD_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("BARD_DB_PATH", str(tmp_path / "bard.db"))
    # Tests always use the mocked worker; the real acestep_worker is
    # production's default (see server/jobs.py) and is exercised only by the
    # manual, non-pytest scripts/smoke-gpu.py.
    monkeypatch.setenv("BARD_WORKER", "mock")
    return tmp_path


@pytest.fixture()
def api_client(lyre_env: Path) -> Iterator[TestClient]:
    """HTTP client with *no* worker draining the queue.

    `enqueue_job` only inserts a `queued` row (SPEC.md sec 5), so jobs posted
    through this client stay `queued` -- which is what tests of enqueue-time
    validation and of job listing/recovery want to observe.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client(lyre_env: Path) -> Iterator[TestClient]:
    """The full local stack: HTTP server plus a worker draining the same
    SQLite queue, so an enqueued job actually runs and produces a take.

    The thread stands in for the dedicated `worker/run_worker.py` process
    that runs in production, fast-polled so tests stay quick. Use
    `api_client` instead when a job must stay `queued`.
    """
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=run_loop, args=(stop_event, 0.01), daemon=True)
    worker_thread.start()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)
