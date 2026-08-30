# Architecture

```
Browser (127.0.0.1:8421)
        │
        ▼
   web/          Vite + React + TypeScript SPA
        │        dev: Vite proxies /api → FastAPI
        │        prod: FastAPI serves web/dist directly
        ▼
   server/       FastAPI. HTTP, project files, job rows.
        │        Never imports acestep. Never touches CUDA.
        │
        │  SQLite (projects/lyre.db)
        │  queued → running → done | error
        ▼
   worker/       Separate OS process. Loads ACE-Step and CUDA.
                 Runs one job at a time under a GPU lease.
                 │
                 ▼
            projects/<id>/   project.json, plan.json, takes/, loras/
```

## Why three processes

The split is the single most load-bearing design decision (SPEC.md §5).

**The server never imports `acestep` or `torch`.** A generation can take
minutes and a native CUDA crash can take the whole process down. Keeping that
in a separate OS process means a GPU failure fails the *job*, not HTTP — the UI
stays up and reports the error. `server/jobs/worker_registry.py` is the only
module that can reach a worker backend, and it only does so from worker-side
code paths.

**They communicate through SQLite, not a socket.** The queue is the IPC. The
server inserts `queued` rows and never runs anything; the worker claims,
executes, and updates them. This means either side can restart independently
without the other noticing, and a crashed worker leaves recoverable state on
disk rather than in memory.

**A claimed job carries a heartbeat lease.** `run_claimed_job` touches
`heartbeat_at` while it works. If the worker dies mid-job the heartbeat stops,
and the next `reclaim_stale_jobs` requeues the job — or, past `MAX_ATTEMPTS`,
marks it `error` rather than leaving it `running` forever.

**One GPU occupant.** A separate `worker_lease` row means two worker processes
cannot both load models. The worker publishes its readiness, loaded profile,
and per-profile capability into SQLite so `/api/health` can report real state
without the server ever asking CUDA anything.

## Package layout

| Path | Role |
|---|---|
| `server/app.py` | Route definitions and exception→HTTP mapping. Thin. |
| `server/config.py` | Paths and bind settings from `LYRE_*`. |
| `server/storage/` | Project/plan/take persistence and the path jail. |
| `server/jobs/` | The queue: schema, validation, enqueue/claim, execution, deletion. |
| `worker/run_worker.py` | The process entry point. Owns the lease and the poll loop. |
| `worker/acestep_worker/` | The real ACE-Step adapter. |
| `worker/mock_worker.py` | GPU-free stand-in. Writes silent WAVs. |
| `web/src/` | The SPA. |

`server/storage`, `server/jobs`, and `worker/acestep_worker` are packages whose
`__init__.py` re-exports the module surface, so callers use `storage.<name>` as
if each were still one module. See the note in
[CONTRIBUTING.md](../CONTRIBUTING.md) about patching these in tests.

## Data on disk

JSON is the source of truth for metadata; audio and weights are never in git.

```
projects/<project_id>/
  project.json          id, title, timestamps, dit_profile, active_take_id, favorite
  plan.json             caption, lyrics, bpm, keyscale, duration, sections[]
  takes/<take_id>/
    mix.wav             the archive copy
    meta.json           seed, task_type, parent_take_id, score, error, notes
    lyrics.lrc          when ACE-Step supplied timestamps
  uploads/              drag-dropped source audio
  loras/<lora_id>/      trained style packs
```

**Takes are immutable.** Cover, repaint, extract, lego, and complete all create
a *new* take with `parent_take_id` pointing at the source. Nothing overwrites
`mix.wav` in place, which is what makes the history walkable and A/B compare
meaningful. The only mutable fields on a take are the user's own annotations
(`favorite`, `notes`).

**Every metadata write is locked.** Both processes read-modify-write the same
`project.json` and `plan.json`, so all mutations go through a cross-process
lock file (`server/storage/locks.py`) and land via atomic
temp-file-plus-`os.replace`. A reader never sees a partial file.

## Where the constraints are enforced

- **Path jail** — `server/storage/paths.py`. Nothing outside `projects/` or
  `output/` is writable.
- **Localhost bind** — `server/config.py`, asserted by
  `tests/test_spec_lock.py`, which fails the build on a public bind host.
- **No other engines** — `tests/test_spec_lock.py` scans Lyre's own source for
  forbidden imports and vendor names.
- **No GPU in tests** — the default backend under `pytest` is
  `worker/mock_worker.py`, and nothing imports `acestep` or `torch` at module
  scope, so CI installs neither.
