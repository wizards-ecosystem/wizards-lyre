# Upgrading to the completion release

This is for anyone running an existing Lyre install (0.2.0 or a checkout
from before the bounded completion in [SPEC.md section 16](../SPEC.md)).
Every change to stored data is additive, and the upgrade migrates in place
when the server or worker starts. Take a backup first anyway: it is the only
way back.

## What changes on disk

| Where | Change | Old data |
|---|---|---|
| `projects/lyre.db`, `jobs` table | New columns: `attempt_token`, `resolved_json`, `remote_handle`, `submission_id`, `retry_of`, and a unique index on `(project_id, submission_id)`. Job payloads gain `creation` and `submission` records. | Old rows read as before; their new columns are null. |
| `plan.json` | New fields: `creation_mode`, `field_origins`, `revision`, `brief`, `preset`. `duration_sec` may be null. | Read with the mode derived by the old rule and every origin `unknown`. The file is rewritten only when the plan is next saved. |
| `takes/<id>/meta.json` | New fields: `job_id` and `provenance` (requested and actual values, model, files with sha256). | Read with `provenance: null` and `provenance_complete: false`. Nothing is backfilled from the current plan. |
| `projects/.staging/` | New: renders in progress. Never listed as takes, and cleaned automatically once their job is no longer running. | None. |
| Export ZIP | Gains `manifest.json` and the take's `lyrics.lrc`. `mix<ext>`, `project.json`, `plan.json`, and stems are where they were. | None. |

## Behavior you will notice

- **Jobs queued before the upgrade are held.** A generate, cover, repaint,
  extract, lego, or complete job that was still queued has no snapshot of
  what it should render, and Lyre will not guess from the current plan. It
  stays queued with stage `needs_confirmation`. In the studio choose Retry
  (it is resubmitted from the current plan, as a new job that names the old
  one in `retry_of`) or Cancel. Agents use `POST /v1/jobs/{id}/retry` or
  `POST /v1/jobs/{id}/cancel`. Style-pack training jobs are not affected.
- **New plans start with caption rewrite off.** Existing plans keep their
  stored value.
- **A finished take no longer replaces the active take.** The first take of
  a project still becomes active; after that, keeping a take is your choice.
- **Some requests are now refused before rendering:** Instrumental with real
  lyrics, BPM outside 30-300, duration outside 10-600 seconds, an
  unrecognized key, meter, or language, and less than 512 MB free disk.
- **Editing during a render no longer waits for it**, and the planner's
  output only updates the plan if you have not edited it since you pressed
  Generate.

## Remote GPU

The host image now includes `server/generation/`, so its build id changes.
A host on the old image still renders, from the frozen plan snapshot, but
with the old rules for which fields the language model may change; the
studio marks it "host runs an older build". Rebuild it with
`./scripts/lyre remote-image` (see [remote-gpu.md](remote-gpu.md)). Nothing in
the upgrade starts, stops, or rents a machine.

## Back up before you upgrade

1. Stop the server and the worker (Ctrl+C in their terminals). Backup and
   restore refuse to run while a worker holds its lease, while a job is
   marked running, or while something listens on the Lyre port.
2. Run:

   ```sh
   ./scripts/lyre backup ~/lyre-backups/2026-10-25
   ```

   The directory must be new or empty. It receives `lyre.db` (copied with
   SQLite's online backup API and integrity-checked, never as a raw file),
   `api-keys.json` (mode `0600`) when you have keys, `projects/` without
   staged renders or lock files, and `backup.json` listing every file with
   its size and sha256.

Copying the `projects/` directory by hand while Lyre runs is not a backup:
the database inside it may be mid-transaction, and a take may be half
promoted. Use the command, with the writers stopped.

## Upgrade

```sh
git pull                # or unpack the new release
./scripts/lyre install
./scripts/lyre server   # migrates on start
./scripts/lyre worker
```

If the first start is interrupted part-way through the migration, start it
again: each step checks what is already there, so a half-finished migration
completes on the next start.

## Restore

Restore writes only into an empty install, so it cannot mix a backup with
newer files.

1. Stop the server and the worker.
2. Move the current `projects/` aside (for example to `projects.before-restore/`).
3. Run:

   ```sh
   ./scripts/lyre restore ~/lyre-backups/2026-10-25
   ```

   Every file is checked against `backup.json` before anything is written;
   a changed or missing file stops the restore. The restored database is
   then brought up to the running build's schema.
4. Start the server and worker, and open a project to check it.

## Downgrading

Running an older Lyre build on an upgraded install is not supported. It
would run held `needs_confirmation` jobs against the live plan and could
replace kept takes again. Reverting the code is not a database rollback: to
go back, check out the older version and restore the backup you took before
upgrading into it.
