# HTTP API

All JSON, all under `http://127.0.0.1:8421` by default. The default server
has no authentication; see [SECURITY.md](../SECURITY.md). The optional Remote
GPU host has its own, authenticated protocol; see
[remote-gpu.md](remote-gpu.md#the-host-protocol).

[SPEC.md](../SPEC.md) section 15 is the agent API. This server now serves the
typed `/v1` reads and the studio-loop writes (projects, plan, jobs, uploads,
annotations, active take, delete, style packs, export), the job object,
`POST /v1/jobs/{job_id}/cancel`, stable artifact GETs, and `Idempotency-Key`
on every `/v1` POST. API keys and a non-loopback bind are opt-in and off
unless you set them ([Configuration](CONFIGURATION.md)). One-shot
`POST /v1/songs` is served. The MCP server is served at `POST /mcp` on the
same port; setup for Claude Desktop and Cursor is in [MCP](MCP.md). SSE,
webhooks, and signed artifact tokens are not served yet. The design notes
are in [adr/0001-agent-api.md](adr/0001-agent-api.md). When a phase mounts a
`/api` or `/v1` route, this table gains a row in the same change: the test
that reads this file fails if the table and the app disagree. `POST /mcp` is
documented in [MCP](MCP.md) rather than this table.

The server is FastAPI, so the authoritative, always-current reference is the
generated schema while it is running:

- **<http://127.0.0.1:8421/docs>:** interactive Swagger UI
- **<http://127.0.0.1:8421/openapi.json>:** raw OpenAPI schema

This page is the orientation; the schema is the specification.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Worker readiness, loaded DiT profile, GPU status. |
| `GET` | `/api/projects` | Project list, newest-updated first. |
| `POST` | `/api/projects` | Create from `{ title?, query? }`. |
| `GET` | `/api/capabilities` | Worker state, profiles (files, validated, loaded, supported), fixed LM, disk, presets, limits. |
| `POST` | `/api/projects/{id}/preset` | Preview a preset, or apply it with `apply: true` (keeps user fields and lyrics). |
| `GET` | `/api/projects/{id}` | Project + plan + takes in one response. |
| `PATCH` | `/api/projects/{id}` | Update `title`, `dit_profile`, or `favorite`. |
| `DELETE` | `/api/projects/{id}` | Delete the project and cancel its queued jobs. `204`. |
| `PUT` | `/api/projects/{id}/plan` | Replace `plan.json`. Validated and normalized. |
| `GET` | `/api/projects/{id}/takes` | Take metadata list. |
| `POST` | `/api/projects/{id}/active_take` | Set the active take. Rejects a failed take. |
| `PATCH` | `/api/projects/{id}/takes/{take_id}` | Update `favorite` / `notes` only. |
| `GET` | `/api/projects/{id}/takes/{take_id}/audio` | Stream `mix.wav` or `mix.mp3`. |
| `GET` | `/api/projects/{id}/takes/{take_id}/lrc` | Timestamped lyrics. `404` when the take has none. |
| `GET` | `/api/projects/{id}/export` | Zip: `project.json`, `plan.json`, active mix, optional stems. |
| `POST` | `/api/projects/{id}/uploads` | Upload a WAV/MP3 as a cover/repaint source. |
| `GET` | `/api/projects/{id}/loras` | Trained style packs for this project. |
| `POST` | `/api/projects/{id}/jobs` | Enqueue a job. Returns the `queued` row. |
| `GET` | `/api/jobs/{job_id}` | One job's status. |
| `GET` | `/api/jobs` | Recent jobs. Filters: `project_id`, `action`, `active`, `limit`. |
| `POST` | `/api/jobs/{job_id}/cancel` | Cooperative cancel, same as `/v1`. |
| `POST` | `/api/jobs/{job_id}/retry` | Queue an explicit retry linked by `retry_of` (SPEC.md section 16.5). |
| `GET` | `/api/remote-gpu` | Remote GPU state: provisioner, price quote, connection, rental, host health. |
| `PUT` | `/api/remote-gpu/connection` | Connect a host you run: `{ base_url, secret }`. |
| `DELETE` | `/api/remote-gpu/connection` | Forget the connected host. |
| `POST` | `/api/remote-gpu/session` | Rent a GPU with the configured provisioner. **Billable.** |
| `DELETE` | `/api/remote-gpu/session` | Terminate the rental. Safe when nothing is running. |
| `GET` | `/v1/health` | Typed health: backend, readiness, GPU, loaded DiT, queue counts. |
| `GET` | `/v1/projects` | Page of project summaries. `limit` (default 50, max 100) and `cursor`. |
| `GET` | `/v1/capabilities` | Same report as `/api/capabilities`. |
| `POST` | `/v1/projects/{id}/preset` | Preview or apply a preset; `409` on a stale `expected_revision`. |
| `GET` | `/v1/projects/{id}` | Project, plan, and takes. |
| `GET` | `/v1/projects/{id}/takes` | Take metadata page. |
| `GET` | `/v1/jobs` | Job page. Filters: `project_id`, `action`, `active`, `limit`, `cursor`. |
| `GET` | `/v1/jobs/{job_id}` | One job object, every status. |
| `POST` | `/v1/projects` | Create from `{ title?, query? }`. |
| `PATCH` | `/v1/projects/{id}` | Update `title`, `dit_profile`, or `favorite`. |
| `DELETE` | `/v1/projects/{id}` | Delete the project and cancel its queued jobs. `204`. |
| `PUT` | `/v1/projects/{id}/plan` | Replace `plan.json`. Validated and normalized. |
| `POST` | `/v1/projects/{id}/active_take` | Set the active take. Rejects a failed take. |
| `PATCH` | `/v1/projects/{id}/takes/{take_id}` | Update `favorite` / `notes` only. |
| `POST` | `/v1/projects/{id}/uploads` | Upload a WAV/MP3 as a cover/repaint source. |
| `GET` | `/v1/projects/{id}/loras` | Trained style packs for this project. |
| `GET` | `/v1/projects/{id}/export` | Zip: `project.json`, `plan.json`, active mix, optional stems. |
| `POST` | `/v1/projects/{id}/jobs` | Enqueue a job. `202`, `Location: /v1/jobs/{job_id}`, and the job object. |
| `POST` | `/v1/jobs/{job_id}/cancel` | Cancel a queued job, or ask a running job to drop its audio. |
| `POST` | `/v1/jobs/{job_id}/retry` | Retry a failed, cancelled, or held job: `202` and the new job, whose `retry_of` names this one. |
| `POST` | `/v1/songs` | Create a project, write its plan, and queue `generate`. `202`, `Location`, and the job. |
| `GET` | `/v1/artifacts/{id}/{take_id}/mix.wav` | Stable `mix.wav` for a take. |
| `GET` | `/v1/artifacts/{id}/{take_id}/mix.mp3` | Stable `mix.mp3` for a take. |
| `GET` | `/v1/artifacts/{id}/{take_id}/lyrics.lrc` | Stable timed lyrics. `404` when the take has none. |

`/` serves the built SPA from `web/dist` when it exists, and otherwise returns
a hint telling you to build it.

## Enqueuing a job

`POST /api/projects/{id}/jobs` and `POST /v1/projects/{id}/jobs` take the same
body. `/v1` answers `202` with `Location: /v1/jobs/{job_id}` and the job
object below. `/api` answers `200` with the queue row.

`POST /api/projects/{id}/jobs`:

```json
{
  "action": "generate",
  "dit_profile": "iterate",
  "source_take_id": null,
  "source_take_ids": null,
  "upload_path": null,
  "repainting_start": 0,
  "repainting_end": -1,
  "track_name": null,
  "name": null,
  "lora_id": null,
  "audio_cover_strength": 0.7,
  "seed": -1,
  "batch_size": 1,
  "creation_mode": null
}
```

`action` is one of `generate`, `cover`, `repaint`, `extract`, `lego`,
`complete`, or `train_lora`. Only `action` is required; everything else has the
default shown.

What is validated at enqueue time, before anything reaches the GPU:

- `cover`, `repaint`, `extract`, `lego`, and `complete` need a real source:
  either `source_take_id` or an `upload_path` from the uploads endpoint.
- `extract`, `lego`, and `complete` are forced to the `studio_ops` profile.
- `lora_id` only applies to `generate`, `cover`, and `repaint`, and the pack
  must exist and have finished training.
- `train_lora` needs a non-empty `name` and at least 8 distinct source takes.
- `seed: -1` means the worker picks one and records the actual value.
- `batch_size` is forced to 1.
- `audio_cover_strength` must be within `0.0`-`1.0` (which also rejects NaN
  and infinities).
- `generate` resolves the plan by the creation contract (SPEC.md section
  16.3). `creation_mode` (`from_idea` re-plans from `query`, `from_brief`
  renders the stored brief) overrides the plan's own mode for this job.
  Unsupported metas fail here with `param` naming the field: BPM 30-300,
  duration 10-600 seconds, a key such as `C major` or `F#m`, a meter of `2`,
  `3`, `4`, `6`, `2/4`, `3/4`, `4/4`, or `6/8`, and an ACE-Step language code.
  Instrumental with real lyrics is `instrumental_lyrics_conflict`.

### Plan writes

`PUT .../plan` replaces the plan. The server adds `creation_mode`,
`field_origins` (`user`, `planner`, `preset`, `default`, or `unknown` for
`caption`, `lyrics`, `vocal_language`, `bpm`, `keyscale`, `timesignature`,
and `duration_sec`), `revision`, and the server-managed `brief`. A field
whose value changed becomes `user` unless the body names another origin; a
changed `query` switches to `from_idea`. An optional `expected_revision`
makes the write conditional (`409` when the plan has moved on). A null
`duration_sec` lets the planner or ACE-Step choose the length.

## Job lifecycle

Enqueuing only inserts a row. Poll `GET /api/jobs/{job_id}` until `status`
leaves `queued`/`running`:

```
queued ---> running ---> done       take_id is set
   |            |     `--> error      error is set, and a meta.json records the failure
   |            `-------> cancelled   audio is dropped; take_id and error are null
   `------------------> cancelled     the worker never claims it
```

`/v1` returns one job object for every state. `stage` is null while queued,
then `loading_model`, `planning` (only when the LM fills a plan), `rendering`,
and `writing_take`. On `done`, `artifacts` lists stable relative paths such as
`/v1/artifacts/{project}/{take}/mix.wav`. `/api` reads the same row and may
include `stage` and `cancel_requested` as extra fields.

`POST /v1/jobs/{job_id}/cancel` returns that object. A queued job becomes
`cancelled` immediately. A running job sets `cancel_requested` and stays
`running` until the worker finishes the current ACE-Step call, then discards
the audio. A job that already reached `done` stays `done`. `cancelled` is not
requeued.

A job stays `queued` while no worker is running. Check `/api/health` or
`/v1/health`.

`GET /api/jobs?project_id=...&action=train_lora&active=true` returns a project's
complete still-active worklist with no recency truncation. That is what lets
the UI rediscover an hour-long training run after a page refresh, even when
newer jobs have piled up behind it.

## Remote GPU

Opt-in, and off unless you configure it. See [remote-gpu.md](remote-gpu.md).
The shared secret is write-only: no response ever contains it.

`GET /api/remote-gpu` returns:

```json
{
  "provisioner": "runpod",
  "label": "Runpod",
  "can_provision": true,
  "stop_on_exit": true,
  "quote": { "gpu": "NVIDIA GeForce RTX 4090", "cloud": "SECURE", "hourly_usd": 0.74 },
  "connection": { "base_url": "https://abc123-8000.proxy.runpod.net" },
  "session": {
    "id": "abc123",
    "state": "starting",
    "detail": "pulling the host image (the slow part)",
    "elapsed_sec": 95.0,
    "hourly_usd": 0.27,
    "cost_estimate_usd": 0.0071
  },
  "host": { "connected": false, "ready": false, "error": "..." }
}
```

`session.state` is `starting`, `ready`, `stopping`, or `error`. It reads
`ready` only when the pod is up **and** its host answers `/health` ready.
`quote` is present only when a start is possible and nothing is running.
`host.stale_build` is true when the host runs different code than this
checkout.

## Errors

| Status | Meaning |
|---|---|
| `400` | Invalid plan field, invalid job body, a path escaping the jail, or a provider refusal. |
| `401` | `/v1` only, when auth is on and the bearer secret is missing or unknown. |
| `403` | `/v1` only. The key is known and does not include the scope. |
| `404` | Unknown project, take, LoRA, or job. |
| `409` | Idempotency key reused with a different body, or a Remote GPU is already running. |
| `413` | Request body over the upload cap. |

`/api` error responses are `{"detail": "..."}`, and the message names the
offending field.

## `/v1`

List responses are `{ "data": [...], "next_cursor": null }`. `limit` defaults
to 50 and cannot exceed 100. `cursor` is opaque; pass `next_cursor` back
unchanged.

`/v1` errors are

```json
{ "error": { "code": "not_found", "message": "project not found: ...", "param": null } }
```

`401` carries `WWW-Authenticate: Bearer`. `403` means the key is known and
lacks the scope. `read` covers health, projects, takes, jobs, artifacts, style packs, and
export. `generate` covers enqueue, cancel, and `POST /v1/songs` (which creates
the project for that song), and may also read jobs and artifacts. `write`
covers project create and update, plan edits, uploads, annotations, the active
take, and project delete. `generate` does not include `write`.

Every `/v1` POST accepts an optional `Idempotency-Key` (1–255 visible ASCII
characters, no spaces). The server stores the key, a hash of the raw body,
and the response in the jobs database for 24 hours. The same key and the same
body return that response and do not run the handler again. A replay of
`POST /v1/songs` or `POST /v1/projects/{id}/jobs` is `200` with the same job;
the first response is `202`. The same key and a different body, including a
different query string, is `409` with `error.code` `idempotency_conflict`.
Keys are scoped to the API key when auth is on. `/api` ignores the header.

## One-shot songs

`POST /v1/songs` is the agent entry. Scope `generate`. It creates a project,
writes `plan.json`, and queues `action: generate`. It does not render inline.
The response is `202`, `Location: /v1/jobs/{job_id}`, and the job object. Poll
`GET /v1/jobs/{job_id}`, then fetch the paths in `artifacts`.

```json
{
  "prompt": "a rainy night jazz ballad",
  "title": null,
  "caption": null,
  "lyrics": null,
  "instrumental": false,
  "vocal_language": "en",
  "bpm": null,
  "keyscale": null,
  "timesignature": null,
  "duration_sec": null,
  "dit_profile": "iterate",
  "seed": -1,
  "caption_rewrite": null,
  "webhook_url": null
}
```

`prompt` is required and is stored as `plan.query`. With no caption and no
lyrics, the song is simple mode: the worker's LM fills caption, lyrics, and
metas, and that filled plan is saved on the project. Caption or lyrics is
custom mode. `caption_rewrite` then defaults to false, so the caller's text is
kept; `true` lets the LM rewrite the caption. `instrumental: true` with no
lyrics stores `[Instrumental]` in custom mode, and leaves the lyrics empty in
simple mode so the LM can fill them. Null BPM, key, or duration are stored as
null. A provided `duration_sec` is stored as given. `dit_profile` is
`iterate` (the default), `polish`, or `quality`. An omitted `title` is
`Untitled`.

`webhook_url` may be null. A non-null URL is `400` until webhook delivery is
served. An idempotent replay of this POST is `200` with the same job.

With no API key configured, `/v1` and `/api` are both open on loopback. Set
`LYRE_REQUIRE_AUTH=1`, or bind a non-loopback address, and the process
requires a key from `./scripts/lyre api-key create` and serves `/v1` and
`POST /mcp` only.
There is no HTTP route that mints a key. The signed query token for players
that cannot send a header is not served yet; use the bearer header, or run
with auth off.
