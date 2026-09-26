# HTTP API

All JSON, all under `http://127.0.0.1:8421` by default. No authentication;
see [SECURITY.md](../SECURITY.md). The optional Remote GPU host has its own,
authenticated protocol; see [remote-gpu.md](remote-gpu.md#the-host-protocol).

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
| `GET` | `/api/remote-gpu` | Remote GPU state: provisioner, price quote, connection, rental, host health. |
| `PUT` | `/api/remote-gpu/connection` | Connect a host you run: `{ base_url, secret }`. |
| `DELETE` | `/api/remote-gpu/connection` | Forget the connected host. |
| `POST` | `/api/remote-gpu/session` | Rent a GPU with the configured provisioner. **Billable.** |
| `DELETE` | `/api/remote-gpu/session` | Terminate the rental. Safe when nothing is running. |

`/` serves the built SPA from `web/dist` when it exists, and otherwise returns
a hint telling you to build it.

## Enqueuing a job

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
  "batch_size": 1
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

## Job lifecycle

Enqueuing only inserts a row. Poll `GET /api/jobs/{job_id}` until `status`
leaves `queued`/`running`:

```
queued ---> running ---> done      take_id is set
                    `--> error     error is set, and a meta.json records the failure
```

A job stays `queued` while no worker is running. Check `/api/health`.

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
| `404` | Unknown project, take, LoRA, or job. |
| `409` | A Remote GPU is already running; stop it first. |
| `413` | Request body over the upload cap. |

Error responses are `{"detail": "..."}`, and the message names the offending
field.
