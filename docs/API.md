# HTTP API

All JSON, all under `http://127.0.0.1:8421` by default. No authentication —
see [SECURITY.md](../SECURITY.md).

The server is FastAPI, so the authoritative, always-current reference is the
generated schema while it is running:

- **<http://127.0.0.1:8421/docs>** — interactive Swagger UI
- **<http://127.0.0.1:8421/openapi.json>** — raw OpenAPI schema

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

- `cover`, `repaint`, `extract`, `lego`, and `complete` need a real source —
  either `source_take_id` or an `upload_path` from the uploads endpoint.
- `extract`, `lego`, and `complete` are forced to the `studio_ops` profile.
- `lora_id` only applies to `generate`, `cover`, and `repaint`, and the pack
  must exist and have finished training.
- `train_lora` needs a non-empty `name` and at least 8 distinct source takes.
- `seed: -1` means the worker picks one and records the actual value.
- `batch_size` is forced to 1.
- `audio_cover_strength` must be within `0.0`–`1.0` (which also rejects NaN
  and infinities).

## Job lifecycle

Enqueuing only inserts a row. Poll `GET /api/jobs/{job_id}` until `status`
leaves `queued`/`running`:

```
queued ──▶ running ──▶ done      take_id is set
                   └─▶ error     error is set, and a meta.json records the failure
```

A job stays `queued` forever if no worker is running — check `/api/health`.

`GET /api/jobs?project_id=…&action=train_lora&active=true` returns a project's
complete still-active worklist with no recency truncation. That is what lets
the UI rediscover an hour-long training run after a page refresh, even when
newer jobs have piled up behind it.

## Errors

| Status | Meaning |
|---|---|
| `400` | Invalid plan field, invalid job body, or a path escaping the jail. |
| `404` | Unknown project, take, LoRA, or job. |
| `413` | Request body over the upload cap. |

Error responses are `{"detail": "..."}`, and the message names the offending
field.
