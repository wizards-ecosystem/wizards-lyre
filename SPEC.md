# The Wizard's Lyre: spec

This file is the sole product spec. Implement it in phase order. Do not invent extra engines or unofficial music APIs. Do not replace the studio workspace with a one-field generate page. The agent API in section 15 is in scope; it writes the same projects the studio edits.

**Target hardware:** a 16 GB-class NVIDIA GPU (developed against an RTX 4070 Ti SUPER) on Linux/WSL2 or Windows.
**User:** one local operator. The studio is the human interface. Section 15 lets that operator's own agents generate and iterate. No accounts, and no hosted deployment of Lyre itself (an opt-in Remote GPU for rendering is section 3.1).
**Bind:** `127.0.0.1` by default, with no auth. A non-loopback bind is opt-in and requires an API key (section 15.2).

---

## 1. Vision

ACE-Step 1.5 supports **human-centered generation**. Lyre is a local generative **studio**: throw a seed, listen to takes, iterate with Cover / Repaint / Extract / Lego / Complete, export. The human rides the model. The workspace in section 9 stays a library of song **projects**, each with a **plan**, a **takes** rail, a waveform with **region select**, and actions that map 1:1 onto ACE-Step `task_type` values.

The same studio is a tool an AI agent can drive. An agent can start a song from one prompt, then keep going: read the filled plan, branch a take, repaint a region, extract a stem, and hand the project back to the human. One-shot generation is a way into that project. It persists the same plan and the same immutable takes the studio already edits. It does not replace the workspace, and it does not wrap another music service.

That is the bar for "better than a hosted prompt box": owned weights, local by default, seed-reproducible takes, region and stem edits, style packs, and one project that both the human and the agent can open. Section 15 is how any agent reaches it. The studio in section 9 is how the human keeps control.

---

## 2. Non-goals (forbidden)

- Unofficial Suno / Udio wrappers or reverse-engineered APIs
- **Lyria 3, Lyria RealTime, ElevenLabs Music, Stability Audio:** no adapters, stubs, API keys, or Gemini client.
- Magenta RealTime 2 (Apple Silicon realtime; its Windows JAX mode is an offline jam tool outside this product)
- LeVo 2 / SongGeneration (VRAM + non-commercial Tencent license)
- YuE as a second generator
- Full DAW: mixer, MIDI piano roll, VST/AU plugins, automation lanes
- Second lyric LLM besides ACE-Step's 5Hz LM (Simple mode uses that LM; Custom mode is text the caller supplies, human or agent)
- RoFormer / Demucs as a second GPU model in v1 (ACE-Step `extract` is the stem path)
- Accounts, multi-tenant hosting, or a cloud deployment of Lyre itself. The operator may opt in to API keys and a non-loopback bind (section 15.2). Docker remains only the Remote GPU host image (section 3.1).
- A one-field page in place of the studio. One-shot generation exists on the API (section 15.4) and shows up in the library as an ordinary project.
- Shipping ACE-Step's Gradio UI as Lyre. Gradio is upstream's demo. Lyre owns the product UI.

---

## 3. Local by default (locked)

By default, every generate / cover / repaint / extract / lego / complete / LoRA job runs **ACE-Step 1.5 on the local GPU**. No network is required after weights are on disk. Hugging Face download happens once at install (`uv run acestep-download` or equivalent). Runtime inference must not call Google, ElevenLabs, or any music API. Exposing the HTTP API (section 15.2) does not change the engine: an agent calls Lyre, and Lyre still renders with ACE-Step.

### 3.1 Remote GPU (opt-in, amended 2026-09-26)

The same ACE-Step 1.5 may render on a GPU the user rents or runs elsewhere, for users without a suitable card. It mirrors The Wizard's Brush's Remote GPU lane. Rules:

- **Off by default.** With no Remote GPU configured, Lyre makes no network calls and section 3 holds as written.
- **Same engine, same adapter.** The host (`worker/remote_host`) wraps `worker/acestep_worker`. Section 2 is unchanged: no other engine, music API, or model, locally or remotely.
- **Selected by the worker backend** (`LYRE_WORKER=remote`). The SQLite queue, the GPU lease, and take writing are unchanged. Projects and takes live only on the local machine; the host keeps a result only until the local worker has downloaded it.
- **The local server's default is unchanged:** `127.0.0.1`, no auth. Opt-in exposure is section 15 and does not change this lane. The remote host authenticates every request with a shared secret, reached over HTTPS only (plain HTTP only to loopback). A job from `/v1` or MCP never rents a machine; session start stays an explicit human action.
- **Provisioners:** `manual` (the default; cannot spend money) and `runpod`. Hardware is rented only from an explicit user action: never on a timer, on launch, or because a job is queued. Stop terminates. A running rental is recorded before it is reported, and reported again on the next start if a crash left it up.
- **Docker is used only for the host image** (`docker/remote-gpu/Dockerfile`). Lyre itself is never containerized.
- **Style packs (LoRA train/apply) stay local-only** until adapter weights can move to and from a host.
- **Tests stay GPU- and network-free.** HTTP goes through `server.remote_gpu.client.TRANSPORT`, which tests replace.

---

## 4. Engine: ACE-Step 1.5

Upstream: https://github.com/ace-step/ACE-Step-1.5 (MIT). Hybrid **5Hz LM planner** + **DiT renderer**. Call the Python API from our worker:

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
```

Reuse the request *schema* if useful (`GenerationParams` / upstream `api_server.py`). Do not mount their Gradio app. Do not shell out to a browser demo.

### 4.1 DiT checkpoints

| Profile | Checkpoint | Steps | CFG | When |
|---|---|---|---|---|
| `iterate` (default) | `acestep-v15-turbo` (2B) | 8 | no | daily generate, cover, repaint |
| `polish` | `acestep-v15-sft` (2B) | 50 | yes | user asks for more prompt adherence / detail |
| `quality` | `acestep-v15-xl-turbo` (4B) | 8 | no | optional; **CPU offload required on 16 GB** |
| `studio_ops` | `acestep-v15-base` (2B) | 50 | yes | **only** extract, lego, complete |

Default LM: `acestep-5Hz-lm-1.7B`. Fallback if VRAM is tight: `acestep-5Hz-lm-0.6B`. LM off (`thinking=false`) is allowed for cover / repaint / extract (upstream ignores LM for those anyway).

### 4.2 Windows backend

ACE-Step docs recommend `vllm` on 8-16 GB. **vLLM is unreliable on native Windows.** Lock:

- Default LM backend: **`pt`**
- Use `vllm` only if the worker detects a working install at startup; never fail install because vLLM is missing

### 4.3 VRAM (16 GB card)

One GPU occupant. One loaded DiT + one LM. Jobs **serialize**. Switching between `iterate` and `studio_ops` **unloads** the previous DiT before loading the next. Show “loading base model...” in the UI when swapping.

Do not load XL and base at the same time. Do not load a stem-separation net alongside ACE-Step.

Loudness normalization: ACE-Step `enable_normalization` (default on). No extra mastering chain in v1.

### 4.4 Task map

| Studio action | `task_type` | Inputs | DiT profile |
|---|---|---|---|
| Simple generate | `text2music` | natural-language query; LM fills caption, lyrics, BPM, key, duration | iterate (or polish) |
| Custom generate | `text2music` | human `plan.json` (caption + lyrics + metas) | iterate (or polish) |
| Cover / remix | `cover` | `src_audio` or `reference_audio`, caption, `audio_cover_strength` | iterate / polish |
| Rewrite region | `repaint` | `src_audio`, `repainting_start`, `repainting_end` | iterate / polish |
| Isolate track | `extract` | `src_audio`, track name | **studio_ops (base)** |
| Add / replace track | `lego` | `src_audio`, track name, caption, optional interval | **studio_ops (base)** |
| Fill arrangement | `complete` | `src_audio` (partial), track classes | **studio_ops (base)** |
| Style pack | LoRA train / load | 8+ songs, ~1 hour on 3090-class GPU | GPU exclusive, phase 4 |

Instrumental: lyrics = `[Instrumental]` or `instrumental: true`.

Caption is the highest-leverage text field (style, instruments, emotion, vocal, progression). Lyrics carry structure tags such as `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`.

---

## 5. Architecture

```
Browser, /v1 clients, and MCP clients
    -> web/     Vite + React + TypeScript (studio; loopback)
    -> server/  FastAPI + SQLite job queue  (also POST /mcp)
    -> worker/  dedicated process, ACE-Step GPU lock
    -> disk     projects/ and output/ (gitignored audio)
```

Agents are another client of the same server. They enqueue the same rows the worker already claims. The server still does not import CUDA.

Three processes conceptually:

1. **server:** HTTP, projects, jobs, file serving. No CUDA.
2. **worker:** loads ACE-Step, runs one job at a time, writes takes to disk, posts status. CUDA lives here so a GPU crash does not kill HTTP.
3. **web:** SPA. Dev: Vite proxy to FastAPI. Prod: FastAPI serves `web/dist`.

IPC: worker pulls jobs from SQLite (`queued` -> `running` -> `done` | `error`) or a localhost queue endpoint. Pick one and stick to it. SQLite is enough.

Port: **8421**. Override with `LYRE_PORT`.

---

## 6. Repo layout

```
SPEC.md                 this file
AGENTS.md               coding-agent rules
README.md               how to run
pyproject.toml
.gitignore
server/                 FastAPI app
worker/                 ACE-Step process, model swap
web/                    Vite React TS
tests/                  pytest, mocked worker, no GPU
scripts/smoke-gpu.py    optional local GPU smoke (manual)
projects/               runtime song data (gitignored)
output/                 extra exports if needed (gitignored)
checkpoints/            ACE-Step weights (gitignored)
```

Python 3.11+ is supported; the app environment excludes Python 3.14. Package manager: `uv` preferred, pip acceptable.

---

## 7. Data model

Audio and weights are gitignored. JSON is the source of truth for metadata.

```
projects/<song_id>/
  project.json
  plan.json
  takes/<take_id>/
    mix.wav              preferred archive
    mix.mp3              playback / download convenience
    meta.json
    lyrics.lrc           optional, phase 4
  stems/                 extract / lego outputs
  loras/                 phase 4
```

### 7.1 `project.json`

```json
{
  "id": "ulid-or-uuid",
  "title": "Untitled",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "dit_profile": "iterate",
  "lm_model": "acestep-5Hz-lm-1.7B",
  "active_take_id": null
}
```

`engine` is always `ace-step-1.5`. Do not add an engine field that implies other backends.

### 7.2 `plan.json`

Local composition plan shared by human and 5Hz LM. Not a third-party API client.

```json
{
  "query": "optional simple-mode seed",
  "caption": "comma tags or prose: genre, instruments, mood, vocal",
  "negative": [],
  "lyrics": "[Verse]\n...\n[Chorus]\n...",
  "instrumental": false,
  "vocal_language": "en",
  "bpm": 120,
  "keyscale": "C Major",
  "timesignature": "4/4",
  "duration_sec": 120,
  "sections": [
    { "name": "intro", "start_sec": 0, "end_sec": 8, "lyrics": "" },
    { "name": "verse", "start_sec": 8, "end_sec": 32, "lyrics": "..." }
  ]
}
```

- **Simple mode:** user types `query`. Worker sets `thinking=true`. LM fills caption, lyrics, metas. Persist the filled plan.
- **Custom mode:** user edits caption, lyrics, metas. `thinking` may still rewrite caption unless the user disables caption rewrite.
- `POST /v1/songs` (section 15.4) uses these same two modes. When the caller sends caption or lyrics, `caption_rewrite` defaults to false.
- Null / omitted BPM, key, duration: let ACE-Step CoT fill them (`use_cot_metas`).

`sections[]` is UI structure (region labels on the waveform). ACE-Step does not require it for `text2music`; it **does** require `repainting_start` / `repainting_end` for repaint.

### 7.3 `takes/<id>/meta.json`

```json
{
  "id": "ulid",
  "parent_take_id": null,
  "task_type": "text2music",
  "dit_profile": "iterate",
  "seed": 12345,
  "duration_sec": 118.4,
  "caption": "...",
  "lyrics": "...",
  "bpm": 120,
  "keyscale": "C Major",
  "created_at": "ISO-8601",
  "score": null,
  "error": null,
  "repaint": null,
  "track_name": null
}
```

Every take is immutable. Cover / repaint / extract create a **new** take with `parent_take_id` set. Never overwrite mix.wav in place.

Seeds: store the actual seed used (`-1` from the user means worker picks and records it).

---

## 8. HTTP API (server)

All JSON. Default bind `127.0.0.1`. No CORS to non-localhost origins. No auth on this surface.

This section is what the server serves today (`docs/API.md` is checked against the code). New agent-facing behavior is section 15 and is not served until its phase lands. From the phase 7 PR onward, `/api` takes bugfixes only. Do not add agent features here.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{ "ok": true, "gpu": "...", "dit_loaded": "iterate" }` |
| GET | `/api/projects` | list `{ id, title, updated_at, active_take_id }` |
| POST | `/api/projects` | create from `{ title?, query? }` |
| GET | `/api/projects/{id}` | project + plan + takes index |
| PATCH | `/api/projects/{id}` | title, dit_profile |
| PUT | `/api/projects/{id}/plan` | replace `plan.json` |
| GET | `/api/projects/{id}/takes` | take metadata list |
| GET | `/api/projects/{id}/takes/{tid}/audio` | stream mix.wav or mix.mp3 |
| POST | `/api/projects/{id}/jobs` | enqueue (body below) |
| GET | `/api/jobs/{id}` | status, error, take_id when done |
| GET | `/api/jobs` | recent jobs |

### 8.1 Job body

```json
{
  "action": "generate | cover | repaint | extract | lego | complete",
  "dit_profile": "iterate | polish | quality | studio_ops",
  "source_take_id": null,
  "upload_path": null,
  "repainting_start": 0,
  "repainting_end": -1,
  "track_name": null,
  "audio_cover_strength": 0.7,
  "seed": -1,
  "batch_size": 1
}
```

- `generate` uses current `plan.json`. Simple vs custom is whether `query` is set and caption/lyrics are empty.
- `cover` / `repaint` / `extract` / `lego` / `complete` require `source_take_id` or a server-side uploaded file under `projects/<id>/`.
- Reject `extract|lego|complete` unless `dit_profile` is `studio_ops` (server may coerce and warn).
- Reject `quality` if worker reports it cannot load XL with offload.
- `batch_size` > 1 is optional later; v1 may force 1 on 16 GB.

Write audio only under `projects/` (and `output/` if you keep a flat export folder). Tests must assert path jail: no writes outside those roots.

---

## 9. UI (web)

One SPA with a dense, local-tool aesthetic. No marketing landing page or accounts. Section 15 does not add or remove panes. A song created by `POST /v1/songs` appears in this library as an ordinary project.

### 9.1 Library

List of projects. New project. Open. Delete (confirm). Play last take inline optional.

### 9.2 Project workspace (three panes)

1. **Plan:** Simple query field; Custom: caption, lyrics (textarea with structure tags), BPM, key, time signature, duration, language, instrumental toggle, caption-rewrite checkbox. Save plan.
2. **Takes:** newest first. Seed, task, duration, score. Play, download WAV/MP3, set active, “use as source”.
3. **Waveform:** wavesurfer.js (or equivalent) on the active take. Drag a region -> Repaint. Buttons: Generate, Cover, Extract, Lego, Complete. Disable Extract/Lego/Complete unless user confirms base-model swap. Show job progress and “loading base model”.

Export: download mix + optional stems + `project.json` / `plan.json` zip. User can drop files into a real DAW.

No mixer, no MIDI, no plugin rack.

---

## 10. Worker contract

Python module that:

1. On start: detect CUDA, log VRAM, load default `iterate` DiT + 1.7B LM with `pt` backend.
2. Expose `run_job(job) -> take_meta` mapping job -> `GenerationParams` -> `generate_music(..., save_dir=take_dir)`.
3. Hold a process-wide lock. If a swap is needed, unload then load, then run.
4. Never import FastAPI. Never bind a public port (localhost queue to server is OK). The Remote GPU host (section 3.1) is a separate process that wraps this module; it is the only Lyre code that serves HTTP from a GPU machine.
5. On failure: write `meta.json` with `error`, mark job `error`, keep GPU lock released.

Mock for tests: a `worker` that writes a tiny silent WAV and valid `meta.json` without importing `acestep`.

`scripts/smoke-gpu.py`: load turbo, generate ~10 s instrumental, print path, exit 0. Not part of default pytest.

Section 15.7 adds `stage` updates and cooperative cancel to this contract. The worker still never imports FastAPI. A cancel flag is data in SQLite, not a CUDA call from the server.

---

## 11. Tests (required, no GPU)

Default `testCommand` is pytest. Mock ACE-Step. Cover at least:

- Health and bind: app listens on 127.0.0.1 (TestClient is enough)
- Create project -> plan round-trip on disk
- Enqueue generate -> mocked worker -> take appears with mix file + meta seed recorded
- Path jail: job cannot write `C:\Users\...` outside `projects/` / `output/`
- `extract` without `studio_ops` is coerced or rejected per section 8.1
- Forbidden engines: grep/static test that source does not import `google.genai`, ElevenLabs, Stability, or Suno/Udio client modules

GPU smoke is manual only.

Phases 7–13 add the tests named in their acceptance criteria. Those tests stay in default `pytest`: mocked worker, no network. The two phases that need a real GPU say so there, and they use `scripts/live_stack_check.py`, not `pytest`.

---

## 12. Implementation phases

Do not skip ahead. Do not open a “add Lyria” task. Phases 1–6 are implemented. Phases 7–13 implement section 15, one pull request each, in order. Rationale and the questions a maintainer may still overturn: [docs/adr/0001-agent-api.md](docs/adr/0001-agent-api.md).

### Phase 1: scaffold + generate

FastAPI health, SQLite jobs, React shell, mocked worker in tests, real worker that can run turbo `text2music`, play/download, take history. README: venv, `acestep-download` for turbo + 1.7B, `LYRE_PORT`.

### Phase 2: studio loop

Project library, plan editor (simple + custom), waveform + region, cover, repaint, parent_take_id chain.

### Phase 3: base swap

Worker unload/load, extract / lego / complete, explicit loading UX, coerce `studio_ops`.

### Phase 4: polish

ACE-Step quality score on takes, LRC if upstream provides timestamps, LoRA train/load (8-song path from ACE-Step Gradio training, wrapped behind Lyre's UI).

### Phase 5: studio ergonomics

Still ACE-Step only. Do not open cloud-music tasks.

- A/B compare two takes (dual play or instant swap)
- Export zip: `project.json`, `plan.json`, active mix, optional stems
- Keyboard shortcuts for generate, play/pause, next/prev take, save plan
- UI to walk `parent_take_id` (restore an earlier take as active without deleting history)

### Phase 6: library and ingest

- Search and favorite projects/takes; free-text take notes on `meta.json`
- Loudness / peak meter on the player (no extra mastering chain)
- Drag-drop a local WAV/MP3 in as a cover/repaint source (path-jailed under `projects/`)

### Phase 7: `/v1` reads and OpenAPI schemas

Typed read models for the data section 8 already stores. No new generation behavior.

- `GET /v1/health`, `GET /v1/projects`, `GET /v1/projects/{id}`, `GET /v1/projects/{id}/takes`, `GET /v1/jobs`, `GET /v1/jobs/{id}` return Pydantic models (section 15.3).
- `app.openapi()` lists each of those operations with a response schema, plus the shared `Error` schema.
- `/api` routes and the studio keep working. `docs/API.md` gains the new rows in the same PR, because the route test compares that table to the app.

**Acceptance:** pytest covers the reads against the mocked stack and asserts the OpenAPI document. **Real GPU:** no.

### Phase 8: `/v1` writes at parity with the studio loop

- `POST /v1/projects`, `PUT /v1/projects/{id}/plan`, `POST /v1/projects/{id}/jobs`, and the upload, annotation, active-take, and delete operations in section 15.3.
- Job bodies match section 8.1. One enqueue function writes the SQLite row; `/api` and `/v1` both call it.
- `Idempotency-Key` on every `/v1` POST (section 15.3).

**Acceptance:** pytest round-trips a plan, enqueues each existing action through `/v1`, and shows the same take the `/api` path would. **Real GPU:** no. The worker contract is unchanged.

### Phase 9: one-shot songs, stages, cancel

- `POST /v1/songs` (section 15.4) creates a project, writes the plan, enqueues `generate`, and returns the job.
- The worker (and the mock worker) publish `stage`. `POST /v1/jobs/{id}/cancel` is cooperative (section 15.5). Status `cancelled` is terminal and is not requeued.
- The new project opens in the existing library.

**Acceptance:** pytest, with the mock worker, shows a prompt-only song persisting a filled plan and a take; a caller-supplied caption is not rewritten unless `caption_rewrite` is true; a queued cancel never calls the worker; a cancel that arrives during `rendering` still lets the call return, then discards the take. **Real GPU:** yes. Extend `scripts/live_stack_check.py` with one `POST /v1/songs` instrumental of about 10 seconds and assert audible audio plus a persisted filled plan. That is the first new generation entry point.

### Phase 10: progress delivery and stable artifacts

- `GET /v1/jobs/{id}/events` (SSE) and terminal webhooks (section 15.5).
- Stable artifact paths, plus the short-lived signed query token for headerless players.
- Webhook tests bind `127.0.0.1` inside the test process. They do not call the public network. SSRF cases (link-local, redirect, non-loopback HTTP) are rejected in pytest.

**Acceptance:** pytest receives a signed terminal webhook from a mock-worker job, reads an SSE stream through to `done`, and fetches `mix.wav` from the stable path with a bearer token and with a signed token. **Real GPU:** no. Stages come from the mock worker.

### Phase 11: opt-in API keys and bind

- CLI mint/revoke (section 15.2). Keys stored hashed, mode `0600`.
- Default bind stays the constant `127.0.0.1` with auth off, and the studio is served.
- Non-loopback bind refuses to start when no key exists. Non-loopback and `LYRE_REQUIRE_AUTH=1` serve `/v1` and `/mcp` only. They do not serve the SPA or `/api`.
- Do not add a `0.0.0.0` literal to source. `tests/test_spec_lock.py` keeps forbidding that string. Operators who want every interface set `LYRE_BIND` themselves.

**Acceptance:** pytest covers the default (no auth, loopback), startup refusal, 401/403, and the hidden SPA. **Real GPU:** no.

### Phase 12: MCP server

- stdio entrypoint and Streamable HTTP `POST /mcp`, protocol revision 2026-07-28 (the current revision when this phase landed; no later revision was published), tasks extension as in section 15.6.
- The adapter is a thin client of the `/v1` HTTP API. It does not open the jobs database, does not import FastAPI route functions, and does not import `acestep` or `torch`. stdio reads `LYRE_URL` and an optional `LYRE_API_KEY`. `POST /mcp` calls `/v1` in-process and forwards the request bearer.
- A generation tool returns a task whose `taskId` is the job id. It returns JSON-RPC `-32021` and does not enqueue when the client did not advertise the tasks extension and did not pass `wait_timeout_sec`. Passing `wait_timeout_sec` enqueues and polls until the job finishes or the timeout elapses, then returns a complete tool result.

**Acceptance:** pytest speaks JSON-RPC in-process (stdio framing and one Streamable HTTP POST). `create_song` enqueues the same job shape phase 9 already GPU-checked. **Real GPU:** no.

### Phase 13: studio moves to `/v1`; `/api` deprecation headers

- The SPA calls `/v1`. `/api` responses gain `Deprecation`, `Sunset`, and `Link` (section 15.8). OpenAPI marks `/api` operations `deprecated: true`.
- Point `scripts/live_stack_check.py` at `/v1` for the flows it already covers. Removal of `/api` is not part of this phase.

**Acceptance:** the web test suite passes against `/v1`, and a browser pass covers the library, plan save, generate, and playback. **Real GPU:** yes. Re-run live-check on the `/v1` paths before calling this phase done.

---

## 13. Upstream references (do not re-research product scope)

- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- Inference API: `docs/en/INFERENCE.md` in that repo (`GenerationParams`, `task_type`, `generate_music`)
- Tutorial (mental model + DiT table): `docs/en/Tutorial.md`
- GPU tiers: `docs/en/GPU_COMPATIBILITY.md`; 16 GB is turbo/sft + 1.7B, XL with offload
- Install: `docs/en/INSTALL.md`; Python 3.11+ stable

If upstream parameter names drift, follow ACE-Step's current `GenerationParams` and keep Lyre's HTTP schema stable with an adapter.

---

## 14. Definition of done (phase 1)

- `pytest` green with mocked worker
- `127.0.0.1:8421` serves the SPA
- One turbo generate from Simple query writes a take you can play and download
- No Gradio, no Lyria, no second engine

---

## 15. Agent API

Specified 2026-10-20. Partially implemented.

Served now: the section 15.3 reads, the job object (every status, including `cancelled`), `POST /v1/jobs/{id}/cancel`, the stable artifact GETs, the section 15.3 writes (projects, plan, jobs, uploads, annotations, active take, delete, style packs, and export), `Idempotency-Key` on every `/v1` POST, `POST /v1/songs`, worker `stage` / `cancel_requested`, the section 15.2 API keys and bind rules, and the section 15.6 MCP server (stdio and `POST /mcp`). `docs/API.md` lists the HTTP routes. `docs/MCP.md` is the client setup. The default process still binds `127.0.0.1` with auth off and still serves the studio.

Not served yet, still one phase each in the section 12 order: SSE, webhooks, and signed artifact tokens (phase 10), and `/api` deprecation headers (phase 13). `POST /v1/songs` accepts `webhook_url: null`; a non-null webhook URL is rejected until phase 10.

Design notes, rejected alternatives, and the defaults a maintainer may overturn: [docs/adr/0001-agent-api.md](docs/adr/0001-agent-api.md). This section wins when they disagree.

### 15.1 What this API is

Lyre's generation tool for any AI client, and the same product the human uses in the studio.

- One SQLite queue, one worker, one project directory. A song an agent starts is a project the library already knows how to open.
- ACE-Step 1.5 only, on the local GPU or the Remote GPU lane in section 3.1. No second engine, locally or behind the API.
- The operator owns the machine. Keys identify that operator's agents. They are not accounts, and Lyre does not become a multi-tenant host.
- Safe defaults: loopback, no auth, studio served, nothing listening on a network interface. Turning any of that off is an explicit setting.

### 15.2 Auth and bind

**Default.** `LYRE_BIND` unset means `127.0.0.1`. Auth is off. `/`, `/api`, `/v1`, and `/mcp` are served. This is the studio install, and it stays the install `./scripts/lyre server` performs with no extra configuration.

**Loopback names.** `127.0.0.1`, `localhost`, and `::1` are loopback. Anything else, including an operator-supplied all-interfaces address, is a network bind.

**Network bind.** `LYRE_BIND` set to a non-loopback address starts only when at least one API key is stored. Otherwise the process exits with an error that names the missing key. A network bind serves `/v1` and `/mcp` only. It does not serve the SPA or `/api`. The studio on another machine is an SSH tunnel to the loopback port, which is the default auth-off server.

**Auth on loopback.** `LYRE_REQUIRE_AUTH=1` applies the same key check and the same route set as a network bind (no SPA, no `/api`). Use it when a reverse proxy or a tunnel forwards traffic onto loopback: the server cannot see the original client, so loopback is not a trust boundary in that setup. `LYRE_REQUIRE_AUTH=0` on a network bind is a startup error.

**Keys.** The operator mints them on the machine:

```bash
./scripts/lyre api-key create --name cursor --scopes read,generate,write
./scripts/lyre api-key revoke <id>
```

The command prints the secret once. The server stores `sha256` of the secret, the id, the label, the scopes, and `created_at` in a file beside the jobs database, mode `0600`. Comparison is constant-time. There is no HTTP endpoint that mints a key.

Clients send `Authorization: Bearer <secret>`. A missing or unknown secret is `401` with `WWW-Authenticate: Bearer`. A known secret without the scope is `403`. Both use the error schema in section 15.3.

| Scope | Allows |
|---|---|
| `read` | Health, list and get projects, takes, jobs, artifacts |
| `generate` | Enqueue jobs, including `POST /v1/songs` (creates the project for that song), and read those jobs and their artifacts |
| `write` | Plan edits, uploads, annotations, project delete |

`generate` does not include `write`. The default CLI invocation with no `--scopes` flag mints `read,generate,write`. Remote GPU session start is not a scope (section 15.7).

**CORS** stays off. Agents are HTTP or MCP clients. The studio is same-origin on loopback.

**Proxy.** Lyre does not ship a reverse-proxy config and does not read `X-Forwarded-For` to decide whether auth is required. The bind address and `LYRE_REQUIRE_AUTH` are the whole decision. TLS in front of a network bind is the operator's proxy, on their machine.

The source default stays the string `127.0.0.1`. Implementations read `LYRE_BIND` from the environment. They do not hardcode a public bind address. `tests/test_spec_lock.py` continues to fail the build if `0.0.0.0` appears in Lyre source.

### 15.3 `/v1` HTTP

Prefix `/v1`. JSON requests and responses. Every success and error response is a Pydantic model, and the generated OpenAPI document at `/openapi.json` contains that schema (`response_model` on each operation, shared component `Error`). `/docs` shows it. A test loads `app.openapi()` and fails if a `/v1` operation has no response schema.

Errors:

```json
{ "error": { "code": "invalid_plan", "message": "duration_sec must be a number", "param": "duration_sec" } }
```

| Status | When |
|---|---|
| `400` | Invalid body, path jail, provider refusal |
| `401` | Auth required and the bearer secret is missing or unknown |
| `403` | Known key, wrong scope |
| `404` | Unknown project, take, or job |
| `409` | Idempotency key reused with a different body; Remote GPU session already running on the loopback studio routes |
| `413` | Body over the upload cap |
| `202` | A job was queued. `Location: /v1/jobs/{id}` |

`/api` keeps today's `{"detail": "..."}` shape until it is removed. `/v1` does not copy that shape.

**Idempotency.** Every `POST` under `/v1` accepts `Idempotency-Key`, including `POST /v1/songs` and `POST /v1/projects/{id}/jobs`. The header is optional. The server stores the key, the request hash, and the response for 24 hours in the jobs database. The same key and the same raw body return the original response and do not run the handler again. A replay of a job that was queued returns that job with `200` (the first response is `202`). The same key and a different body return `409`.

**Pages.** List endpoints take `limit` (default 50, max 100) and `cursor`. The response is `{ "data": [...], "next_cursor": null }`.

**Reads (phase 7).**

| Method | Path | Response |
|---|---|---|
| GET | `/v1/health` | `ok`, `backend` (`acestep` / `mock` / `remote`), `ready`, `gpu`, `dit_loaded`, `queue` (`queued`, `running`) |
| GET | `/v1/projects` | Page of `{ id, title, updated_at, active_take_id, favorite }` |
| GET | `/v1/projects/{id}` | Project, plan, takes index |
| GET | `/v1/projects/{id}/takes` | Take metadata page |
| GET | `/v1/jobs` | Filters `project_id`, `action`, `active`, `limit`, `cursor` |
| GET | `/v1/jobs/{id}` | Job object below |

**Writes (phase 8), same behavior as section 8.**

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/projects` | `{ title?, query? }` |
| PATCH | `/v1/projects/{id}` | `title`, `dit_profile`, `favorite` |
| DELETE | `/v1/projects/{id}` | `204`. Cancels queued jobs |
| PUT | `/v1/projects/{id}/plan` | Replace `plan.json`, validated |
| POST | `/v1/projects/{id}/active_take` | Reject a failed take |
| PATCH | `/v1/projects/{id}/takes/{take_id}` | `favorite`, `notes` |
| POST | `/v1/projects/{id}/uploads` | WAV/MP3, path-jailed |
| GET | `/v1/projects/{id}/loras` | Style packs |
| GET | `/v1/projects/{id}/export` | Zip, same contents as today |
| POST | `/v1/projects/{id}/jobs` | Section 8.1 body. `202` and the job |

**Job object.**

```json
{
  "id": "ulid",
  "project_id": "ulid",
  "action": "generate",
  "dit_profile": "iterate",
  "status": "running",
  "stage": "rendering",
  "take_id": null,
  "error": null,
  "seed": null,
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "artifacts": []
}
```

`status` is `queued`, `running`, `done`, `error`, or `cancelled`. `stage` is null while queued, then `loading_model`, `planning`, `rendering`, `writing_take`. On `done`, `artifacts` lists stable relative URLs (section 15.5) and `take_id` is set. On `error`, `error` is set. On `cancelled`, `error` is null and `take_id` is null.

`/api` and `/v1` read the same row. Once `cancelled` and `stage` exist, `/api` may return them as extra fields. That is compatible. Do not fork the queue into two tables.

### 15.4 One-shot songs

`POST /v1/songs` is the agent entry. Scope `generate`. `202` and the job object. The handler creates a project, writes `plan.json`, and enqueues `action: generate`. It does not render inline.

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

Mode follows section 7.2:

- **Prompt only** (no caption, no lyrics): Simple mode. `query` is the prompt, `thinking` is on, the 5 Hz LM fills caption, lyrics, and metas, and the filled plan is persisted on the project.
- **Caption or lyrics present:** Custom mode. Those fields are the plan. `caption_rewrite` defaults to `false` so the caller's text is kept. `true` allows the LM to rewrite the caption, which is the studio checkbox's default for a human who is still drafting. An API caller who sent the text has already drafted.
- `instrumental: true` with empty lyrics stores lyrics as `[Instrumental]`.
- Null BPM, key, or duration are omitted so ACE-Step CoT can fill them. A provided `duration_sec` is stored as given.
- `dit_profile` defaults to `iterate`. `extract` / `lego` / `complete` are not one-shot actions; they go through `POST /v1/projects/{id}/jobs` with a source take.

The human then opens that project in section 9 and continues with cover, repaint, or anything else. The agent can do the same through the phase 8 job route. One-shot is the first call, not the only call.

### 15.5 Progress, cancel, artifacts

**Stages** are written by the worker to the job row (section 15.7). The server reads them. It does not poll the GPU.

**SSE.** `GET /v1/jobs/{id}/events` returns `text/event-stream`.

```
event: status
data: {"status":"running","stage":"rendering","updated_at":"..."}

event: done
data: { "...job object..." }
```

Terminal event names are `done`, `error`, and `cancelled`. The server then closes the stream. A comment line (`: ping`) goes out every 15 seconds while the stream is open. Reconnect by calling the endpoint again; the snapshot is the job row, so a missed event is recovered from `GET /v1/jobs/{id}`. Auth, when on, is the bearer header. The auth-off studio may use `EventSource`. Clients that cannot set a header use `GET /v1/jobs/{id}` polling. The signed artifact token in the next paragraph is not valid on this URL.

**Webhooks.** Optional `webhook_url` on `POST /v1/songs` and on the job POST. Delivery runs only on the transition to `done`, `error`, or `cancelled`. The body is the job object. Headers: `X-Lyre-Event` (`job.done`, `job.error`, `job.cancelled`) and `X-Lyre-Signature: sha256=<hmac-sha256 of the raw body>`. The HMAC key is a secret the server generates per webhook registration. The `202` includes it as `webhook_secret`, and so does an idempotent replay of that same POST. `GET /v1/jobs/{id}` omits it. At-least-once delivery, five attempts (1s, 5s, 30s, 2m, 10m), success is any `2xx`. No redirects.

URL rules: `https` anywhere, or `http` to loopback only. Reject link-local, multicast, and cloud-metadata addresses. Resolve DNS at send time and apply the same check to the address, so a name cannot rebind to a metadata IP between check and send.

**Cancel.** `POST /v1/jobs/{id}/cancel` returns the job object. Queued jobs become `cancelled` and the worker never claims them. A running job sets `cancel_requested`. ACE-Step cannot interrupt a diffusion call. The worker finishes that call, discards the audio, writes no take, and marks the job `cancelled`. A job that reached `done` before the flag was honored stays `done`. `cancelled` is terminal: stale-job reclaim must not requeue it.

**Artifact URLs.** Takes are immutable, so these paths are stable for the life of the take:

| Method | Path |
|---|---|
| GET | `/v1/artifacts/{project_id}/{take_id}/mix.wav` |
| GET | `/v1/artifacts/{project_id}/{take_id}/mix.mp3` |
| GET | `/v1/artifacts/{project_id}/{take_id}/lyrics.lrc` |

Paths in JSON are relative (`/v1/artifacts/...`). Clients resolve them against the origin they already use. When auth is off, the path is enough. When auth is on, the bearer header works for HTTP clients. Players that cannot set a header use `POST /v1/artifacts/{project_id}/{take_id}/sign` (scope `read` or `generate`), which returns `{ "url": "/v1/artifacts/...?access_token=...&expires=...", "expires_at": "..." }` with a 10-minute TTL, scoped to that one file. The token is a MAC over the path and the expiry. It is not the API key.

### 15.6 MCP

An MCP server exposes Lyre's tools to AI clients. Protocol revision **2026-07-28**, which was current when phase 12 was implemented. No later revision was published. A later revision, if one appears, is a spec change of its own.

References:

- Tools: https://modelcontextprotocol.io/specification/2026-07-28/server/tools
- Transports: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports
- Tasks extension (`io.modelcontextprotocol/tasks`): https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks
- Authorization (optional; this server does not implement it): https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization

**Transports.** Two bindings, one tool list.

| Binding | When |
|---|---|
| **stdio** | Local agents. `./scripts/lyre mcp` speaks newline-delimited JSON-RPC on stdin/stdout. The credential, when the HTTP server requires auth, is `LYRE_API_KEY` in the environment. stdio does not use the MCP OAuth flow. |
| **Streamable HTTP** | Remote agents, on the same port as `/v1`. A single `POST /mcp`. Every JSON-RPC message is its own POST. Responses are `application/json` or a request-scoped `text/event-stream`. |

Streamable HTTP follows the 2026-07-28 rules: no protocol session, no `Mcp-Session-Id`, no GET stream, no legacy HTTP+SSE transport from 2024-11-05. `tasks/get`, `tasks/update`, and `tasks/cancel` set the `Mcp-Name` header to the task id. An `Origin` header that is present and not the server's own origin gets `403`. A missing `Origin` (non-browser clients) is allowed.

Auth on `POST /mcp` is the same bearer key as `/v1`. The MCP OAuth authorization-server flow is not part of this spec. One key model for HTTP and for MCP.

**Discovery.** `server/discover` advertises protocol `2026-07-28`, the `tools` capability (`listChanged: false`; the list is static and returned in the order below), the `resources` capability, and extension `io.modelcontextprotocol/tasks`.

**Tools.** Each tool has an `inputSchema` and an `outputSchema`. Sync results set `resultType: "complete"` and include `structuredContent` plus the same JSON as a text content block. Generation tools return a task (below) unless the caller passes `wait_timeout_sec`, in which case the server polls and returns a complete result.

| Tool | Kind | Calls |
|---|---|---|
| `create_song` | task | `POST /v1/songs` |
| `transform_take` | task | `POST /v1/projects/{id}/jobs` for `cover`, `repaint`, `extract`, `lego`, `complete` |
| `get_job` | sync | `GET /v1/jobs/{id}` |
| `cancel_job` | sync | `POST /v1/jobs/{id}/cancel` |
| `list_projects` | sync | `GET /v1/projects` |
| `get_project` | sync | `GET /v1/projects/{id}` |
| `update_plan` | sync | `PUT /v1/projects/{id}/plan` |
| `list_takes` | sync | `GET /v1/projects/{id}/takes` |
| `get_artifact` | sync | `HEAD` of the three artifact URLs in section 15.4 |

`create_song` and `transform_take` send `Idempotency-Key`: the caller's `idempotency_key`, or a generated UUID. The key matches the `/v1` rule (`^[!-~]{1,255}$`). `transform_take` arguments are the section 8.1 fields for that action (`project_id`, `action`, `source_take_id` or `upload_path`, region, `track_name`, `audio_cover_strength`, `dit_profile`, `seed`, `batch_size`). `get_artifact` takes `project_id` and `take_id` and HEADs `mix.wav`, `mix.mp3`, and `lyrics.lrc`. It returns each file's relative `url`, `absolute_url` (resolved from `LYRE_URL`, or from `http://{bind}:{port}` for `POST /mcp`), `media_type`, `available`, `content_length`, and `http_status`. It does not download the bytes. `401` and `403` are tool errors. Any other non-200 status sets `available` to false. `delete_project`, `train_lora`, file upload, and Remote GPU session start are not tools. Those stay on the HTTP API (`train_lora` and upload) or on the loopback studio (session start, project delete).

**Tasks.** `create_song` and `transform_take` need the tasks extension unless the caller passes `wait_timeout_sec`. When that argument is omitted and the request's `_meta` capabilities include `io.modelcontextprotocol/tasks`, the server enqueues first, then returns `CreateTaskResult` (`resultType: "task"`) with `taskId` equal to the Lyre job id, `status: "working"`, `ttlMs: null`, `pollIntervalMs: 2000`, and a `statusMessage` of the current stage (or `queued`). When the client did not advertise the extension and did not pass `wait_timeout_sec`, the server returns JSON-RPC error `-32021` (Missing Required Client Capability) and does not enqueue.

`wait_timeout_sec` is the path for clients that cannot advertise tasks. The server validates it before any HTTP call: a finite number from 0 through 86400. An invalid value is a tool error and does not enqueue. A valid value enqueues, then polls `GET /v1/jobs/{id}` until a terminal status or the timeout. The result is a complete tool result, not a task. A timeout returns the current job plus a second text block and is not a tool error. A job status of `error` is a tool error. Generation still takes minutes; the default remains the task handle, and a blocking `tools/call` is only what the caller asked for by passing the timeout.

`tasks/get` is `GET /v1/jobs/{id}`. `tasks/cancel` is `POST /v1/jobs/{id}/cancel` and acknowledges immediately; the status may stay `working` until the worker honors the flag, which matches the tasks spec (cancellation is cooperative). `tasks/update` has nothing to do in this version: no tool enters `input_required`. A call for a known task returns an empty success; a call for an unknown task is a JSON-RPC error. Clients poll. `notifications/tasks` is not required in phase 12.

| Lyre job | MCP task `status` |
|---|---|
| `queued`, `running` | `working` (`statusMessage` is the stage) |
| `done` | `completed`, `result` is the `tools/call` result |
| `error` | `failed`, `error` carries the job error |
| `cancelled` | `cancelled` |

The completed tool result's `structuredContent` is the job object, including artifact paths. It also includes a `resource_link` content item with URI `lyre://take/{project_id}/{take_id}` and the take's JSON as a resource. It does not include an `audio` content block. A mix is minutes of WAV; base64 through stdio is the wrong pipe. Bytes are fetched from the artifact URL.

**Resources.** `lyre://project/{id}` and `lyre://take/{project_id}/{take_id}` return JSON text. `resources/read` of audio is not offered.

The MCP adapter is a thin client of the `/v1` HTTP API. It does not duplicate enqueue, plan, or artifact logic. It does not import FastAPI route functions, `acestep`, or `torch`, and it does not open the jobs database. stdio (`./scripts/lyre mcp`) reads `LYRE_URL` (default `http://127.0.0.1:8421`) and an optional `LYRE_API_KEY`, and calls that origin. The HTTP server must already be running. `POST /mcp` ignores `LYRE_URL` and calls `/v1` in-process, forwarding the request bearer. The worker remains the only process that claims jobs.

### 15.7 Worker and Remote GPU

Section 10 and section 3.1 stay in force. The agent API adds columns, not a second executor.

- Job rows gain `stage` (text, nullable) and `cancel_requested` (integer, default 0). Existing rows migrate with null and 0.
- The worker writes `stage` around the work it already does: before a DiT swap (`loading_model`), before the LM plan fill (`planning`), around `generate_music` (`rendering`), and while writing the take (`writing_take`). A stage that does not apply to the action is skipped. The mock worker writes the same sequence so pytest can see it.
- The worker checks `cancel_requested` before the swap, before `generate_music`, and after it returns. Set beforehand: do not call `generate_music`. Set during the call: drop the audio, write no take, status `cancelled`. The GPU lock is released either way. The remote host already has this shape (it cannot interrupt ACE-Step; it discards a cancelled render after the call). The local worker grows the same check. The host's `POST /jobs/{token}/cancel` is what the remote backend calls.
- A `/v1` or MCP enqueue does not call a provisioner. With `LYRE_WORKER=remote` and no host ready, the job stays `queued` and `GET /v1/health` reports `ready: false`. The human starts or stops a Runpod session from the studio, which is served only on the auth-off loopback server. Session start is not on `/v1` and not an MCP tool.
- Style-pack train and apply stay local-only, as section 3.1 already says. `train_lora` remains a `/v1` job action for parity and is not an MCP tool.
- Artifacts are written under `projects/` on the operator's machine. The remote host still deletes its copy after ack. Stable URLs point at the Lyre server.

### 15.8 Deprecating `/api`

`/api` is the studio contract in section 8. It stays mounted on the auth-off loopback server through phase 13.

- Phase 7 onward: bugfixes only. New behavior lands on `/v1`.
- Phase 13: the SPA calls `/v1`. Every `/api` response carries `Deprecation: <HTTP-date>` (the date `/v1` reached parity, RFC 9745), `Sunset: <HTTP-date>` (at least 90 days later, and not before two minor releases have shipped with these headers, RFC 8594), and `Link: </v1/...>; rel="successor-version"` pointing at the replacement path. OpenAPI marks those operations deprecated.
- Removing `/api` is a later decision, and not before version 1.0.0. This amendment does not schedule the removal.

### 15.9 Tests

Default `pytest` stays GPU-free and network-free, and it still must not import `acestep` or `torch` at module scope. Phases 9 and 13 are the ones that need a real GPU, and only through `./scripts/lyre live-check`. The phase 11 bind test configures the app in-process; it does not open a public socket.
