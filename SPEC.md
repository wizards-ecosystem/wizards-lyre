# Wizard's Lyre — SPEC

This file is the sole product spec. Implement it in phase order. Do not invent extra engines, unofficial APIs, or a one-click generate page. Do not follow `wizards-conclave/docs/BARD_HANDOFF.md`; that document is a pointer here.

**Machine:** Windows, RTX 4070 Ti SUPER 16 GB, 32 GB RAM, i7-13700KF.
**User:** single local user. No auth. No cloud deploy.
**Bind:** `127.0.0.1` only.

---

## 1. Vision

ACE-Step 1.5 is built for **human-centered generation**, not one-click vendor mode. Lyre is a local generative **studio**: throw a seed, listen to takes, iterate with Cover / Repaint / Extract / Lego / Complete, export. The human rides the model; the model does not deliver a finished Spotify track from one prompt.

The product ACE-Step's authors reject (and the old handoff specified): one page with prompt, lyrics, duration, generate, play, download.

The product this spec locks: a library of song **projects**, each with a **plan**, a **takes** rail, a waveform with **region select**, and actions that map 1:1 onto ACE-Step `task_type` values.

---

## 2. Non-goals (forbidden)

- Unofficial Suno / Udio wrappers or reverse-engineered APIs
- **Lyria 3, Lyria RealTime, ElevenLabs Music, Stability Audio** — no adapters, no stubs, no API keys, no Gemini client. `agy` Google login is Conclave coding, not music.
- Magenta RealTime 2 (Apple Silicon realtime; Windows JAX is offline jam, not this product)
- LeVo 2 / SongGeneration (VRAM + non-commercial Tencent license)
- YuE as a second generator
- Full DAW: mixer, MIDI piano roll, VST/AU plugins, automation lanes
- Second lyric LLM besides ACE-Step's 5Hz LM (Simple mode uses that LM; Custom mode is the human)
- RoFormer / Demucs as a second GPU model in v1 (ACE-Step `extract` is the stem path)
- Auth, multi-user, reverse proxy, Docker, cloud GPU
- Shipping ACE-Step's Gradio UI as Lyre. Gradio is upstream's demo. Lyre owns the product UI.

---

## 3. Fully local (locked)

Every generate / cover / repaint / extract / lego / complete / LoRA job runs **ACE-Step 1.5 on the RTX 4070 Ti SUPER**. No network is required after weights are on disk. Hugging Face download happens once at install (`uv run acestep-download` or equivalent). Runtime inference must not call Google, ElevenLabs, or any music API.

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

ACE-Step docs recommend `vllm` on 8–16 GB. **vLLM is not a reliable native-Windows backend.** Lock:

- Default LM backend: **`pt`**
- Use `vllm` only if the worker detects a working install at startup; never fail install because vLLM is missing

### 4.3 VRAM (16 GB card)

One GPU occupant. One loaded DiT + one LM. Jobs **serialize**. Switching `iterate` ↔ `studio_ops` **unloads** the previous DiT before loading the next. Show “loading base model…” in the UI when swapping.

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
Browser (127.0.0.1)
    → web/     Vite + React + TypeScript
    → server/  FastAPI + SQLite job queue
    → worker/  dedicated process, ACE-Step GPU lock
    → disk     projects/ and output/ (gitignored audio)
```

Three processes conceptually:

1. **server** — HTTP, projects, jobs, file serving. No CUDA.
2. **worker** — loads ACE-Step, runs one job at a time, writes takes to disk, posts status. CUDA lives here so a GPU crash does not kill HTTP.
3. **web** — SPA. Dev: Vite proxy to FastAPI. Prod: FastAPI serves `web/dist`.

IPC: worker pulls jobs from SQLite (`queued` → `running` → `done` | `error`) or a localhost queue endpoint. Pick one and stick to it. SQLite is enough.

Port: **8421** (Conclave dashboard is 8420). Override with `BARD_PORT`.

---

## 6. Repo layout

```
SPEC.md                 this file
AGENTS.md               Conclave / coding-agent rules
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

Python 3.11+ (stable, not 3.14 for the app venv). Package manager: `uv` preferred, pip acceptable.

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

All JSON. Bind `127.0.0.1`. No CORS to non-localhost origins in production.

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

One SPA. Dark, dense, local-tool aesthetic — not a marketing landing page. No accounts.

### 9.1 Library

List of projects. New project. Open. Delete (confirm). Play last take inline optional.

### 9.2 Project workspace (three panes)

1. **Plan** — Simple query field; Custom: caption, lyrics (textarea with structure tags), BPM, key, time signature, duration, language, instrumental toggle, caption-rewrite checkbox. Save plan.
2. **Takes** — newest first. Seed, task, duration, score. Play, download WAV/MP3, set active, “use as source”.
3. **Waveform** — wavesurfer.js (or equivalent) on the active take. Drag a region → Repaint. Buttons: Generate, Cover, Extract, Lego, Complete. Disable Extract/Lego/Complete unless user confirms base-model swap. Show job progress and “loading base model”.

Export: download mix + optional stems + `project.json` / `plan.json` zip. User can drop files into a real DAW.

No mixer, no MIDI, no plugin rack.

---

## 10. Worker contract

Python module that:

1. On start: detect CUDA, log VRAM, load default `iterate` DiT + 1.7B LM with `pt` backend.
2. Expose `run_job(job) -> take_meta` mapping job → `GenerationParams` → `generate_music(..., save_dir=take_dir)`.
3. Hold a process-wide lock. If a swap is needed, unload then load, then run.
4. Never import FastAPI. Never bind a public port (localhost queue to server is OK).
5. On failure: write `meta.json` with `error`, mark job `error`, keep GPU lock released.

Mock for tests: a `worker` that writes a tiny silent WAV and valid `meta.json` without importing `acestep`.

`scripts/smoke-gpu.py`: load turbo, generate ~10 s instrumental, print path, exit 0. Not part of default pytest.

---

## 11. Tests (required, no GPU)

Default `testCommand` is pytest. Mock ACE-Step. Cover at least:

- Health and bind: app listens on 127.0.0.1 (TestClient is enough)
- Create project → plan round-trip on disk
- Enqueue generate → mocked worker → take appears with mix file + meta seed recorded
- Path jail: job cannot write `C:\Users\...` outside `projects/` / `output/`
- `extract` without `studio_ops` is coerced or rejected per §8.1
- Forbidden engines: grep/static test that source does not import `google.genai`, ElevenLabs, Stability, or Suno/Udio client modules

GPU smoke is manual only.

---

## 12. Implementation phases (Conclave order)

Do not skip ahead. Do not open a “add Lyria” task.

### Phase 1 — Scaffold + generate

FastAPI health, SQLite jobs, React shell, mocked worker in tests, real worker that can run turbo `text2music`, play/download, take history. README: venv, `acestep-download` for turbo + 1.7B, `BARD_PORT`.

### Phase 2 — Studio loop

Project library, plan editor (simple + custom), waveform + region, cover, repaint, parent_take_id chain.

### Phase 3 — Base swap

Worker unload/load, extract / lego / complete, explicit loading UX, coerce `studio_ops`.

### Phase 4 — Polish

ACE-Step quality score on takes, LRC if upstream provides timestamps, LoRA train/load (8-song path from ACE-Step Gradio training, wrapped — not Gradio itself).

### Phase 5 — Studio ergonomics

Still ACE-Step only. Do not open cloud-music tasks.

- A/B compare two takes (dual play or instant swap)
- Export zip: `project.json`, `plan.json`, active mix, optional stems
- Keyboard shortcuts for generate, play/pause, next/prev take, save plan
- UI to walk `parent_take_id` (restore an earlier take as active without deleting history)

### Phase 6 — Library and ingest

- Search and favorite projects/takes; free-text take notes on `meta.json`
- Loudness / peak meter on the player (no extra mastering chain)
- Drag-drop a local WAV/MP3 in as a cover/repaint source (path-jailed under `projects/`)

---

## 13. Upstream references (do not re-research product scope)

- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- Inference API: `docs/en/INFERENCE.md` in that repo (`GenerationParams`, `task_type`, `generate_music`)
- Tutorial (mental model + DiT table): `docs/en/Tutorial.md`
- GPU tiers: `docs/en/GPU_COMPATIBILITY.md` — 16 GB is turbo/sft + 1.7B; XL with offload
- Install: `docs/en/INSTALL.md` — Python 3.11+ stable

If upstream parameter names drift, follow ACE-Step's current `GenerationParams` and keep Lyre's HTTP schema stable with an adapter.

---

## 14. Definition of done (phase 1)

- `pytest` green with mocked worker
- `127.0.0.1:8421` serves the SPA
- One turbo generate from Simple query writes a take you can play and download
- No Gradio, no Lyria, no second engine
