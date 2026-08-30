"""End-to-end check against the live stack: HTTP server + worker + real GPU.

Unlike `scripts/smoke-gpu.py`, which calls `worker.acestep_worker.run_job`
directly, this drives the actual HTTP API against a running server and a
running worker. That is the only way to exercise what the two processes do
*together* -- the SQLite queue, the heartbeat lease, the base-model swap, plan
persistence, and the file layout on disk -- which is exactly the surface no
unit test can cover, because `pytest` deliberately never loads a GPU
(SPEC.md sec 11).

Nothing here imports torch or acestep. It is an HTTP client and a WAV reader,
so it is safe to run, lint, and type-check anywhere; only the *server* it
talks to needs the GPU.

Usage (from the repository root, with both processes already running):

    ./scripts/lyre server            # terminal 1
    ./scripts/lyre worker            # terminal 2
    ./scripts/lyre live-check        # terminal 3

    ./scripts/lyre live-check --stages generate,cover     # a subset
    ./scripts/lyre live-check --include-lora              # adds ~1 hour
    ./scripts/lyre live-check --keep                      # keep the project

Exit code is 0 only if every selected stage passed.
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A generated take must actually contain sound. A run that silently produces
# digital silence is the failure mode a "did the job say done?" check misses
# entirely, and it is exactly what a broken checkpoint or a misrouted VAE
# decode looks like from the outside.
MIN_RMS = 0.001
# worker/mock_worker.py writes a half-second of silence at 8 kHz. Real
# ACE-Step output is 44.1/48 kHz and seconds long, so a take matching this is
# the mock backend rather than a broken GPU -- worth saying explicitly.
MOCK_SAMPLE_RATE = 8000
MOCK_MAX_DURATION_SEC = 1.0
# ACE-Step does not hit a requested duration exactly; it lands within a frame
# or so of it. Wide enough not to be flaky, tight enough that a profile
# ignoring `duration` entirely still fails.
DURATION_TOLERANCE_SEC = 6.0
# Real generation is slow, and a studio_ops swap unloads and reloads a
# checkpoint first.
JOB_TIMEOUT_SEC = 15 * 60
LORA_TRAIN_TIMEOUT_SEC = 3 * 60 * 60
POLL_INTERVAL_SEC = 2.0


class CheckFailed(Exception):
    """A stage assertion failed. Carries a message meant for a human."""


@dataclass
class Stage:
    name: str
    detail: str
    ok: bool = False
    skipped: bool = False
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# HTTP


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None, raw: bool = False):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise CheckFailed(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise CheckFailed(
                f"{method} {path} -> cannot reach the server at {self.base} ({exc.reason}). "
                "Is `./scripts/lyre server` running?"
            ) from exc
        return payload if raw else (json.loads(payload) if payload else None)

    def get(self, path: str):
        return self._request("GET", path)

    def get_bytes(self, path: str) -> bytes:
        return self._request("GET", path, raw=True)

    def post(self, path: str, body: dict | None = None):
        return self._request("POST", path, body)

    def put(self, path: str, body: dict):
        return self._request("PUT", path, body)

    def patch(self, path: str, body: dict):
        return self._request("PATCH", path, body)

    def delete(self, path: str) -> None:
        self._request("DELETE", path)

    def upload(self, path: str, filename: str, content: bytes) -> dict:
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
                b"Content-Type: audio/wav\r\n\r\n",
                content,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            self.base + path,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())


# --------------------------------------------------------------------------
# Audio inspection -- stdlib only, so this script stays importable anywhere


def wav_stats(data: bytes) -> tuple[float, float, int]:
    """(duration_sec, peak-normalized RMS, sample_rate) for a PCM WAV payload."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        raw = handle.readframes(frames)

    if width != 2:
        # Only 16-bit PCM is inspected; anything else still reports duration.
        return (frames / rate if rate else 0.0, float("nan"), rate)

    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return (0.0, 0.0, rate)
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    rms = math.sqrt(total / len(samples)) / 32768.0
    return (frames / rate if rate else 0.0, rms, rate)


def silent_wav(seconds: float = 8.0, rate: int = 44100) -> bytes:
    """A valid WAV to feed the upload/ingest stage, so that stage tests the
    upload path rather than depending on a file the operator has lying about."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


# --------------------------------------------------------------------------
# Shared helpers


class Run:
    """State threaded through the stages: the client, the project under test,
    and the takes produced so far."""

    def __init__(
        self, api: Api, project_id: str, projects_dir: Path | None, allow_mock: bool = False
    ) -> None:
        self.api = api
        self.project_id = project_id
        self.projects_dir = projects_dir
        self.allow_mock = allow_mock
        self.takes: dict[str, dict] = {}  # label -> take meta

    def project_path(self) -> Path | None:
        """This project's directory, or None when the server's storage is not
        readable from here -- it may be another machine, or a different
        LYRE_PROJECTS_DIR. Stages degrade to API-only checks rather than
        failing on something that is not the server's fault."""
        if self.projects_dir is None:
            return None
        path = self.projects_dir / self.project_id
        return path if path.exists() else None

    def wait_for_job(self, job_id: str, timeout: float = JOB_TIMEOUT_SEC) -> dict:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            job = self.api.get(f"/api/jobs/{job_id}")
            if job["status"] in ("done", "error"):
                if job["status"] == "error":
                    raise CheckFailed(f"job {job['action']} failed: {job['error']}")
                return job
            status = f"{job['action']}:{job['status']}"
            if status != last:
                print(f"      {status} ...", flush=True)
                last = status
            time.sleep(POLL_INTERVAL_SEC)
        raise CheckFailed(
            f"job {job_id} still {job['status']} after {timeout / 60:.0f} min. "
            "Check the worker's terminal -- a stuck job usually means it crashed or is "
            "waiting on a model download."
        )

    def enqueue(self, body: dict) -> dict:
        job = self.api.post(f"/api/projects/{self.project_id}/jobs", body)
        finished = self.wait_for_job(
            job["id"], LORA_TRAIN_TIMEOUT_SEC if body["action"] == "train_lora" else JOB_TIMEOUT_SEC
        )
        return finished

    def take(self, take_id: str) -> dict:
        for meta in self.api.get(f"/api/projects/{self.project_id}/takes"):
            if meta["id"] == take_id:
                return meta
        raise CheckFailed(f"take {take_id} is missing from the take list after its job finished")

    def audio(self, take_id: str) -> bytes:
        return self.api.get_bytes(f"/api/projects/{self.project_id}/takes/{take_id}/audio")

    def check_audio(self, take_id: str, expected_sec: float | None, notes: list[str]) -> None:
        data = self.audio(take_id)
        duration, rms, rate = wav_stats(data)
        notes.append(f"{len(data) / 1e6:.1f} MB, {duration:.1f}s, {rate} Hz, rms {rms:.4f}")

        looks_mocked = rate == MOCK_SAMPLE_RATE and duration <= MOCK_MAX_DURATION_SEC
        if looks_mocked and not self.allow_mock:
            raise CheckFailed(
                f"this take is {duration:.1f}s of silence at {rate} Hz, which is what "
                "worker/mock_worker.py writes -- the worker is running the mocked backend. "
                "Every audio, duration, and model-swap assertion here would be meaningless. "
                "Restart the worker without LYRE_WORKER=mock, or pass --allow-mock to check "
                "this script itself."
            )
        if self.allow_mock:
            notes.append("audio assertions skipped (--allow-mock)")
            return
        if rms == 0.0:
            raise CheckFailed(
                f"take {take_id[:8]} is digital silence. The job reported success, so this is a "
                "generation or VAE-decode problem, not a queue problem."
            )
        if not math.isnan(rms) and rms < MIN_RMS:
            raise CheckFailed(f"take {take_id[:8]} is effectively silent (rms {rms:.5f})")
        if expected_sec is not None and abs(duration - expected_sec) > DURATION_TOLERANCE_SEC:
            raise CheckFailed(
                f"take {take_id[:8]} is {duration:.1f}s but {expected_sec:.0f}s was requested "
                f"(tolerance {DURATION_TOLERANCE_SEC:.0f}s)"
            )

    def loaded_profile(self) -> str | None:
        return self.api.get("/api/health").get("dit_loaded")


# --------------------------------------------------------------------------
# Stages
#
# Each returns notes for the report and raises CheckFailed with an actionable
# message on failure. They run in order and later stages reuse earlier takes,
# so `--stages` selects a prefix-consistent subset rather than arbitrary ones.


def stage_preflight(run: Run, notes: list[str]) -> None:
    """The stack is up and the worker has actually reported in.

    Whether it is the real backend or the mock is settled by the first take's
    audio, not here: `/api/health` does not name the backend once a worker has
    published a status, so the fingerprint in `Run.check_audio` is the reliable
    signal."""
    health = run.api.get("/api/health")
    notes.append(f"gpu: {health['gpu']}")
    if "not reported yet" in (health.get("gpu") or ""):
        raise CheckFailed(
            "no worker has ever checked in. Start `./scripts/lyre worker` in another terminal."
        )
    if str(health.get("gpu", "")).startswith("unavailable:"):
        raise CheckFailed(
            f"the worker is not ready: {health['gpu']}. Its terminal will say why -- usually "
            "missing weights (`./scripts/lyre models`) or no CUDA device."
        )
    notes.append(f"dit_loaded: {health.get('dit_loaded')}")


def stage_simple_generate(run: Run, notes: list[str]) -> None:
    """Simple mode: a natural-language query, and ACE-Step's 5Hz LM fills the
    plan (SPEC.md sec 7.2). The filled plan must be persisted, not just used."""
    run.api.put(
        f"/api/projects/{run.project_id}/plan",
        {"query": "slow melancholy piano, felt hammers, room tone", "duration_sec": 20},
    )
    job = run.enqueue({"action": "generate", "seed": -1})
    take = run.take(job["take_id"])
    run.takes["simple"] = take

    if take["seed"] in (-1, None):
        raise CheckFailed("seed -1 must be replaced by the seed actually used (SPEC.md sec 7.3)")
    if take["task_type"] != "text2music":
        raise CheckFailed(f"expected task_type text2music, got {take['task_type']}")

    plan = run.api.get(f"/api/projects/{run.project_id}")["plan"]
    if not (plan.get("caption") or "").strip():
        raise CheckFailed(
            "the LM did not fill (or the server did not persist) a caption for a simple-mode "
            "generate. The plan patch is what makes simple mode reusable."
        )
    notes.append(f"seed {take['seed']}, caption {plan['caption'][:60]!r}")
    if take.get("score") is not None:
        notes.append(f"quality score {take['score']}")
    if take.get("has_lrc"):
        lrc = run.api.get_bytes(f"/api/projects/{run.project_id}/takes/{take['id']}/lrc")
        notes.append(f"lrc {len(lrc)} bytes")
    run.check_audio(take["id"], 20, notes)


def stage_custom_generate(run: Run, notes: list[str]) -> None:
    """Custom mode with an explicit plan and a fixed seed. Running it twice
    with the same seed must produce the same seed in both takes -- a cheap
    check that seeding is wired through rather than ignored."""
    plan = {
        "caption": "sparse ambient drone, tape hiss, no drums",
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "bpm": 70,
        "keyscale": "A Minor",
        "timesignature": "4/4",
        "vocal_language": "en",
        "duration_sec": 20,
        "caption_rewrite": False,
        "sections": [],
        "negative": [],
        "query": "",
    }
    run.api.put(f"/api/projects/{run.project_id}/plan", plan)

    first = run.take(run.enqueue({"action": "generate", "seed": 4242})["take_id"])
    if first["seed"] != 4242:
        raise CheckFailed(f"explicit seed 4242 was not recorded (got {first['seed']})")
    if first["caption"] != plan["caption"]:
        raise CheckFailed(
            "caption_rewrite was false, but the caption changed -- the LM rewrote a caption the "
            "user locked (SPEC.md sec 7.2)"
        )
    run.takes["custom"] = first
    notes.append(f"seed {first['seed']} honored, caption preserved")
    run.check_audio(first["id"], 20, notes)


def stage_cover(run: Run, notes: list[str]) -> None:
    """Cover from an existing take. The result must be a NEW take whose
    parent_take_id points at the source -- takes are immutable (SPEC.md 7.3)."""
    source = run.takes.get("custom") or run.takes["simple"]
    take = run.take(
        run.enqueue(
            {"action": "cover", "source_take_id": source["id"], "audio_cover_strength": 0.6}
        )["take_id"]
    )
    if take["id"] == source["id"]:
        raise CheckFailed("cover overwrote its source take instead of creating a new one")
    if take["parent_take_id"] != source["id"]:
        raise CheckFailed(
            f"cover's parent_take_id is {take['parent_take_id']}, expected {source['id']}"
        )
    if take["task_type"] != "cover":
        raise CheckFailed(f"expected task_type cover, got {take['task_type']}")
    run.takes["cover"] = take
    notes.append(f"parent chain ok, seed {take['seed']}")
    run.check_audio(take["id"], None, notes)


def stage_repaint(run: Run, notes: list[str]) -> None:
    """Repaint a region. The recorded interval must match what was asked for,
    which is what the waveform drag-selection depends on."""
    source = run.takes.get("custom") or run.takes["simple"]
    take = run.take(
        run.enqueue(
            {
                "action": "repaint",
                "source_take_id": source["id"],
                "repainting_start": 4.0,
                "repainting_end": 12.0,
            }
        )["take_id"]
    )
    repaint = take.get("repaint") or {}
    if repaint.get("start") != 4.0 or repaint.get("end") != 12.0:
        raise CheckFailed(f"repaint interval was not recorded as asked: {repaint}")
    if take["parent_take_id"] != source["id"]:
        raise CheckFailed("repaint did not chain to its source take")
    run.takes["repaint"] = take
    notes.append(f"interval {repaint['start']}-{repaint['end']}s recorded")
    run.check_audio(take["id"], None, notes)


def stage_upload(run: Run, notes: list[str]) -> None:
    """Drag-drop ingest (SPEC.md sec 12 Phase 6): a local WAV becomes a
    cover source, path-jailed under the project."""
    result = run.api.upload(
        f"/api/projects/{run.project_id}/uploads", "ingest.wav", silent_wav(8.0)
    )
    upload_path = result["upload_path"]
    if not upload_path.startswith("uploads/"):
        raise CheckFailed(f"upload landed outside uploads/: {upload_path}")
    notes.append(f"stored at {upload_path}")
    on_disk = run.project_path()
    if on_disk is None:
        notes.append("skipped the on-disk check: --projects-dir not readable from here")
    elif not (on_disk / upload_path).exists():
        raise CheckFailed(f"upload was accepted but {on_disk / upload_path} does not exist")

    take = run.take(
        run.enqueue({"action": "cover", "upload_path": upload_path, "audio_cover_strength": 0.7})[
            "take_id"
        ]
    )
    if take["parent_take_id"] is not None:
        raise CheckFailed("a cover from an uploaded file has no parent take, but one was recorded")
    run.takes["upload_cover"] = take
    notes.append("cover from upload ok")
    run.check_audio(take["id"], None, notes)


def stage_studio_ops(run: Run, notes: list[str]) -> None:
    """extract / lego / complete on the base checkpoint.

    This is the stage no mocked test can stand in for: it forces the worker to
    unload the current DiT and load another (SPEC.md sec 4.3, one GPU
    occupant), which is where VRAM pressure and load ordering actually bite.
    The check watches `dit_loaded` flip to studio_ops and back.
    """
    source = run.takes.get("custom") or run.takes["simple"]
    before = run.loaded_profile()
    notes.append(f"loaded before swap: {before}")

    extract = run.take(
        run.enqueue(
            {
                "action": "extract",
                "source_take_id": source["id"],
                "dit_profile": "studio_ops",
                "track_name": "vocals",
            }
        )["take_id"]
    )
    after = run.loaded_profile()
    if after != "studio_ops":
        raise CheckFailed(
            f"after an extract job the worker reports dit_loaded={after!r}, expected "
            "'studio_ops'. The base-model swap did not happen, so the UI's 'loading base "
            "model' state is reporting something untrue."
        )
    if extract.get("track_name") != "vocals":
        raise CheckFailed(f"extract did not record track_name (got {extract.get('track_name')})")
    run.takes["extract"] = extract
    notes.append(f"swap to studio_ops confirmed; extract take {extract['id'][:8]}")
    run.check_audio(extract["id"], None, notes)

    lego = run.take(
        run.enqueue(
            {
                "action": "lego",
                "source_take_id": source["id"],
                "dit_profile": "studio_ops",
                "track_name": "drums",
            }
        )["take_id"]
    )
    run.takes["lego"] = lego
    run.check_audio(lego["id"], None, notes)

    complete = run.take(
        run.enqueue(
            {
                "action": "complete",
                "source_take_id": source["id"],
                "dit_profile": "studio_ops",
                "track_name": "strings",
            }
        )["take_id"]
    )
    run.takes["complete"] = complete
    run.check_audio(complete["id"], None, notes)
    notes.append("extract / lego / complete all produced audio")

    # And the swap must be reversible: an ordinary generate has to bring the
    # iterate checkpoint back, or every subsequent job runs on the wrong model.
    run.enqueue({"action": "generate", "dit_profile": "iterate", "seed": -1})
    back = run.loaded_profile()
    if back != "iterate":
        raise CheckFailed(
            f"after swapping back the worker reports dit_loaded={back!r}, expected 'iterate'. "
            "The swap is one-way, so studio_ops would silently serve every later job."
        )
    notes.append("swapped back to iterate")


def stage_studio_ops_guard(run: Run, notes: list[str]) -> None:
    """The server must refuse extract/lego/complete on a non-studio_ops profile
    rather than quietly running them on the wrong checkpoint (SPEC.md sec 8.1)."""
    source = run.takes.get("custom") or run.takes["simple"]
    try:
        run.api.post(
            f"/api/projects/{run.project_id}/jobs",
            {
                "action": "extract",
                "source_take_id": source["id"],
                "dit_profile": "iterate",
                "track_name": "vocals",
            },
        )
    except CheckFailed as exc:
        if "studio_ops" not in str(exc):
            raise CheckFailed(
                f"extract was rejected, but not for the expected reason: {exc}"
            ) from exc
        notes.append("extract on 'iterate' correctly rejected")
        return
    raise CheckFailed("extract was accepted on the 'iterate' profile; it must require studio_ops")


def stage_annotations_and_active(run: Run, notes: list[str]) -> None:
    """Favorites, notes, and the active take -- the metadata the library and
    export depend on, exercised against the real files on disk."""
    take = run.takes.get("custom") or run.takes["simple"]
    run.api.patch(
        f"/api/projects/{run.project_id}/takes/{take['id']}",
        {"favorite": True, "notes": "live-stack check"},
    )
    run.api.post(f"/api/projects/{run.project_id}/active_take", {"take_id": take["id"]})

    detail = run.api.get(f"/api/projects/{run.project_id}")
    stored = next(t for t in detail["takes"] if t["id"] == take["id"])
    if not stored["favorite"] or stored["notes"] != "live-stack check":
        raise CheckFailed("take annotations did not round-trip through meta.json")
    if detail["project"]["active_take_id"] != take["id"]:
        raise CheckFailed("active_take_id was not persisted")

    project_path = run.project_path()
    if project_path is None:
        notes.append("favorite/notes/active_take round-tripped through the API")
        notes.append("skipped the meta.json check: --projects-dir not readable from here")
        return
    meta_path = project_path / "takes" / take["id"] / "meta.json"
    if not meta_path.exists():
        raise CheckFailed(f"{meta_path} is missing; the API and the disk layout disagree")
    on_disk = json.loads(meta_path.read_text())
    if on_disk.get("notes") != "live-stack check":
        raise CheckFailed("the API reported the note but meta.json on disk does not carry it")
    notes.append("favorite/notes/active_take round-tripped to meta.json")


def stage_export(run: Run, notes: list[str]) -> None:
    """The export zip is what a user drops into a DAW, so its contents are a
    contract (SPEC.md sec 9.2)."""
    payload = run.api.get_bytes(f"/api/projects/{run.project_id}/export?include_stems=true")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        bad = archive.testzip()
        if bad is not None:
            raise CheckFailed(f"the export archive is corrupt at member {bad}")
        for required in ("project.json", "plan.json"):
            if required not in names:
                raise CheckFailed(f"export is missing {required}; got {names}")
        if not any(n.startswith("mix.") for n in names):
            raise CheckFailed(f"export has no active mix; got {names}")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise CheckFailed(f"export member escapes the archive root: {name!r}")
        mix = next(n for n in names if n.startswith("mix."))
        if mix.endswith(".wav") and not run.allow_mock:
            _duration, rms, _rate = wav_stats(archive.read(mix))
            if rms == 0.0:
                raise CheckFailed("the exported mix is silent")
    notes.append(f"{len(payload) / 1e6:.1f} MB, {len(names)} members: {', '.join(sorted(names))}")


def stage_path_jail(run: Run, notes: list[str]) -> None:
    """The jail is enforced against the running server, not just in unit tests
    (SPEC.md sec 11)."""
    for bad_path in ("../../../etc/passwd", "/etc/passwd", "uploads/../../escape.wav"):
        try:
            run.api.post(
                f"/api/projects/{run.project_id}/jobs",
                {"action": "cover", "upload_path": bad_path},
            )
        except CheckFailed:
            continue
        raise CheckFailed(f"upload_path {bad_path!r} was accepted; the path jail is not holding")
    notes.append("traversal attempts rejected")


def stage_lora(run: Run, notes: list[str]) -> None:
    """Train a style pack from this project's takes and generate with it.

    Opt-in: SPEC.md sec 4.4 sizes training at roughly an hour on a 3090-class
    GPU, and it holds the GPU exclusively for that whole time.
    """
    takes = run.api.get(f"/api/projects/{run.project_id}/takes")
    playable = [t["id"] for t in takes if not t.get("error")]
    if len(playable) < 8:
        raise CheckFailed(
            f"style-pack training needs 8 distinct takes and this project has {len(playable)}. "
            "Run the earlier stages first, or generate more takes."
        )
    started = time.monotonic()
    job = run.enqueue(
        {"action": "train_lora", "name": "live-stack pack", "source_take_ids": playable[:8]}
    )
    lora_id = job.get("lora_id")
    if not lora_id:
        raise CheckFailed("training finished without recording a lora_id")
    notes.append(f"trained in {(time.monotonic() - started) / 60:.1f} min")

    packs = run.api.get(f"/api/projects/{run.project_id}/loras")
    pack = next((p for p in packs if p["id"] == lora_id), None)
    if pack is None:
        raise CheckFailed(f"lora {lora_id} is not in the project's pack list")
    if pack.get("error"):
        raise CheckFailed(f"the trained pack records an error: {pack['error']}")

    take = run.take(run.enqueue({"action": "generate", "lora_id": lora_id, "seed": -1})["take_id"])
    if take.get("lora_id") != lora_id:
        raise CheckFailed("the generated take does not record the style pack that was applied")
    run.check_audio(take["id"], None, notes)
    notes.append(f"generated with pack {lora_id[:8]} applied")


# --------------------------------------------------------------------------
# Driver

STAGES: list[tuple[str, str, object]] = [
    ("preflight", "server up, worker reported in, GPU visible", stage_preflight),
    ("generate", "simple mode: LM fills and persists the plan", stage_simple_generate),
    (
        "custom",
        "custom mode: explicit plan, fixed seed, caption not rewritten",
        stage_custom_generate,
    ),
    ("cover", "cover creates a new take chained to its source", stage_cover),
    ("repaint", "repaint records the requested region", stage_repaint),
    ("upload", "drag-drop ingest becomes a cover source", stage_upload),
    ("studio-ops", "extract / lego / complete swap the base checkpoint and back", stage_studio_ops),
    ("guard", "extract is refused on a non-studio_ops profile", stage_studio_ops_guard),
    ("annotations", "favorite / notes / active take reach meta.json", stage_annotations_and_active),
    ("export", "export zip contents and member safety", stage_export),
    ("jail", "path traversal is rejected by the running server", stage_path_jail),
    ("lora", "train a style pack and generate with it (~1 hour)", stage_lora),
]
DEFAULT_STAGES = [name for name, _, _ in STAGES if name != "lora"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end check against a running Lyre server, worker, and GPU.",
        epilog="Start `./scripts/lyre server` and `./scripts/lyre worker` first.",
    )
    parser.add_argument(
        "--base-url",
        default=f"http://127.0.0.1:{os.environ.get('LYRE_PORT', '8421')}",
        help="the running server (default: 127.0.0.1 on $LYRE_PORT, else 8421)",
    )
    parser.add_argument(
        "--projects-dir",
        default=os.environ.get("LYRE_PROJECTS_DIR") or str(REPO_ROOT / "projects"),
        help=(
            "where the server stores projects, for the on-disk checks. Those are skipped "
            "with a note when it is not readable from here, e.g. a server on another machine."
        ),
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help=(
            "run even when the worker is the mocked backend. Only useful for checking this "
            "script itself: the mock writes silent half-second WAVs, so every audio "
            "assertion is meaningless against it."
        ),
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help="comma-separated stage names (default: everything except lora)",
    )
    parser.add_argument(
        "--include-lora",
        action="store_true",
        help="also train a style pack and generate with it; adds roughly an hour",
    )
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the project afterwards so its takes can be listened to",
    )
    args = parser.parse_args()

    if args.list:
        width = max(len(name) for name, _, _ in STAGES)
        for name, detail, _ in STAGES:
            default = "" if name in DEFAULT_STAGES else "   (opt-in)"
            print(f"  {name.ljust(width)}  {detail}{default}")
        return 0

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.include_lora and "lora" not in selected:
        selected.append("lora")
    unknown = [s for s in selected if s not in {name for name, _, _ in STAGES}]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    projects_dir = Path(args.projects_dir)
    api = Api(args.base_url)
    print(f"Live stack check against {args.base_url}")
    print(f"Stages: {', '.join(selected)}\n")

    try:
        project = api.post(
            "/api/projects", {"title": f"Live stack check {time.strftime('%Y-%m-%d %H:%M')}"}
        )
    except CheckFailed as exc:
        print(f"FAILED before starting: {exc}", file=sys.stderr)
        return 2
    run = Run(
        api,
        project["id"],
        projects_dir if projects_dir.is_dir() else None,
        allow_mock=args.allow_mock,
    )
    print(f"Project {project['id']}\n")

    results: list[Stage] = []
    failed = False
    for name, detail, fn in STAGES:
        stage = Stage(name=name, detail=detail)
        if name not in selected:
            stage.skipped = True
            results.append(stage)
            continue
        if failed:
            # Later stages reuse earlier takes; running them after a failure
            # produces confusing cascades rather than more information.
            stage.skipped = True
            stage.notes.append("skipped after an earlier failure")
            results.append(stage)
            continue

        print(f"[{name}] {detail}")
        started = time.monotonic()
        try:
            fn(run, stage.notes)  # type: ignore[operator]
            stage.ok = True
        except CheckFailed as exc:
            stage.error = str(exc)
            failed = True
        except Exception as exc:
            # Report cleanly: this is a hands-on-the-machine script, and a raw
            # traceback buries which stage failed.
            stage.error = f"unexpected {type(exc).__name__}: {exc}"
            failed = True
        stage.seconds = time.monotonic() - started
        for note in stage.notes:
            print(f"      {note}")
        print(f"      {'PASS' if stage.ok else 'FAIL'} in {stage.seconds:.1f}s")
        if stage.error:
            print(f"      {stage.error}")
        print()
        results.append(stage)

    print("=" * 72)
    width = max(len(s.name) for s in results)
    for stage in results:
        mark = "skip" if stage.skipped else ("PASS" if stage.ok else "FAIL")
        timing = "" if stage.skipped else f"  {stage.seconds:6.1f}s"
        print(f"  {mark:>4}  {stage.name.ljust(width)}{timing}")
    total = sum(s.seconds for s in results)
    ran = [s for s in results if not s.skipped]
    print(f"\n  {sum(1 for s in ran if s.ok)}/{len(ran)} stages passed in {total / 60:.1f} min")

    if args.keep or failed:
        location = run.project_path() or f"project {run.project_id} (see ./scripts/lyre paths)"
        print(f"\n  Project kept for inspection: {location}")
        if failed:
            print("  (kept because a stage failed -- its takes are on disk to listen to)")
    else:
        api.delete(f"/api/projects/{run.project_id}")
        print("\n  Project deleted. Pass --keep to listen to the takes.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
