"""Render the release listening set (SPEC.md sec 16.8, gate 4).

Six fixed briefs, two recorded seeds each, against a running server and real
worker. Durations are fixed here, before the run. Every outcome is kept,
including takes a listener later rejects, with the prompt, the constraints,
the resolved brief, and automatic audio checks; `ratings.md` is the sheet a
human fills in. This script renders and measures; it does not judge music.

    ./scripts/lyre server && ./scripts/lyre worker   # first
    .venv/bin/python scripts/listening_set.py [--base-url URL] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_stack_check import Api, CheckFailed, wav_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = (11, 23)
# Agreed before rendering: ACE-Step lands on the requested length within a
# frame or so on the pinned revision (docs/COMPLETION_BENCHMARK.md: 30.0 s
# for 30 s requested on every row), so a take more than a second off counts
# as a duration miss.
DURATION_TOLERANCE_SEC = 1.0

SUPPLIED_LYRICS = """[Verse]
Paper boats along the gutter
carry all the things I meant to say
[Chorus]
Let the rain come, let it find me
I will walk home anyway
[Verse]
Streetlights hum a song for no one
and I hum it back the whole way
[Chorus]
Let the rain come, let it find me
I will walk home anyway"""

BRIEFS: list[dict] = [
    {
        "id": "acoustic-lyric",
        "kind": "song",
        "plan": {
            "query": (
                "an intimate acoustic song about an old friend, fingerpicked guitar, soft voice"
            ),
            "instrumental": False,
            "vocal_language": "en",
            "duration_sec": 60,
        },
        "locked": ["duration_sec", "vocal_language"],
        "mode": "from_idea",
    },
    {
        "id": "rhythmic-lyric",
        "kind": "song",
        "plan": {
            "query": "an upbeat funk-pop song with a tight groove and a catchy chorus",
            "instrumental": False,
            "vocal_language": "en",
            "bpm": 112,
            "duration_sec": 60,
        },
        "locked": ["duration_sec", "vocal_language", "bpm"],
        "mode": "from_idea",
    },
    {
        "id": "supplied-lyrics",
        "kind": "song",
        "plan": {
            "query": "a wistful indie folk song in the rain",
            "lyrics": SUPPLIED_LYRICS,
            "instrumental": False,
            "vocal_language": "en",
            "duration_sec": 60,
        },
        "locked": ["duration_sec", "vocal_language", "lyrics"],
        "mode": "from_idea",
    },
    {
        "id": "exploration-ambience",
        "kind": "game cue",
        "preset": "exploration_ambience",
        "plan": {"query": "", "duration_sec": 60},
        "locked": ["duration_sec"],
        "mode": "from_brief",
    },
    {
        "id": "combat-cue",
        "kind": "game cue",
        "preset": "combat_cue",
        "plan": {"query": "", "duration_sec": 45},
        "locked": ["duration_sec"],
        "mode": "from_brief",
    },
    {
        "id": "menu-theme",
        "kind": "game cue",
        "preset": "menu_theme",
        "plan": {"query": "", "duration_sec": 45},
        "locked": ["duration_sec"],
        "mode": "from_brief",
    },
]


def _wait(api: Api, job_id: str) -> dict:
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        job = api.get(f"/v1/jobs/{job_id}")
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(2)
    raise CheckFailed(f"job {job_id} did not finish in 20 minutes")


def render(api: Api, out: Path) -> list[dict]:
    outcomes: list[dict] = []
    for brief in BRIEFS:
        project = api.post("/v1/projects", {"title": f"Listening: {brief['id']}"})
        project_id = project["id"]
        plan = dict(brief["plan"])
        origins = dict.fromkeys(brief["locked"], "user")
        if brief.get("preset"):
            api.post(
                f"/v1/projects/{project_id}/preset",
                {"preset_id": brief["preset"], "apply": True},
            )
            current = api.get(f"/v1/projects/{project_id}")["plan"]
            plan = {**current, **plan}
            origins = {**current.get("field_origins", {}), **origins}
        api.put(f"/v1/projects/{project_id}/plan", {**plan, "field_origins": origins})
        for seed in SEEDS:
            body = {"action": "generate", "seed": seed, "creation_mode": brief["mode"]}
            job = _wait(api, api.post(f"/v1/projects/{project_id}/jobs", body)["id"])
            outcome: dict = {"brief": brief["id"], "seed": seed, "job": job["id"]}
            if job["status"] != "done":
                outcome.update(status=job["status"], error=job.get("error"))
                outcomes.append(outcome)
                print(f"  {brief['id']} seed {seed}: {job['status']} {job.get('error')}")
                continue
            take = next(
                t
                for t in api.get(f"/v1/projects/{project_id}/takes")["data"]
                if t["id"] == job["take_id"]
            )
            data = api.get_bytes(f"/v1/artifacts/{project_id}/{take['id']}/mix.wav")
            name = f"{brief['id']}-seed{seed}.wav"
            (out / name).write_bytes(data)
            duration, rms, rate = wav_stats(data)
            requested = brief["plan"]["duration_sec"]
            outcome.update(
                status="done",
                file=name,
                sha256=hashlib.sha256(data).hexdigest(),
                sample_rate=rate,
                duration_sec=round(duration, 2),
                duration_ok=abs(duration - requested) <= DURATION_TOLERANCE_SEC,
                rms=round(rms, 4),
                audible=rms >= 0.001,
                caption=take.get("caption"),
                lyrics=take.get("lyrics"),
                brief_record=take["provenance"]["brief"],
                actual=take["provenance"]["actual"],
                project_id=project_id,
                take_id=take["id"],
            )
            outcomes.append(outcome)
            print(
                f"  {brief['id']} seed {seed}: {duration:.1f}s {rate} Hz rms {rms:.3f} "
                f"duration_ok={outcome['duration_ok']}"
            )
    return outcomes


def rating_sheet(outcomes: list[dict]) -> str:
    dims = (
        "prompt/style fit",
        "musical coherence",
        "clipping or glitches",
        "vocals and lyric adherence (songs) / no unwanted vocals (cues)",
        "usable start and ending",
    )
    lines = [
        "# Listening ratings",
        "",
        "Rate each dimension acceptable, minor limitation, or unusable, with short",
        "time-stamped notes. Do not use the alignment score instead of listening.",
        "Threshold (SPEC.md section 16.8): at least one take per brief usable for its",
        "purpose, and no catastrophic artifact in a kept take.",
        "",
    ]
    for outcome in outcomes:
        lines.append(f"## {outcome['brief']} (seed {outcome['seed']})")
        lines.append("")
        if outcome["status"] != "done":
            lines.append(f"Did not render: {outcome['status']} {outcome.get('error')}")
            lines.append("")
            continue
        lines.append(f"File: `{outcome['file']}`  ")
        lines.append(f"Caption: {outcome.get('caption')!s}")
        lines.append("")
        lines.extend(f"- {dim}: " for dim in dims)
        lines.append("- usable for its purpose (yes/no): ")
        lines.append("- notes: ")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8421")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "listening" / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    api = Api(args.base_url)
    health = api.get("/v1/health")
    print(f"Listening set against {args.base_url} ({health.get('gpu')})\n")
    outcomes = render(api, args.out)
    record = {
        "schema": "lyre.listening-set/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "worker": health.get("gpu"),
        "seeds": list(SEEDS),
        "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
        "briefs": BRIEFS,
        "outcomes": outcomes,
    }
    (args.out / "listening-set.json").write_text(json.dumps(record, indent=2))
    (args.out / "ratings.md").write_text(rating_sheet(outcomes))
    rendered = [o for o in outcomes if o["status"] == "done"]
    bad = [o for o in rendered if not (o["audible"] and o["duration_ok"])]
    print(f"\n  {len(rendered)}/{len(outcomes)} rendered; {len(bad)} failed the automatic checks")
    print(f"  wrote {args.out} -- a human now fills in ratings.md")
    return 1 if bad or len(rendered) != len(outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
