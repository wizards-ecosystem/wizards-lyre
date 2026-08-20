"""Manual GPU smoke -- not part of pytest (SPEC.md sec 10, sec 11).

Loads ACE-Step turbo, generates ~10s of instrumental text2music, prints the
output path, and exits 0. Run by hand on the real GPU box, e.g.:

    python scripts/smoke-gpu.py

Requires ACE-Step 1.5 installed and weights downloaded (SPEC.md sec 4, 13).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.acestep_worker import WorkerUnavailable, run_job  # noqa: E402

INSTRUMENTAL_PLAN = {
    "caption": "warm analog synth pad, gentle percussion, instrumental",
    "lyrics": "[Instrumental]",
    "instrumental": True,
    "bpm": 100,
    "keyscale": "C Major",
    "duration_sec": 10,
}


def main() -> int:
    take_dir = Path(tempfile.mkdtemp(prefix="bard-smoke-"))
    job = {
        "action": "generate",
        "dit_profile": "iterate",
        "seed": -1,
    }
    try:
        meta, _plan_patch = run_job(job=job, plan=INSTRUMENTAL_PLAN, take_id="smoke", take_dir=take_dir)
    except WorkerUnavailable as exc:
        print(f"GPU smoke unavailable: {exc}", file=sys.stderr)
        return 2

    audio_path = take_dir / "mix.wav"
    print(f"seed={meta['seed']} duration_sec={meta['duration_sec']}")
    print(audio_path)
    return 0 if audio_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
