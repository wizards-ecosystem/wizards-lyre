"""Manual GPU smoke -- not part of pytest (SPEC.md sec 10, sec 11).

Loads ACE-Step turbo, generates ~10s of instrumental text2music, prints the
output path, and exits 0. Run by hand on the real GPU box, e.g.:

    python scripts/smoke-gpu.py

Requires ACE-Step 1.5 installed and weights downloaded (SPEC.md sec 4, 13).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import storage  # noqa: E402
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
    # SPEC.md sec 8.1/11: generated audio must land under projects/ or
    # output/, never a bare OS temp dir -- jailed_output_path enforces (and
    # raises PathJailError on) any escape from output/.
    take_id = f"smoke-{uuid.uuid4().hex}"
    take_dir = storage.jailed_output_path("smoke-gpu", take_id)
    job = {
        "action": "generate",
        "dit_profile": "iterate",
        "seed": -1,
    }
    try:
        meta, _plan_patch = run_job(job=job, plan=INSTRUMENTAL_PLAN, take_id=take_id, take_dir=take_dir)
    except WorkerUnavailable as exc:
        print(f"GPU smoke unavailable: {exc}", file=sys.stderr)
        return 2

    audio_path = take_dir / "mix.wav"
    print(f"seed={meta['seed']} duration_sec={meta['duration_sec']}")
    print(audio_path)
    return 0 if audio_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
