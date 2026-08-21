"""Manual GPU smoke -- not part of pytest (SPEC.md sec 10, sec 11).

Loads ACE-Step turbo, generates ~10s of instrumental text2music, prints the
output path, and exits 0. Run by hand on the real GPU box, e.g.:

    python scripts/smoke-gpu.py

Requires a CUDA GPU, and ACE-Step 1.5 installed with turbo weights downloaded
(SPEC.md sec 4, 13). torch and acestep are only ever imported lazily, inside
functions -- importing this module (or running it without either installed)
must not crash with a raw traceback; `main()` reports a clean local-setup
message on stderr and exits nonzero instead. No web UI toolkit, no cloud
music API: generation goes straight through worker.acestep_worker's local
ACE-Step adapter, same as production.
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


def _cuda_status() -> str:
    """Lazy, non-fatal CUDA availability check (mirrors
    worker.acestep_worker._log_cuda_status) -- printed up front so a run
    always shows whether CUDA is actually visible before ACE-Step tries to
    use it, without needing torch importable just to import this script."""
    try:
        import torch
    except ImportError:
        return "torch is not installed in this environment"
    if not torch.cuda.is_available():
        return "CUDA is not available (no GPU attached, or a CPU-only torch build)"
    return f"CUDA available: {torch.cuda.get_device_name(0)}"


def main() -> int:
    print(f"CUDA check: {_cuda_status()}")

    # SPEC.md sec 8.1/11: generated audio must land under projects/ or
    # output/, never a bare OS temp dir -- jailed_output_path enforces (and
    # raises PathJailError on) any escape from output/. Lands under
    # output/smoke/<take_id>/.
    take_id = f"smoke-{uuid.uuid4().hex}"
    take_dir = storage.jailed_output_path("smoke", take_id)
    job = {
        "action": "generate",
        "dit_profile": "iterate",
        "seed": -1,
    }
    try:
        meta, _plan_patch, _lrc_text = run_job(
            job=job, plan=INSTRUMENTAL_PLAN, take_id=take_id, take_dir=take_dir
        )
    except WorkerUnavailable as exc:
        print(f"GPU smoke unavailable: {exc}", file=sys.stderr)
        print(
            "Install ACE-Step 1.5 and download the turbo checkpoint, and make sure a "
            "CUDA GPU is attached in this environment (SPEC.md sec 4, 13).",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - a CUDA/driver error or other
        # local setup failure can surface directly from torch/acestep here,
        # not wrapped as WorkerUnavailable (e.g. no GPU attached at all) --
        # report it as a clean local-setup message instead of a raw
        # traceback (this is a manual, hands-on-the-machine script).
        print(f"GPU smoke failed: {exc}", file=sys.stderr)
        print(
            "Check that a CUDA GPU is attached and ACE-Step 1.5 (with the turbo "
            "checkpoint downloaded) is installed in this environment (SPEC.md sec 4, 13).",
            file=sys.stderr,
        )
        return 2

    audio_path = take_dir / "mix.wav"
    print(f"seed={meta['seed']} duration_sec={meta['duration_sec']}")
    print(audio_path)
    return 0 if audio_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
