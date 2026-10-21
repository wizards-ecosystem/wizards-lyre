"""Write the playback MP3 next to a take's WAV archive (SPEC.md sec 7).

`mix.wav` is the archive. `mix.mp3` is the convenience copy the stable
artifact URL serves. The encoder is ffmpeg, which Lyre already requires
for style-pack training. A missing or failed ffmpeg leaves no `mix.mp3`,
so a job's artifact list — which only names files that exist — does not
advertise one.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_ENCODE_TIMEOUT_SEC = 180


def write_mix_mp3(wav_path: Path) -> bool:
    """Encode `wav_path` to a sibling `mix.mp3`.

    Returns True when `mix.mp3` is in place. Never raises: the WAV is the
    archive, and a playback-copy failure must not fail an otherwise finished
    take. A partial encode is not renamed into place.
    """
    destination = wav_path.with_name("mix.mp3")
    if not wav_path.is_file():
        print(f"mix.mp3 skipped: source WAV does not exist at '{wav_path}'")
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(
            "mix.mp3 was not written: ffmpeg is not on PATH. "
            f"'{wav_path.name}' is the archive. Install ffmpeg to produce the playback MP3."
        )
        return False

    partial = wav_path.with_name("mix.mp3.partial")
    partial.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(partial),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=_ENCODE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        partial.unlink(missing_ok=True)
        print(f"mix.mp3 was not written: ffmpeg failed ({exc}). '{wav_path.name}' is the archive.")
        return False
    if completed.returncode != 0 or not partial.is_file():
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        partial.unlink(missing_ok=True)
        suffix = f": {detail}" if detail else ""
        print(
            f"mix.mp3 was not written: ffmpeg exited {completed.returncode}{suffix}. "
            f"'{wav_path.name}' is the archive."
        )
        return False
    partial.replace(destination)
    return destination.is_file()
