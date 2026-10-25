"""Facts about produced files and the code that produced them (SPEC.md sec 16.4).

Take provenance and the export manifest both describe audio the same way:
format, sample rate, channels, frames, duration, size, and sha256, read from
the file itself rather than from what a backend claimed. Values the file
cannot tell us (an MP3's sample rate, without a decoder) are None, meaning
unknown, never guessed.
"""

from __future__ import annotations

import hashlib
import subprocess
import wave
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_CHUNK = 1024 * 1024
MEDIA_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".lrc": "text/plain"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    """Size, hash, and (for WAV) the audio format read from the header.
    Raises OSError when the file is missing or unreadable; a zero-byte file
    is reported as such so callers can refuse it."""
    size = path.stat().st_size
    info: dict[str, Any] = {
        "file": path.name,
        "format": path.suffix.lstrip(".").lower(),
        "media_type": MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        "bytes": size,
        "sha256": sha256_file(path),
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "duration_sec": None,
    }
    if path.suffix.lower() == ".wav" and size > 0:
        try:
            with wave.open(str(path), "rb") as wav:
                rate = wav.getframerate()
                frames = wav.getnframes()
                info["sample_rate"] = rate
                info["channels"] = wav.getnchannels()
                info["frames"] = frames
                info["duration_sec"] = round(frames / rate, 3) if rate else None
        except (wave.Error, EOFError):
            # 32-bit float WAV (WAVE_FORMAT_IEEE_FLOAT) is valid but not
            # readable by the stdlib; leave the header fields unknown.
            pass
    return info


@lru_cache(maxsize=1)
def app_version() -> str:
    try:
        return metadata.version("wizards-lyre")
    except metadata.PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def app_revision() -> str | None:
    """The checkout's git commit, or None outside a git checkout (a release
    bundle, the Remote GPU image)."""
    if not (_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


@lru_cache(maxsize=1)
def ace_step_revision() -> str | None:
    """The pinned upstream revision this checkout installs (ACE_STEP_REVISION)."""
    path = _ROOT / "ACE_STEP_REVISION"
    try:
        revision = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return revision or None
