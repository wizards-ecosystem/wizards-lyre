"""SPEC.md sec 7: mix.mp3 is the playback copy of mix.wav.

The job artifact list only names files that exist, so a failed encode must
not leave a partial MP3 that GET would serve as audio.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import pytest

from worker import playback


def test_missing_wav_does_not_invent_an_mp3(tmp_path: Path) -> None:
    assert playback.write_mix_mp3(tmp_path / "mix.wav") is False
    assert not (tmp_path / "mix.mp3").exists()


def test_missing_ffmpeg_leaves_the_wav_as_the_only_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(playback.shutil, "which", lambda _name: None)
    monkeypatch.setattr(playback, "_ffmpeg_from_imageio", lambda: None)
    assert playback.write_mix_mp3(wav) is False
    assert wav.is_file()
    assert not (tmp_path / "mix.mp3").exists()


def test_encode_command_forces_the_mp3_muxer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.partial` suffix is not `.mp3`. Without `-f mp3` ffmpeg exits 234."""
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")
    seen: list[list[str]] = []

    def encode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(command)
        Path(command[-1]).write_bytes(b"ID3-audio")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(playback.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(playback.subprocess, "run", encode)
    assert playback.write_mix_mp3(wav) is True
    command = seen[0]
    input_at = command.index("-i")
    format_at = command.index("-f")
    assert format_at > input_at
    assert command[format_at + 1] == "mp3"
    assert command[format_at + 2] == command[-1]
    assert command[-1].endswith("mix.mp3.partial")


def test_imageio_ffmpeg_is_used_when_nothing_is_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")
    bundled = tmp_path / "imageio-ffmpeg"
    bundled.write_bytes(b"")
    seen: list[str] = []

    def encode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(command[0])
        Path(command[-1]).write_bytes(b"ID3-audio")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(playback.shutil, "which", lambda _name: None)
    monkeypatch.setattr(playback, "_ffmpeg_from_imageio", lambda: str(bundled))
    monkeypatch.setattr(playback.subprocess, "run", encode)
    assert playback.write_mix_mp3(wav) is True
    assert seen == [str(bundled)]


def test_path_ffmpeg_wins_over_imageio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")
    seen: list[str] = []

    def encode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(command[0])
        Path(command[-1]).write_bytes(b"ID3-audio")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(playback.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(playback, "_ffmpeg_from_imageio", lambda: "/should-not-run")
    monkeypatch.setattr(playback.subprocess, "run", encode)
    assert playback.write_mix_mp3(wav) is True
    assert seen == ["/usr/bin/ffmpeg"]


def test_real_ffmpeg_encodes_a_wav_when_one_is_available(tmp_path: Path) -> None:
    if playback._resolve_ffmpeg() is None:
        pytest.skip("no ffmpeg on PATH and imageio-ffmpeg is not importable")

    wav = tmp_path / "mix.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)
    assert playback.write_mix_mp3(wav) is True
    encoded = (tmp_path / "mix.mp3").read_bytes()
    assert encoded[:3] == b"ID3" or encoded[:1] == b"\xff"
    assert not (tmp_path / "mix.mp3.partial").exists()


def test_imageio_resolver_returns_the_bundled_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "ffmpeg"
    exe.write_bytes(b"bin")

    class _Imageio:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return str(exe)

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Imageio())
    assert playback._ffmpeg_from_imageio() == str(exe)


def test_imageio_resolver_skips_a_missing_or_failing_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Missing:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return str(tmp_path / "nope")

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Missing())
    assert playback._ffmpeg_from_imageio() is None

    class _Broken:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            raise RuntimeError("binary not unpacked")

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", _Broken())
    assert playback._ffmpeg_from_imageio() is None

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", object())
    assert playback._ffmpeg_from_imageio() is None

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    assert playback._ffmpeg_from_imageio() is None


def test_successful_encode_replaces_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")

    def encode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        Path(command[-1]).write_bytes(b"ID3-audio")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(playback.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(playback.subprocess, "run", encode)
    assert playback.write_mix_mp3(wav) is True
    assert (tmp_path / "mix.mp3").read_bytes() == b"ID3-audio"
    assert not (tmp_path / "mix.mp3.partial").exists()


def test_failed_encode_removes_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")

    def encode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(command, 1, b"", b"Encoder not found")

    monkeypatch.setattr(playback.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(playback.subprocess, "run", encode)
    assert playback.write_mix_mp3(wav) is False
    assert not (tmp_path / "mix.mp3").exists()
    assert not (tmp_path / "mix.mp3.partial").exists()


def test_ffmpeg_timeout_or_oserror_removes_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(playback.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def explode(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(playback.subprocess, "run", explode)
    assert playback.write_mix_mp3(wav) is False
    assert not (tmp_path / "mix.mp3.partial").exists()

    def missing(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise OSError("ffmpeg vanished")

    monkeypatch.setattr(playback.subprocess, "run", missing)
    assert playback.write_mix_mp3(wav) is False
    assert not (tmp_path / "mix.mp3").exists()
