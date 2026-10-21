"""SPEC.md sec 7: mix.mp3 is the playback copy of mix.wav.

The job artifact list only names files that exist, so a failed encode must
not leave a partial MP3 that GET would serve as audio.
"""

from __future__ import annotations

import subprocess
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
    assert playback.write_mix_mp3(wav) is False
    assert wav.is_file()
    assert not (tmp_path / "mix.mp3").exists()


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
