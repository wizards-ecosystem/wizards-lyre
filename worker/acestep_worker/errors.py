"""The one error type this worker raises for an unusable ACE-Step stack."""

from __future__ import annotations


class WorkerUnavailable(RuntimeError):
    """ACE-Step is not installed, has no usable GPU, or its Python API no
    longer matches this adapter."""
