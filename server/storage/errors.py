"""Lookup and path-jail errors raised across the storage package.

`server/app.py` maps each of these onto an HTTP status via an exception
handler, so they are part of the API contract, not internal detail.
"""

from __future__ import annotations


class PathJailError(ValueError):
    """A resolved filesystem path escaped its allowed root."""


class ProjectNotFound(LookupError):
    pass


class TakeNotFound(LookupError):
    pass


class LoraNotFound(LookupError):
    pass
