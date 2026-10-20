"""Job-queue error types.

`server/app.py` maps JobError onto HTTP 400 and JobNotFound onto 404, so
these are part of the API contract.
"""

from __future__ import annotations


class JobError(ValueError):
    pass


class JobNotFound(LookupError):
    pass


class JobCancelled(Exception):
    """Cooperative cancel (SPEC.md sec 15.5). Not a failure: the runner
    discards any audio and marks the job `cancelled`."""


class ProjectDeletionConflict(JobError):
    """Raised when a project already has a deletion in progress/completed."""
