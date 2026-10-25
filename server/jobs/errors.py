"""Job-queue error types.

`server/app.py` maps JobError onto HTTP 400 and JobNotFound onto 404, so
these are part of the API contract.
"""

from __future__ import annotations


class JobError(ValueError):
    """`param` names the offending request field and `code` is the `/v1`
    error code, when the validator knows them (SPEC.md sec 16.3)."""

    def __init__(
        self, message: str = "", *, param: str | None = None, code: str = "invalid_request"
    ) -> None:
        super().__init__(message)
        self.param = param
        self.code = code


class JobNotFound(LookupError):
    pass


class JobCancelled(Exception):
    """Cooperative cancel (SPEC.md sec 15.5). Not a failure: the runner
    discards any audio and marks the job `cancelled`."""


class ProjectDeletionConflict(JobError):
    """Raised when a project already has a deletion in progress/completed."""


class SubmissionConflict(Exception):
    """A `submission_id` already names a different request in this project
    (SPEC.md sec 16.4). Mapped onto HTTP 409."""


class JobFenced(Exception):
    """This attempt no longer owns its job: the attempt token changed (the
    job was reclaimed) or the job left `running`. Nothing this attempt
    produced may be published, and it must not touch the job row."""
