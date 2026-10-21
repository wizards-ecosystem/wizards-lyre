"""Pydantic models for `/v1`. Every success and error body is one of these,
so `app.openapi()` carries a response schema for each operation
(SPEC.md sec 15.3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
JobStage = Literal["loading_model", "planning", "rendering", "writing_take"]


class ErrorObject(BaseModel):
    code: str
    message: str
    param: str | None = None


class Error(BaseModel):
    """Shared error component. `/v1` does not use `/api`'s `{"detail": ...}`."""

    error: ErrorObject


class QueueCounts(BaseModel):
    queued: int
    running: int


class Health(BaseModel):
    ok: bool
    backend: str
    ready: bool
    gpu: str
    dit_loaded: str | None
    queue: QueueCounts


class ProjectSummary(BaseModel):
    id: str
    title: str
    updated_at: str | None = None
    active_take_id: str | None = None
    favorite: bool = False


class ProjectPage(BaseModel):
    data: list[ProjectSummary]
    next_cursor: str | None


class Project(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None
    dit_profile: str | None = None
    lm_model: str | None = None
    active_take_id: str | None = None
    favorite: bool = False


class PlanSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    lyrics: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = ""
    caption: str = ""
    negative: list[str] = Field(default_factory=list)
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "en"
    bpm: float | None = None
    keyscale: str | None = None
    timesignature: str = "4/4"
    duration_sec: float | None = None
    sections: list[PlanSection] = Field(default_factory=list)
    caption_rewrite: bool = False


class Take(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    parent_take_id: str | None = None
    task_type: str | None = None
    dit_profile: str | None = None
    seed: int | None = None
    duration_sec: float | None = None
    caption: str | None = None
    lyrics: str | None = None
    bpm: float | None = None
    keyscale: str | None = None
    created_at: str | None = None
    score: float | None = None
    has_lrc: bool | None = None
    error: str | None = None
    track_name: str | None = None
    lora_id: str | None = None
    favorite: bool = False
    notes: str = ""


class TakePage(BaseModel):
    data: list[Take]
    next_cursor: str | None


class ProjectDetail(BaseModel):
    project: Project
    plan: Plan
    takes: list[Take]


class Job(BaseModel):
    """One object for every state (SPEC.md sec 15.3). `stage` is null while
    queued. `artifacts` is empty until `done`."""

    id: str
    project_id: str
    action: str
    dit_profile: str
    status: JobStatus
    stage: JobStage | None = None
    take_id: str | None = None
    error: str | None = None
    seed: int | None = None
    created_at: str
    updated_at: str
    artifacts: list[str]


class JobPage(BaseModel):
    data: list[Job]
    next_cursor: str | None
