"""Pydantic models for `/v1`. Every success and error body is one of these,
so `app.openapi()` carries a response schema for each operation
(SPEC.md sec 15.3).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ProjectCreate(BaseModel):
    """`POST /v1/projects`. Omitted fields match `POST /api/projects`."""

    title: str | None = None
    query: str | None = None


class ProjectPatch(BaseModel):
    """`PATCH /v1/projects/{id}`. Only fields the client sends are applied."""

    title: str | None = None
    dit_profile: str | None = None
    favorite: bool | None = None


class PlanWrite(BaseModel):
    """`PUT /v1/projects/{id}/plan`. A partial object is filled by
    `validate_plan`; unknown keys are dropped, same as `/api`."""

    model_config = ConfigDict(extra="ignore")

    query: str | None = None
    caption: str | None = None
    negative: list[str] | None = None
    lyrics: str | None = None
    instrumental: bool | None = None
    vocal_language: str | None = None
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str | None = None
    duration_sec: float | None = None
    sections: list[PlanSection] | None = None
    caption_rewrite: bool | None = None


class ActiveTake(BaseModel):
    take_id: str


class TakePatch(BaseModel):
    favorite: bool | None = None
    notes: str | None = None


class JobCreate(BaseModel):
    """Section 8.1 job body. `enqueue_job` is what writes the row."""

    action: str
    dit_profile: str | None = None
    source_take_id: str | None = None
    source_take_ids: list[str] | None = None
    upload_path: str | None = None
    repainting_start: float = 0
    repainting_end: float = -1
    track_name: str | None = None
    name: str | None = None
    lora_id: str | None = None
    audio_cover_strength: float = Field(0.7, ge=0.0, le=1.0)
    seed: int = -1
    batch_size: int = 1


class UploadCreated(BaseModel):
    upload_path: str


class SongCreate(BaseModel):
    """`POST /v1/songs` (SPEC.md sec 15.4).

    Prompt only (empty caption and lyrics) is simple mode: the worker's LM
    fills the plan. Caption or lyrics is custom mode, and `caption_rewrite`
    then defaults to false. `webhook_url` is accepted as null; delivery is a
    later phase, so a non-null URL is rejected rather than dropped.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    title: str | None = None
    caption: str | None = None
    lyrics: str | None = None
    instrumental: bool = False
    vocal_language: str = Field(default="en", min_length=1)
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str | None = None
    duration_sec: float | None = None
    dit_profile: Literal["iterate", "polish", "quality"] = "iterate"
    seed: int = -1
    caption_rewrite: bool | None = None
    webhook_url: str | None = None

    @field_validator("prompt", "vocal_language", mode="before")
    @classmethod
    def _strip_required(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("title", "caption", "lyrics", "keyscale", "timesignature", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("duration_sec")
    @classmethod
    def _positive_duration(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value <= 0:
            raise ValueError("duration_sec must be a positive finite number")
        return value

    @field_validator("webhook_url")
    @classmethod
    def _webhook_not_yet(cls, value: str | None) -> str | None:
        if value is not None:
            raise ValueError("webhook delivery is not available yet")
        return value


class Lora(BaseModel):
    """A style pack's meta.json. Training writes more fields than a failure."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    created_at: str | None = None
    source_take_count: int | None = None
    base_checkpoint: str | None = None
    dit_profile: str | None = None
    status: str | None = None
    error: str | None = None
