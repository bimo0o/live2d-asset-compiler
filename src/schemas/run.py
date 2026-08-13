from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class StageRecord(BaseModel):
    stage: str
    status: str
    timestamp: str
    inputs_hash: str | None = None
    outputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunInfo(BaseModel):
    run_id: str
    status: str = "running"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    completed_at: str | None = None
    quality: str = "standard"
    stages: list[StageRecord] = Field(default_factory=list)


class PipelineRun(BaseModel):
    run_id: str
    run_dir: Path
    info: RunInfo

    model_config = {"arbitrary_types_allowed": True}

