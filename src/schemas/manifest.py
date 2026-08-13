from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Canvas(BaseModel):
    width: int
    height: int


class ReconstructionStatus(BaseModel):
    required: bool = False
    status: Literal["not_required", "pending", "complete", "failed"] = "not_required"


class ValidationStatus(BaseModel):
    coverage: float | None = None
    edge_quality: float | None = None
    confidence: float = 0.75
    warnings: list[str] = Field(default_factory=list)


class ManifestPart(BaseModel):
    id: str
    name: str
    group: str
    parent: str | None = None
    source_layer: str
    mask: str
    material: str
    bbox: list[int]
    z_index: int
    deformable: bool = True
    physics_candidate: bool = False
    physics_type: str | None = None
    reconstruction: ReconstructionStatus = ReconstructionStatus()
    validation: ValidationStatus = ValidationStatus()


class Manifest(BaseModel):
    version: str = "0.1"
    canvas: Canvas
    source: str
    quality: str = "standard"
    parts: list[ManifestPart] = Field(default_factory=list)
    recommended_parameters: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

