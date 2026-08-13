from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "live2d_oneclick"


class InputConfig(BaseModel):
    path: str = "input"


class OutputConfig(BaseModel):
    path: str = "output"


class OpenRouterConfig(BaseModel):
    enabled: bool = False
    api_key_env: str = "OPENROUTER_API_KEY"
    vision_model: str = "configurable"
    image_model: str = "configurable"
    require: bool = False


class ExecutionConfig(BaseModel):
    profile: str = "cloud_first"
    allow_local_fallback: bool = False
    prepare_vast_job: bool = True


class UpscaleConfig(BaseModel):
    enabled: bool = True
    scale: int = Field(default=2, ge=1, le=4)
    provider: str = "realesrgan_anime"
    remote_neural: bool = True


class DecompositionConfig(BaseModel):
    enabled: bool = True
    provider: str = "vast_qwen_layered"
    model: str = "Qwen/Qwen-Image-Layered"
    layers: int = Field(default=8, ge=3, le=24)
    remote_artifacts_path: str | None = None


class VastConfig(BaseModel):
    enabled: bool = True
    api_key_env: str = "VAST_API_KEY"
    min_vram_gb: int = 24
    disk_gb: int = 100
    docker_image: str = "ghcr.io/bimo0o/live2d-asset-compiler:latest"
    workdir: str = "/workspace/live2d_compiler"
    qwen_resolution: int = 640
    qwen_steps: int = 50
    reconstruction_model: str = "Qwen/Qwen-Image-Edit"
    reconstruction_resolution: int = 1024
    reconstruction_steps: int = 28
    max_reconstruction_tasks: int = 3


class ToggleModelConfig(BaseModel):
    enabled: bool = False
    model: str = "configurable"


class PsdConfig(BaseModel):
    enabled: bool = True


class ValidationConfig(BaseModel):
    strict: bool = True
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    openrouter: OpenRouterConfig = OpenRouterConfig()
    execution: ExecutionConfig = ExecutionConfig()
    upscale: UpscaleConfig = UpscaleConfig()
    decomposition: DecompositionConfig = DecompositionConfig()
    vast: VastConfig = VastConfig()
    segmentation: ToggleModelConfig = ToggleModelConfig()
    reconstruction: ToggleModelConfig = ToggleModelConfig()
    psd: PsdConfig = PsdConfig()
    validation: ValidationConfig = ValidationConfig()

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as file:
            return cls.model_validate(json.load(file))
