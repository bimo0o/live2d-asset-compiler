from __future__ import annotations

import secrets
import zipfile
from datetime import datetime
from pathlib import Path

from src.pipeline.context import PipelineContext, load_existing_run
from src.pipeline.stages import (
    AnalysisStage,
    DecompositionStage,
    InputStage,
    ManifestStage,
    MaterialStage,
    PsdStage,
    ReportStage,
    UpscaleStage,
)
from src.schemas.config import AppConfig
from src.schemas.run import PipelineRun, RunInfo
from src.utils.logging import setup_logging
from src.validation.materials import validate_material_image
from PIL import Image


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        setup_logging()
        self.config = config
        self.stages = [
            InputStage(),
            UpscaleStage(),
            AnalysisStage(),
            DecompositionStage(),
            ManifestStage(),
            MaterialStage(),
            PsdStage(),
            ReportStage(),
        ]

    def build(self, input_override: Path | None = None, quality: str = "standard") -> PipelineRun:
        run_id = self._new_run_id()
        run_dir = Path(self.config.output.path) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run = PipelineRun(run_id=run_id, run_dir=run_dir, info=RunInfo(run_id=run_id, quality=quality))
        context = PipelineContext(self.config, run)
        context.save_run()
        try:
            for stage in self.stages:
                stage.run(context, input_override=input_override, quality=quality)
        except RuntimeError as exc:
            if "Remote artifacts are not present yet" in str(exc):
                run.info.status = "waiting_for_vast"
                context.save_run()
                return run
            raise
        run.info.status = "complete"
        run.info.completed_at = datetime.now().isoformat(timespec="seconds")
        context.save_run()
        return run

    def resume(self, run_id: str) -> PipelineRun:
        run = load_existing_run(self.config, run_id)
        context = PipelineContext(self.config, run)
        for stage in self.stages:
            if stage.name == "analysis" and context.stage_status("analysis") == "review_required":
                continue
            if not context.stage_complete(stage.name):
                stage.run(context, input_override=None, quality=run.info.quality)
        run.info.status = "complete"
        run.info.completed_at = datetime.now().isoformat(timespec="seconds")
        context.save_run()
        return run

    def prepare_vast(self, input_override: Path | None = None, quality: str = "standard") -> PipelineRun:
        run_id = self._new_run_id()
        run_dir = Path(self.config.output.path) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run = PipelineRun(run_id=run_id, run_dir=run_dir, info=RunInfo(run_id=run_id, quality=quality))
        context = PipelineContext(self.config, run)
        context.save_run()
        for stage in [InputStage(), UpscaleStage()]:
            stage.run(context, input_override=input_override, quality=quality)
        try:
            AnalysisStage().run(context, input_override=input_override, quality=quality)
        except Exception as exc:
            AnalysisStage().write_fallback(context, quality, str(exc))
        from src.cloud.vast import write_vast_job_package

        outputs = write_vast_job_package(context.run_dir, self.config)
        archive = self.archive_run(run.run_id)
        outputs.append(archive)
        context.record_stage("vast", "ready", outputs)
        run.info.status = "waiting_for_vast"
        context.save_run()
        return run

    def plan_openrouter(self, input_override: Path | None = None, quality: str = "standard") -> PipelineRun:
        run_id = self._new_run_id()
        run_dir = Path(self.config.output.path) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        run = PipelineRun(run_id=run_id, run_dir=run_dir, info=RunInfo(run_id=run_id, quality=quality))
        context = PipelineContext(self.config, run)
        context.save_run()
        for stage in [InputStage(), UpscaleStage(), AnalysisStage()]:
            stage.run(context, input_override=input_override, quality=quality)
        run.info.status = "analysis_complete"
        context.save_run()
        return run

    def archive_run(self, run_id: str) -> Path:
        run = load_existing_run(self.config, run_id)
        archive_path = run.run_dir.parent / f"{run.run_id}.zip"
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in run.run_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(run.run_dir.parent))
        return archive_path

    def validate_run(self, run_id: str) -> PipelineRun:
        run = load_existing_run(self.config, run_id)
        run_dir = run.run_dir
        source = run_dir / "upscale" / f"master_{self.config.upscale.scale}x.png"
        if not source.exists():
            raise FileNotFoundError(f"Missing upscaled source: {source}")
        expected_size = Image.open(source).size
        raw_dir = run_dir / "decomposition" / "raw"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Missing decomposition/raw in {run_dir}")
        layers = sorted(raw_dir.glob("*.png"))
        if not layers:
            raise FileNotFoundError(f"No PNG layers in {raw_dir}")
        warnings = []
        for layer in layers:
            warnings.extend(f"{layer.name}: {warning}" for warning in validate_material_image(layer, expected_size))
        context = PipelineContext(self.config, run)
        context.record_stage("remote_validation", "complete", layers, warnings=warnings)
        return run

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{secrets.token_hex(2)}"
