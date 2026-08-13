from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.schemas.config import AppConfig
from src.schemas.run import PipelineRun, RunInfo, StageRecord


class PipelineContext:
    def __init__(self, config: AppConfig, run: PipelineRun) -> None:
        self.config = config
        self.run = run
        self.run_dir = run.run_dir

    def dir(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def run_json_path(self) -> Path:
        return self.run_dir / "run.json"

    def record_stage(
        self,
        stage: str,
        status: str,
        outputs: list[Path] | None = None,
        inputs_hash: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        record = StageRecord(
            stage=stage,
            status=status,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            inputs_hash=inputs_hash,
            outputs=[str(path.relative_to(self.run_dir)) for path in outputs or []],
            warnings=warnings or [],
        )
        self.run.info.stages = [item for item in self.run.info.stages if item.stage != stage]
        self.run.info.stages.append(record)
        self.save_run()
        stage_path = self.run_dir / stage / "stage.json"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def stage_complete(self, stage: str) -> bool:
        return any(item.stage == stage and item.status == "complete" for item in self.run.info.stages)

    def stage_status(self, stage: str) -> str | None:
        for item in reversed(self.run.info.stages):
            if item.stage == stage:
                return item.status
        return None

    def save_run(self) -> None:
        self.run_json_path.write_text(self.run.info.model_dump_json(indent=2), encoding="utf-8")


def load_existing_run(config: AppConfig, run_id: str) -> PipelineRun:
    run_dir = Path(config.output.path) / run_id
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Run does not exist: {run_id}")
    info = RunInfo.model_validate(json.loads(run_json.read_text(encoding="utf-8")))
    return PipelineRun(run_id=run_id, run_dir=run_dir, info=info)
