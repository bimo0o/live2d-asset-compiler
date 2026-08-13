from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Vast.ai decomposition job from a prepared run zip.")
    parser.add_argument("--run-zip", required=True, help="Prepared run zip from app.py prepare-vast")
    parser.add_argument("--workspace", default="/workspace/live2d_jobs", help="Remote working directory")
    parser.add_argument("--keep-workdir", action="store_true", help="Do not delete extracted working directory")
    parser.add_argument("--skip-decomposition", action="store_true", help="Package an already populated run without launching Qwen")
    parser.add_argument("--preflight", action="store_true", help="Check zip, CUDA, imports, and disk without launching Qwen")
    parser.add_argument("--plan-only", action="store_true", help="Run remote neural upscale and OpenRouter planning, then package the run")
    args = parser.parse_args(argv)

    run_zip = Path(args.run_zip).resolve()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not run_zip.exists():
        raise FileNotFoundError(f"Run zip does not exist: {run_zip}")

    run_dir = _extract_run_zip(run_zip, workspace)
    _write_worker_status(run_dir, "running")
    try:
        if args.preflight:
            report = _preflight(run_dir, workspace)
            _write_worker_status(run_dir, "preflight_complete")
            print(json.dumps(report, indent=2))
            return 0
        if args.plan_only:
            _run_neural_upscale(run_dir)
            _run_local_analysis_refresh(run_dir)
            _write_worker_status(run_dir, "plan_complete")
            done_zip = _zip_run(run_dir, run_zip)
            print(f"REMOTE_PLAN_ZIP={done_zip}")
            return 0
        if not args.skip_decomposition:
            _run_neural_upscale(run_dir)
            _run_decomposition(run_dir)
        _assert_remote_outputs(run_dir)
        _write_worker_status(run_dir, "complete")
        done_zip = _zip_run(run_dir, run_zip)
        print(f"REMOTE_DONE_ZIP={done_zip}")
    except Exception as exc:
        _write_worker_status(run_dir, "failed", str(exc))
        raise
    finally:
        if not args.keep_workdir:
            shutil.rmtree(run_dir, ignore_errors=True)
    return 0


def _extract_run_zip(run_zip: Path, workspace: Path) -> Path:
    with zipfile.ZipFile(run_zip, "r") as archive:
        names = [name for name in archive.namelist() if name and not name.endswith("/")]
        if not names:
            raise ValueError(f"Run zip is empty: {run_zip}")
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1:
            raise ValueError(f"Run zip must contain one top-level run directory, got: {sorted(roots)}")
        run_id = next(iter(roots))
        target = workspace / run_id
        if target.exists():
            shutil.rmtree(target)
        archive.extractall(workspace)
    return target


def _run_decomposition(run_dir: Path) -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("src.remote.decompose"):
            subprocess.run([sys.executable, "-m", "src.remote.decompose", "--run-dir", str(run_dir)], check=True)
            return
    except Exception:
        pass
    script = run_dir / "vast" / "run_remote_decomposition.py"
    if script.exists():
        subprocess.run([sys.executable, str(script), "--run-dir", str(run_dir)], check=True)
        return
    raise FileNotFoundError(f"Missing remote decomposition runner: {script} and src.remote.decompose")


def _run_neural_upscale(run_dir: Path) -> None:
    subprocess.run([sys.executable, "-m", "src.remote.upscale", "--run-dir", str(run_dir)], check=True)


def _run_local_analysis_refresh(run_dir: Path) -> None:
    from src.pipeline.context import PipelineContext
    from src.pipeline.stages import AnalysisStage
    from src.schemas.config import AppConfig
    from src.schemas.run import PipelineRun, RunInfo

    config = AppConfig.load(Path("config.json"))
    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    run = PipelineRun(run_id=run_dir.name, run_dir=run_dir, info=info)
    context = PipelineContext(config, run)
    AnalysisStage().run(context, quality=run.info.quality)


def _preflight(run_dir: Path, workspace: Path) -> dict:
    source = run_dir / "upscale" / "master_2x.png"
    prompt = run_dir / "analysis" / "qwen_prompt.txt"
    targets = run_dir / "analysis" / "material_targets.json"
    script = run_dir / "vast" / "run_remote_decomposition.py"
    job_config = run_dir / "vast" / "vast_job.json"
    missing = [str(path) for path in (source, prompt, targets, script) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Prepared run is missing files: {missing}")
    job = json.loads(job_config.read_text(encoding="utf-8")) if job_config.exists() else {}

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "workspace": str(workspace),
        "free_disk_gb": _free_disk_gb(workspace),
        "files": {
            "source": str(source),
            "prompt": str(prompt),
            "targets": str(targets),
            "runner": str(script),
        },
        "model": job.get("model"),
        "layers": job.get("layers"),
        "qwen_resolution": job.get("qwen_resolution"),
        "qwen_steps": job.get("qwen_steps"),
        "reconstruction_model": job.get("reconstruction_model"),
        "reconstruction_resolution": job.get("reconstruction_resolution"),
        "reconstruction_steps": job.get("reconstruction_steps"),
        "max_reconstruction_tasks": job.get("max_reconstruction_tasks"),
        "torch": None,
        "cuda_available": False,
        "gpu": None,
        "qwen_import": False,
        "qwen_edit_import": False,
        "realesrgan_cli": bool(_which("realesrgan-ncnn-vulkan") or _which("realesrgan")),
        "realesrgan_python": Path("/opt/Real-ESRGAN/inference_realesrgan.py").exists(),
        "realesrgan_anime_weights": Path("/opt/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth").exists(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception as exc:
        report["torch_error"] = str(exc)
    try:
        from diffusers import QwenImageLayeredPipeline  # noqa: F401

        report["qwen_import"] = True
    except Exception as exc:
        report["qwen_import_error"] = str(exc)
    try:
        from diffusers import QwenImageEditInpaintPipeline  # noqa: F401

        report["qwen_edit_import"] = True
    except Exception as exc:
        try:
            from diffusers import DiffusionPipeline  # noqa: F401

            report["qwen_edit_import"] = "generic_diffusion_pipeline"
            report["qwen_edit_import_warning"] = str(exc)
        except Exception as fallback_exc:
            report["qwen_edit_import_error"] = f"{exc}; generic fallback failed: {fallback_exc}"
    if not report["cuda_available"]:
        raise RuntimeError(f"CUDA is not available: {report}")
    if not report["qwen_import"]:
        raise RuntimeError(f"QwenImageLayeredPipeline import failed: {report}")
    if report["free_disk_gb"] < 15:
        raise RuntimeError(f"Not enough free disk for remote job: {report}")
    return report


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / 1024**3, 2)


def _assert_remote_outputs(run_dir: Path) -> None:
    raw_dir = run_dir / "decomposition" / "raw"
    metadata = run_dir / "decomposition" / "layers.json"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing remote layer directory: {raw_dir}")
    layers = sorted(raw_dir.glob("*.png"))
    if not layers:
        raise FileNotFoundError(f"Remote run produced no layers in {raw_dir}")
    if not metadata.exists():
        raise FileNotFoundError(f"Remote run produced no metadata: {metadata}")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    if not data:
        raise ValueError("Remote metadata is empty")


def _write_worker_status(run_dir: Path, status: str, error: str | None = None) -> None:
    status_path = run_dir / "vast" / "worker_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "error": error,
    }
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _zip_run(run_dir: Path, original_zip: Path) -> Path:
    done_zip = original_zip.with_name(f"{run_dir.name}_done.zip")
    if done_zip.exists():
        done_zip.unlink()
    with zipfile.ZipFile(done_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir.parent))
    return done_zip


if __name__ == "__main__":
    raise SystemExit(main())
