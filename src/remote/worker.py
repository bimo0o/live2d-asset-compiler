from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args(argv)

    run_zip = Path(args.run_zip).resolve()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not run_zip.exists():
        raise FileNotFoundError(f"Run zip does not exist: {run_zip}")

    run_dir = _extract_run_zip(run_zip, workspace)
    _write_worker_status(run_dir, "running")
    try:
        if not args.skip_decomposition:
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
    script = run_dir / "vast" / "run_remote_decomposition.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing remote decomposition script: {script}")
    subprocess.run([sys.executable, str(script), "--run-dir", str(run_dir)], check=True)


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
