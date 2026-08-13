from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote neural upscale stage.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    source = next((run_dir / "source").glob("master_original.*"))
    output = run_dir / "upscale" / "master_neural_2x.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "remote_neural_upscale",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": "realesrgan_anime",
        "status": "pending",
        "warnings": [],
    }
    if _run_realesrgan(source, output, args.scale):
        status["status"] = "complete"
    else:
        image = Image.open(run_dir / "upscale" / f"master_{args.scale}x.png").convert("RGBA")
        image.save(output)
        status["status"] = "fallback"
        status["warnings"].append("Real-ESRGAN is not available in this image; reused prepared upscale.")
    (run_dir / "upscale" / "remote_neural_upscale.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0


def _run_realesrgan(source: Path, output: Path, scale: int) -> bool:
    python_script = Path("/opt/Real-ESRGAN/inference_realesrgan.py")
    model_path = Path("/opt/Real-ESRGAN/weights/RealESRGAN_x4plus_anime_6B.pth")
    if python_script.exists() and model_path.exists():
        output_dir = output.parent / "realesrgan_tmp"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(python_script),
            "-n",
            "RealESRGAN_x4plus_anime_6B",
            "-i",
            str(source),
            "-o",
            str(output_dir),
            "--outscale",
            str(scale),
            "--fp32",
        ]
        try:
            subprocess.run(command, cwd="/opt/Real-ESRGAN", check=True)
        except Exception:
            return False
        candidates = sorted(output_dir.glob(f"{source.stem}*"))
        if candidates:
            candidates[0].replace(output)
            return output.exists()
    executable = _which("realesrgan-ncnn-vulkan") or _which("realesrgan")
    if not executable:
        return False
    command = [executable, "-i", str(source), "-o", str(output), "-s", str(scale), "-n", "realesrgan-x4plus-anime"]
    try:
        subprocess.run(command, check=True)
    except Exception:
        return False
    return output.exists()


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


if __name__ == "__main__":
    raise SystemExit(main())
