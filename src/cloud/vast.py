from __future__ import annotations

import json
from pathlib import Path

from src.schemas.config import AppConfig


def write_vast_job_package(run_dir: Path, config: AppConfig) -> list[Path]:
    package_dir = run_dir / "vast"
    package_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _write_readme(package_dir, config),
        _write_requirements(package_dir),
        _write_remote_runner(package_dir),
        _write_job_config(package_dir, config),
        _write_cli_notes(package_dir, config),
    ]
    return files


def _write_readme(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "README_VAST.md"
    path.write_text(
        f"""# Vast.ai Decomposition Job

This run is cloud-first. Do not run Qwen Image Layered on the local machine.

Use the Docker image built from this GitHub repository:

```text
{config.vast.docker_image}
```

Upload `output/{package_dir.parent.name}.zip` to the Vast.ai instance, then run:

```bash
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --preflight
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --plan-only
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip
```

Expected output contract:

```text
decomposition/raw/layer_00_<semantic>.png
decomposition/raw/layer_01_<semantic>.png
...
decomposition/layers.json
decomposition/preview.png
```

The worker writes:

```text
/workspace/{package_dir.parent.name}_done.zip
```

Download that `*_done.zip`, extract it over `output/{package_dir.parent.name}/`,
then resume locally:

```powershell
python app.py resume {package_dir.parent.name}
```

The generated remote runner uses the Qwen Image Layered pipeline contract.
Keep the output paths unchanged so the local compiler can continue after the
run directory is downloaded back.

If the image is private, make the GHCR package public or run Docker login on
the Vast instance before starting the container.
""",
        encoding="utf-8",
    )
    return path


def _write_requirements(package_dir: Path) -> Path:
    path = package_dir / "requirements-vast.txt"
    root_requirements = Path("requirements-vast.txt")
    if root_requirements.exists():
        path.write_text(root_requirements.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text("pillow>=10.0\nnumpy>=2.0\n", encoding="utf-8")
    return path


def _write_job_config(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "vast_job.json"
    path.write_text(
        json.dumps(
            {
                "docker_image": config.vast.docker_image,
                "min_vram_gb": config.vast.min_vram_gb,
                "disk_gb": config.vast.disk_gb,
                "workdir": config.vast.workdir,
                "model": config.decomposition.model,
                "layers": config.decomposition.layers,
                "qwen_resolution": config.vast.qwen_resolution,
                "qwen_steps": config.vast.qwen_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_cli_notes(package_dir: Path, config: AppConfig) -> Path:
    path = package_dir / "vast_cli_notes.md"
    path.write_text(
        f"""# Vast.ai CLI Notes

Install and authenticate:

```bash
pip install vastai
export VAST_API_KEY="..."
vastai show user
```

Search for a suitable instance:

```bash
vastai search offers 'gpu_ram>={config.vast.min_vram_gb} disk_space>={config.vast.disk_gb} verified=true rentable=true'
```

Create one from an offer id:

```bash
vastai create instance <offer_id> --image {config.vast.docker_image} --disk {config.vast.disk_gb} --ssh --direct --label live2d-qwen-layered
```

Upload the prepared zip, then run:

```bash
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --preflight
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip --plan-only
python -m src.remote.worker --run-zip /workspace/{package_dir.parent.name}.zip
```

If the GHCR package is private, create a GitHub personal access token with
package read permission and configure Docker login on the Vast instance.

The compiler does not store secrets in config files. Keep `OPENROUTER_API_KEY`
and `VAST_API_KEY` in environment variables or the provider's secret store.
""",
        encoding="utf-8",
    )
    return path


def _write_remote_runner(package_dir: Path) -> Path:
    path = package_dir / "run_remote_decomposition.py"
    path.write_text(
        '''from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Layered")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    source = run_dir / "upscale" / "master_neural_2x.png"
    if not source.exists():
        source = run_dir / "upscale" / "master_2x.png"
    if not source.exists():
        raise FileNotFoundError(f"Missing upscaled source: {source}")

    raw_dir = run_dir / "decomposition" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    analysis = _load_analysis(run_dir)
    targets = _load_targets(run_dir)
    prompt = _load_prompt(run_dir) or _build_prompt(analysis, targets)
    source_image = Image.open(source).convert("RGBA")
    model_image = _fit_for_qwen(source_image, args.resolution)

    pipeline = QwenImageLayeredPipeline.from_pretrained(args.model)
    pipeline = pipeline.to("cuda", torch.bfloat16)
    pipeline.set_progress_bar_config(disable=None)

    inputs = {
        "image": model_image,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": "redesign, different character, changed face, changed colors, changed clothing",
        "num_inference_steps": args.steps,
        "num_images_per_prompt": 1,
        "layers": args.layers,
        "resolution": args.resolution,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    output_layers = output.images[0]
    generated = []
    analysis_parts = analysis.get("parts", []) if isinstance(analysis, dict) else []
    semantic_parts = targets.get("parts", []) or analysis_parts
    for index, layer in enumerate(output_layers):
        full_canvas = _fit_to_canvas(layer.convert("RGBA"), source_image.size)
        if not _has_visible_pixels(full_canvas):
            continue
        generated.append({"index": index, "image": full_canvas, "bbox": _alpha_bbox(full_canvas)})

    assignments = _assign_layers_to_targets(generated, semantic_parts)
    metadata = []
    for output_index, item in enumerate(assignments):
        part = item["part"]
        full_canvas = item["image"]
        source_index = item["source_index"]
        part_id = _sanitize(part.get("id") or f"qwen_layer_{source_index:02d}")
        path = raw_dir / f"layer_{output_index:02d}_{part_id}.png"
        full_canvas.save(path)
        metadata.append(
            {
                "id": part_id,
                "name": _title(part_id),
                "group": part.get("group", "ROOT"),
                "z_index": int(part.get("depth", 10 + output_index * 10)),
                "layer": str(path.relative_to(run_dir)),
                "mask": "",
                "confidence": float(part.get("confidence", 0.72)),
                "warnings": [f"generated on Vast.ai with Qwen Image Layered; source_layer={source_index}"],
            }
        )

    (run_dir / "decomposition" / "layers.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_preview(run_dir, metadata)
    (run_dir / "decomposition" / "vast_result.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "steps": args.steps,
                "resolution": args.resolution,
                "layers": len(metadata),
                "prompt": prompt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def _load_analysis(run_dir: Path) -> dict:
    path = run_dir / "analysis" / "character_analysis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_targets(run_dir: Path) -> dict:
    path = run_dir / "analysis" / "material_targets.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prompt(run_dir: Path) -> str:
    path = run_dir / "analysis" / "qwen_prompt.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _build_prompt(analysis: dict, targets: dict) -> str:
    analysis_parts = analysis.get("parts", []) if isinstance(analysis, dict) else []
    parts = targets.get("parts") or analysis_parts
    part_names = ", ".join(part.get("id", "") for part in parts[:24] if part.get("id"))
    return (
        "Decompose this anime character illustration into transparent RGBA layers for Live2D material separation. "
        "Preserve source identity, line art, colors, geometry, and visible pixels. "
        "Prefer coherent semantic materials, not arbitrary color fragments. "
        f"Expected visible parts: {part_names}."
    )


def _fit_for_qwen(image: Image.Image, resolution: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= resolution:
        return image
    scale = resolution / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _fit_to_canvas(layer: Image.Image, size: tuple[int, int]) -> Image.Image:
    if layer.size == size:
        return layer
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    resized = layer.resize(size, Image.Resampling.LANCZOS)
    canvas.alpha_composite(resized)
    return canvas


def _has_visible_pixels(layer: Image.Image) -> bool:
    return layer.getchannel("A").getbbox() is not None


def _alpha_bbox(layer: Image.Image) -> list[int]:
    bbox = layer.getchannel("A").getbbox()
    if not bbox:
        return [0, 0, 0, 0]
    left, top, right, bottom = bbox
    return [left, top, right - left, bottom - top]


def _assign_layers_to_targets(generated: list[dict], targets: list[dict]) -> list[dict]:
    remaining = list(generated)
    assignments = []
    for target in sorted(targets, key=lambda row: int(row.get("depth", 50))):
        if not remaining:
            break
        target_box = target.get("bbox") or [0, 0, 0, 0]
        best = max(remaining, key=lambda item: _bbox_score(item["bbox"], target_box))
        remaining.remove(best)
        assignments.append({"part": target, "image": best["image"], "source_index": best["index"]})
    for item in remaining:
        assignments.append(
            {
                "part": {"id": f"qwen_layer_{item['index']:02d}", "group": "ROOT", "depth": 200 + item["index"], "confidence": 0.65},
                "image": item["image"],
                "source_index": item["index"],
            }
        )
    return assignments


def _bbox_score(layer_box: list[int], target_box: list[int]) -> float:
    intersection = _intersection_area(layer_box, target_box)
    layer_area = max(1, layer_box[2] * layer_box[3])
    target_area = max(1, target_box[2] * target_box[3])
    layer_cx = layer_box[0] + layer_box[2] / 2.0
    layer_cy = layer_box[1] + layer_box[3] / 2.0
    target_cx = target_box[0] + target_box[2] / 2.0
    target_cy = target_box[1] + target_box[3] / 2.0
    distance = abs(layer_cx - target_cx) + abs(layer_cy - target_cy)
    return intersection / target_area + intersection / layer_area - distance / 100000.0


def _intersection_area(a: list[int], b: list[int]) -> int:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _write_preview(run_dir: Path, metadata: list[dict]) -> None:
    source = Image.open(run_dir / "upscale" / "master_2x.png").convert("RGBA")
    composite = Image.new("RGBA", source.size, (0, 0, 0, 0))
    for item in sorted(metadata, key=lambda row: row["z_index"]):
        composite.alpha_composite(Image.open(run_dir / item["layer"]).convert("RGBA"))
    preview = run_dir / "decomposition" / "preview.png"
    composite.save(preview)

    thumb_w = 220
    rows = max(1, len(metadata))
    sheet = Image.new("RGB", (thumb_w * 2, rows * 250), "white")
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(metadata):
        image = Image.open(run_dir / item["layer"]).convert("RGBA")
        thumb = Image.new("RGBA", image.size, (255, 255, 255, 255))
        thumb.alpha_composite(image)
        thumb.thumbnail((thumb_w, 220), Image.Resampling.LANCZOS)
        y = row * 250
        sheet.paste(thumb.convert("RGB"), ((thumb_w - thumb.width) // 2, y + 24))
        draw.text((thumb_w + 8, y + 32), f"{item['id']}\\n{item['group']}\\nz={item['z_index']}", fill=(20, 20, 20))
    sheet.save(run_dir / "decomposition" / "preview_sheet.png")


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned or "material"


def _title(value: str) -> str:
    return " ".join(piece.capitalize() for piece in value.split("_") if piece)


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    return path
