from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen Layered + Live2D hidden-area reconstruction on Vast.ai.")
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--reconstruction-model", default=None)
    parser.add_argument("--reconstruction-steps", type=int, default=None)
    parser.add_argument("--reconstruction-resolution", type=int, default=None)
    parser.add_argument("--max-reconstruction-tasks", type=int, default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    job = _load_job_config(run_dir)
    layers = args.layers if args.layers is not None else int(job.get("layers", 8))
    model = args.model or str(job.get("model") or "Qwen/Qwen-Image-Layered")
    steps = args.steps if args.steps is not None else int(job.get("qwen_steps", 50))
    resolution = args.resolution if args.resolution is not None else int(job.get("qwen_resolution", 640))
    reconstruction_model = args.reconstruction_model or str(job.get("reconstruction_model") or "Qwen/Qwen-Image-Edit")
    reconstruction_steps = (
        args.reconstruction_steps if args.reconstruction_steps is not None else int(job.get("reconstruction_steps", 28))
    )
    reconstruction_resolution = (
        args.reconstruction_resolution
        if args.reconstruction_resolution is not None
        else int(job.get("reconstruction_resolution", 1024))
    )
    max_reconstruction_tasks = (
        args.max_reconstruction_tasks
        if args.max_reconstruction_tasks is not None
        else int(job.get("max_reconstruction_tasks", 3))
    )

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
    model_image = _fit_for_qwen(source_image, resolution)

    pipeline = QwenImageLayeredPipeline.from_pretrained(model)
    pipeline = pipeline.to("cuda", torch.bfloat16)
    pipeline.set_progress_bar_config(disable=None)

    inputs = {
        "image": model_image,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": "redesign, different character, changed face, changed colors, changed clothing, "
        "flat opaque background, merged layers",
        "num_inference_steps": steps,
        "num_images_per_prompt": 1,
        "layers": layers,
        "resolution": resolution,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)

    output_layers = output.images[0]
    del pipeline
    _clear_cuda_cache()

    generated = []
    analysis_parts = analysis.get("parts", []) if isinstance(analysis, dict) else []
    semantic_parts = targets.get("parts", []) or analysis_parts
    for index, layer in enumerate(output_layers):
        full_canvas = _fit_to_canvas(layer.convert("RGBA"), source_image.size)
        if not _has_visible_pixels(full_canvas):
            continue
        generated.append({"index": index, "image": full_canvas, "bbox": _alpha_bbox(full_canvas)})

    assignments = _assign_layers_to_targets(generated, semantic_parts)
    reconstruction_queue = _select_reconstruction_queue(assignments, max_reconstruction_tasks, args.skip_reconstruction)
    reconstruction_pipeline = None
    metadata = []
    reconstruction_reports = []

    for output_index, item in enumerate(assignments):
        part = item["part"]
        full_canvas = item["image"]
        source_index = item["source_index"]
        part_id = _sanitize(part.get("id") or f"qwen_layer_{source_index:02d}")
        reconstruction = _initial_reconstruction_status(part)
        should_reconstruct = output_index in reconstruction_queue
        if should_reconstruct:
            try:
                if reconstruction_pipeline is None:
                    reconstruction_pipeline = _load_reconstruction_pipeline(reconstruction_model)
                full_canvas, reconstruction = _reconstruct_layer(
                    pipe=reconstruction_pipeline,
                    run_dir=run_dir,
                    source_image=source_image,
                    layer=full_canvas,
                    part={**part, "id": part_id},
                    steps=reconstruction_steps,
                    max_resolution=reconstruction_resolution,
                    seed=args.seed + 1000 + output_index,
                )
            except Exception as exc:  # keep the paid coarse decomposition instead of throwing it away
                reconstruction = {
                    "required": bool(part.get("needs_reconstruction", False)),
                    "status": "failed",
                    "warnings": [f"Qwen Image Edit reconstruction failed: {exc}"],
                }
                (run_dir / "reconstruction" / f"{part_id}_error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
        path = raw_dir / f"layer_{output_index:02d}_{part_id}.png"
        full_canvas.save(path)
        reconstruction_reports.append({"id": part_id, **reconstruction})
        metadata.append(
            {
                "id": part_id,
                "name": _title(part_id),
                "group": part.get("group", "ROOT"),
                "z_index": int(part.get("depth", 10 + output_index * 10)),
                "layer": str(path.relative_to(run_dir)),
                "mask": "",
                "confidence": float(part.get("confidence", 0.72)),
                "reconstruction": reconstruction,
                "warnings": [
                    f"generated on Vast.ai with Qwen Image Layered; source_layer={source_index}",
                    *reconstruction.get("warnings", []),
                ],
            }
        )

    if reconstruction_pipeline is not None:
        del reconstruction_pipeline
        _clear_cuda_cache()

    (run_dir / "decomposition" / "layers.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_preview(run_dir, metadata)
    (run_dir / "reconstruction" / "reconstruction_report.json").write_text(
        json.dumps(reconstruction_reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "decomposition" / "vast_result.json").write_text(
        json.dumps(
            {
                "model": model,
                "seed": args.seed,
                "steps": steps,
                "resolution": resolution,
                "layers": len(metadata),
                "prompt": prompt,
                "reconstruction_model": reconstruction_model,
                "reconstruction_steps": reconstruction_steps,
                "reconstruction_resolution": reconstruction_resolution,
                "max_reconstruction_tasks": max_reconstruction_tasks,
                "reconstruction": reconstruction_reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


def _load_job_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "vast" / "vast_job.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_analysis(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "character_analysis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_targets(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "material_targets.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prompt(run_dir: Path) -> str:
    path = run_dir / "analysis" / "qwen_prompt.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _build_prompt(analysis: dict[str, Any], targets: dict[str, Any]) -> str:
    analysis_parts = analysis.get("parts", []) if isinstance(analysis, dict) else []
    parts = targets.get("parts") or analysis_parts
    part_names = ", ".join(part.get("id", "") for part in parts[:24] if part.get("id"))
    reconstruction_names = ", ".join(
        part.get("id", "") for part in parts[:24] if part.get("id") and part.get("needs_reconstruction")
    )
    return (
        "Decompose this anime character illustration into transparent RGBA layers for Live2D material separation. "
        "Preserve source identity, line art, colors, geometry, and visible pixels. "
        "Create coherent semantic materials, not arbitrary color fragments. "
        "Layers must be useful for rigging, with separate face, eyes, mouth, hair, body, clothes, and accessories. "
        f"Expected materials: {part_names}. "
        f"Materials that require hidden-area reconstruction after decomposition: {reconstruction_names}."
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


def _assign_layers_to_targets(generated: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "part": {
                    "id": f"qwen_layer_{item['index']:02d}",
                    "group": "ROOT",
                    "depth": 200 + item["index"],
                    "confidence": 0.65,
                    "needs_reconstruction": False,
                },
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


def _select_reconstruction_queue(assignments: list[dict[str, Any]], limit: int, skip: bool) -> set[int]:
    if skip or limit <= 0:
        return set()
    priority = {
        "head_face": 0,
        "hair_back": 1,
        "body": 2,
        "clothes": 3,
        "hair_front": 4,
        "accessories": 5,
    }
    candidates = []
    for index, item in enumerate(assignments):
        part = item["part"]
        if not part.get("needs_reconstruction"):
            continue
        part_id = str(part.get("id") or "")
        candidates.append((priority.get(part_id, 20), index))
    return {index for _, index in sorted(candidates)[:limit]}


def _initial_reconstruction_status(part: dict[str, Any]) -> dict[str, Any]:
    required = bool(part.get("needs_reconstruction", False))
    return {
        "required": required,
        "status": "pending" if required else "not_required",
        "warnings": [] if required else ["hidden-area reconstruction not required for this material"],
    }


def _load_reconstruction_pipeline(model: str):
    try:
        from diffusers import QwenImageEditInpaintPipeline

        pipe = QwenImageEditInpaintPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    except ImportError:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _reconstruct_layer(
    pipe: Any,
    run_dir: Path,
    source_image: Image.Image,
    layer: Image.Image,
    part: dict[str, Any],
    steps: int,
    max_resolution: int,
    seed: int,
) -> tuple[Image.Image, dict[str, Any]]:
    part_id = _sanitize(str(part.get("id") or "material"))
    out_dir = run_dir / "reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)
    required_mask = _required_mask_for_part(part, source_image.size)
    visible_mask = _binary_alpha(layer)
    missing_mask = ImageChops.subtract(required_mask, visible_mask)
    missing_mask = _clean_mask(missing_mask)
    missing_bbox = missing_mask.getbbox()
    if not missing_bbox:
        return layer, {
            "required": True,
            "status": "complete",
            "warnings": ["no missing hidden pixels detected after mask comparison"],
        }

    required_pixels = _mask_pixels(required_mask)
    missing_pixels = _mask_pixels(missing_mask)
    missing_fraction = missing_pixels / max(1, required_pixels)
    if missing_fraction > _max_missing_fraction(part):
        _save_reconstruction_masks(out_dir, part_id, layer, required_mask, missing_mask)
        return layer, {
            "required": True,
            "status": "failed",
            "missing_fraction": round(missing_fraction, 4),
            "warnings": [
                "missing mask is too large for safe inpainting; coarse layer assignment likely needs review"
            ],
        }

    crop_box = _expand_xyxy(missing_bbox, source_image.size, pad_ratio=0.55, min_pad=48)
    source_crop = source_image.crop(crop_box).convert("RGB")
    mask_crop = missing_mask.crop(crop_box).convert("L")
    original_crop_size = source_crop.size
    source_edit, mask_edit, resize_scale = _resize_pair_for_edit(source_crop, mask_crop, max_resolution)
    prompt = _reconstruction_prompt(part)
    with torch.inference_mode():
        output = _call_reconstruction_pipe(
            pipe=pipe,
            image=source_edit,
            mask=mask_edit,
            prompt=prompt,
            steps=steps,
            seed=seed,
        )
    generated_crop = output.images[0].convert("RGBA")
    if resize_scale != 1.0:
        generated_crop = generated_crop.resize(original_crop_size, Image.Resampling.LANCZOS)

    generated_pixels = Image.new("RGBA", source_image.size, (0, 0, 0, 0))
    generated_pixels_crop = generated_crop.copy()
    generated_pixels_crop.putalpha(mask_crop)
    generated_pixels.alpha_composite(generated_pixels_crop, dest=(crop_box[0], crop_box[1]))

    final = layer.copy()
    final.alpha_composite(generated_pixels)

    _save_reconstruction_diagnostics(
        out_dir=out_dir,
        part_id=part_id,
        source_image=source_image,
        before=layer,
        required_mask=required_mask,
        missing_mask=missing_mask,
        generated_pixels=generated_pixels,
        final=final,
        crop_box=crop_box,
        generated_crop=generated_crop,
    )
    return final, {
        "required": True,
        "status": "complete",
        "missing_fraction": round(missing_fraction, 4),
        "missing_bbox": [missing_bbox[0], missing_bbox[1], missing_bbox[2] - missing_bbox[0], missing_bbox[3] - missing_bbox[1]],
        "crop_bbox": [crop_box[0], crop_box[1], crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
        "warnings": [],
    }


def _call_reconstruction_pipe(
    pipe: Any,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    steps: int,
    seed: int,
) -> Any:
    common = {
        "image": image,
        "mask_image": mask,
        "prompt": prompt,
        "negative_prompt": "changed face, changed identity, changed colors, different art style, blurry, "
        "extra objects, text, watermark, white background",
        "num_inference_steps": steps,
        "generator": _generator_for_pipeline(pipe, seed),
    }
    attempts = [
        {**common, "strength": 1.0, "true_cfg_scale": 4.0},
        {**common, "strength": 1.0},
        common,
        {
            "image": image,
            "mask_image": mask,
            "prompt": prompt,
            "num_inference_steps": steps,
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return pipe(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("reconstruction pipeline call failed")


def _generator_for_pipeline(pipe: Any, seed: int) -> torch.Generator:
    try:
        device = getattr(pipe, "_execution_device", None)
        if device is None and torch.cuda.is_available():
            device = "cuda"
        if device is None:
            device = "cpu"
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.manual_seed(seed)


def _required_mask_for_part(part: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    bbox = _clamp_bbox(part.get("bbox") or [0, 0, size[0], size[1]], size)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return mask
    x, y, w, h = bbox
    draw = ImageDraw.Draw(mask)
    shape = (x, y, x + w, y + h)
    group = str(part.get("group") or "").upper()
    part_id = str(part.get("id") or "")
    if group == "HEAD" or part_id == "head_face":
        draw.rounded_rectangle(shape, radius=max(8, int(min(w, h) * 0.22)), fill=255)
    elif group in {"EYES", "BROWS", "MOUTH"}:
        draw.rounded_rectangle(shape, radius=max(2, int(min(w, h) * 0.25)), fill=255)
    else:
        draw.rectangle(shape, fill=255)
    return _soften_mask(mask)


def _binary_alpha(image: Image.Image) -> Image.Image:
    return image.convert("RGBA").getchannel("A").point(lambda value: 255 if value > 16 else 0)


def _clean_mask(mask: Image.Image) -> Image.Image:
    cleaned = mask.convert("L").point(lambda value: 255 if value > 18 else 0)
    cleaned = cleaned.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))
    return _soften_mask(cleaned)


def _soften_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("L").filter(ImageFilter.GaussianBlur(0.45)).point(lambda value: 255 if value > 8 else 0)


def _mask_pixels(mask: Image.Image) -> int:
    histogram = mask.convert("L").histogram()
    return sum(histogram[1:])


def _max_missing_fraction(part: dict[str, Any]) -> float:
    part_id = str(part.get("id") or "")
    group = str(part.get("group") or "").upper()
    if part_id == "head_face":
        return 0.72
    if group == "HAIR":
        return 0.92
    if group in {"BODY", "CLOTHES"}:
        return 0.70
    return 0.65


def _resize_pair_for_edit(
    image: Image.Image, mask: Image.Image, max_resolution: int
) -> tuple[Image.Image, Image.Image, float]:
    longest = max(image.size)
    if longest <= max_resolution:
        return image, mask, 1.0
    scale = max_resolution / longest
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return (
        image.resize(new_size, Image.Resampling.LANCZOS),
        mask.resize(new_size, Image.Resampling.NEAREST),
        scale,
    )


def _reconstruction_prompt(part: dict[str, Any]) -> str:
    part_id = str(part.get("id") or "material")
    group = str(part.get("group") or "material")
    return (
        f"Reconstruct only the masked hidden area for the Live2D material '{part_id}' ({group}). "
        "Continue the exact same anime illustration behind the occluding object. Preserve identity, anatomy, "
        "line art, colors, lighting, shading style, proportions, and geometry. Do not redesign the character. "
        "Do not alter unmasked visible pixels. The completed pixels must look like they were originally painted "
        "as part of this same character layer for Live2D rigging."
    )


def _save_reconstruction_masks(
    out_dir: Path, part_id: str, before: Image.Image, required_mask: Image.Image, missing_mask: Image.Image
) -> None:
    before.save(out_dir / f"{part_id}_before.png")
    required_mask.save(out_dir / f"{part_id}_required_mask.png")
    missing_mask.save(out_dir / f"{part_id}_missing_mask.png")


def _save_reconstruction_diagnostics(
    out_dir: Path,
    part_id: str,
    source_image: Image.Image,
    before: Image.Image,
    required_mask: Image.Image,
    missing_mask: Image.Image,
    generated_pixels: Image.Image,
    final: Image.Image,
    crop_box: tuple[int, int, int, int],
    generated_crop: Image.Image,
) -> None:
    _save_reconstruction_masks(out_dir, part_id, before, required_mask, missing_mask)
    generated_pixels.save(out_dir / f"{part_id}_generated_missing.png")
    generated_crop.save(out_dir / f"{part_id}_generated_crop.png")
    final.save(out_dir / f"{part_id}_final.png")
    comparison = _comparison_sheet(
        [
            ("source crop", source_image.crop(crop_box).convert("RGBA")),
            ("before material", before.crop(crop_box)),
            ("missing mask", Image.merge("RGBA", [missing_mask.crop(crop_box)] * 4)),
            ("generated", generated_pixels.crop(crop_box)),
            ("final material", final.crop(crop_box)),
        ]
    )
    comparison.save(out_dir / f"{part_id}_comparison.png")


def _comparison_sheet(items: list[tuple[str, Image.Image]]) -> Image.Image:
    thumb_w = 220
    label_h = 22
    sheet = Image.new("RGB", (thumb_w * len(items), thumb_w + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        thumb = Image.new("RGBA", image.size, (255, 255, 255, 255))
        thumb.alpha_composite(image.convert("RGBA"))
        thumb.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
        x = index * thumb_w + (thumb_w - thumb.width) // 2
        sheet.paste(thumb.convert("RGB"), (x, label_h))
        draw.text((index * thumb_w + 4, 4), label, fill=(20, 20, 20))
    return sheet


def _intersection_area(a: list[int], b: list[int]) -> int:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _clamp_bbox(bbox: list[int], size: tuple[int, int]) -> list[int]:
    x, y, w, h = [int(value) for value in bbox[:4]]
    width, height = size
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(0, min(width, x + w))
    bottom = max(0, min(height, y + h))
    return [left, top, max(0, right - left), max(0, bottom - top)]


def _expand_xyxy(
    bbox: tuple[int, int, int, int], size: tuple[int, int], pad_ratio: float = 0.35, min_pad: int = 32
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    pad = max(min_pad, int(max(width, height) * pad_ratio))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(size[0], right + pad),
        min(size[1], bottom + pad),
    )


def _write_preview(run_dir: Path, metadata: list[dict[str, Any]]) -> None:
    source = Image.open(run_dir / "upscale" / "master_2x.png").convert("RGBA")
    composite = Image.new("RGBA", source.size, (0, 0, 0, 0))
    for item in sorted(metadata, key=lambda row: row["z_index"]):
        composite.alpha_composite(Image.open(run_dir / item["layer"]).convert("RGBA"))
    preview = run_dir / "decomposition" / "preview.png"
    composite.save(preview)

    thumb_w = 220
    label_h = 24
    rows = max(1, len(metadata))
    columns = 4
    sheet = Image.new("RGB", (thumb_w * columns, rows * (thumb_w + label_h)), "#f7f7f7")
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(metadata):
        image = Image.open(run_dir / item["layer"]).convert("RGBA")
        y = row * (thumb_w + label_h)
        backgrounds = [
            ("checker", _checkerboard(image.size)),
            ("white", Image.new("RGBA", image.size, "white")),
            ("black", Image.new("RGBA", image.size, "black")),
        ]
        for column, (_, background) in enumerate(backgrounds):
            preview_image = background.copy()
            preview_image.alpha_composite(image)
            preview_image.thumbnail((thumb_w, thumb_w), Image.Resampling.LANCZOS)
            x = column * thumb_w + (thumb_w - preview_image.width) // 2
            sheet.paste(preview_image.convert("RGB"), (x, y + label_h))
        draw.text((6, y + 4), f"{item['id']}  {item['group']}  z={item['z_index']}", fill="#111111")
        info = f"recon={item.get('reconstruction', {}).get('status', 'n/a')}"
        draw.text((thumb_w * 3 + 8, y + label_h + 8), info, fill="#111111")
    sheet.save(run_dir / "decomposition" / "preview_sheet.png")


def _checkerboard(size: tuple[int, int], tile: int = 24) -> Image.Image:
    image = Image.new("RGBA", size, "#ffffff")
    draw = ImageDraw.Draw(image)
    width, height = size
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = "#d8d8d8" if (x // tile + y // tile) % 2 else "#f4f4f4"
            draw.rectangle((x, y, min(x + tile, width), min(y + tile, height)), fill=color)
    return image


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return cleaned or "material"


def _title(value: str) -> str:
    return " ".join(piece.capitalize() for piece in value.split("_") if piece)


def _clear_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
