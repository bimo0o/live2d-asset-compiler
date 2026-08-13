from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from src.ai.qwen_layered import DecomposedLayer, QwenLayeredClient
from src.ai.openrouter import OpenRouterClient
from src.ai.upscaler import FaithfulUpscaler
from src.cloud.vast import write_vast_job_package
from src.pipeline.context import PipelineContext
from src.psd.writer import PsdLayer, is_valid_psd, write_psd
from src.schemas.manifest import Canvas, Manifest, ManifestPart, ValidationStatus
from src.utils.hashing import file_sha256
from src.utils.images import (
    SUPPORTED_EXTENSIONS,
    alpha_bbox,
    checkerboard,
    composite_on,
    difference_map,
    has_visible_pixels,
    open_rgba,
    save_png,
)
from src.validation.materials import alpha_coverage, validate_material_image


class Stage:
    name = "stage"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        raise NotImplementedError


class InputStage(Stage):
    name = "source"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        input_override = kwargs.get("input_override")
        source_path = self._find_input(Path(context.config.input.path), input_override)
        with Image.open(source_path) as source:
            source.verify()
        image = open_rgba(source_path)
        self._validate_image(image, source_path)

        destination = context.dir("source") / f"master_original{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        save_png(image, context.dir("previews") / "original.png")
        context.record_stage(self.name, "complete", [destination, context.dir("previews") / "original.png"], file_sha256(destination))

    @staticmethod
    def _find_input(input_dir: Path, explicit: Path | None) -> Path:
        if explicit:
            if explicit.exists() and explicit.suffix.lower() in SUPPORTED_EXTENSIONS:
                return explicit
            raise FileNotFoundError(f"Input image is invalid or unsupported: {explicit}")
        if not input_dir.exists():
            raise FileNotFoundError("Missing input directory. Create input/ and add master.png, master.jpg, or master.webp.")
        for name in ("master.png", "master.jpg", "master.jpeg", "master.webp"):
            candidate = input_dir / name
            if candidate.exists():
                return candidate
        candidates = [path for path in input_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError("Put exactly one supported image into input/, preferably named master.png.")

    @staticmethod
    def _validate_image(image: Image.Image, path: Path) -> None:
        width, height = image.size
        if width < 256 or height < 256:
            raise ValueError(f"Input is too small for Live2D material separation: {path} is {width}x{height}")
        ratio = max(width / height, height / width)
        if ratio > 4.0:
            raise ValueError(f"Input aspect ratio is too extreme: {width}x{height}")
        if not has_visible_pixels(image):
            raise ValueError("Input image has no visible pixels.")


class UpscaleStage(Stage):
    name = "upscale"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        source_path = next((context.run_dir / "source").glob("master_original.*"))
        image = open_rgba(source_path)
        upscaled = FaithfulUpscaler(context.config.upscale.scale).upscale(image) if context.config.upscale.enabled else image
        destination = context.dir("upscale") / f"master_{context.config.upscale.scale}x.png"
        preview = context.dir("previews") / "upscaled.png"
        save_png(upscaled, destination)
        save_png(upscaled, preview)
        context.record_stage(self.name, "complete", [destination, preview], file_sha256(source_path))


class AnalysisStage(Stage):
    name = "analysis"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        source = context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png"
        output_dir = context.dir("analysis")
        output = output_dir / "character_analysis.json"
        warnings: list[str] = []
        client = OpenRouterClient(
            enabled=context.config.openrouter.enabled,
            api_key_env=context.config.openrouter.api_key_env,
            vision_model=context.config.openrouter.vision_model,
            image_model=context.config.openrouter.image_model,
        )
        try:
            analysis = client.analyze_character(source)
        except Exception as exc:
            if context.config.openrouter.require:
                raise
            analysis = {
                "status": "skipped",
                "character": {},
                "parts": [],
                "warnings": [f"OpenRouter analysis failed: {exc}"],
            }
            warnings.append(str(exc))
        output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        material_targets = _material_targets_from_analysis(analysis, kwargs.get("quality") or context.run.info.quality, source)
        targets_path = output_dir / "material_targets.json"
        prompt_path = output_dir / "qwen_prompt.txt"
        overlay_path = context.dir("previews") / "target_overlay.png"
        targets_path.write_text(json.dumps(material_targets, indent=2, ensure_ascii=False), encoding="utf-8")
        prompt_path.write_text(_qwen_prompt(material_targets), encoding="utf-8")
        _write_target_overlay(source, material_targets, overlay_path)
        context.record_stage(self.name, "complete", [output, targets_path, prompt_path, overlay_path], file_sha256(source), warnings)

    def write_fallback(self, context: PipelineContext, quality: str, reason: str) -> None:
        source = context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png"
        output_dir = context.dir("analysis")
        analysis = {
            "status": "review_required",
            "character": {},
            "parts": [],
            "warnings": [reason],
        }
        material_targets = _material_targets_from_analysis(analysis, quality, source)
        output = output_dir / "character_analysis.json"
        targets_path = output_dir / "material_targets.json"
        prompt_path = output_dir / "qwen_prompt.txt"
        overlay_path = context.dir("previews") / "target_overlay.png"
        output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        targets_path.write_text(json.dumps(material_targets, indent=2, ensure_ascii=False), encoding="utf-8")
        prompt_path.write_text(_qwen_prompt(material_targets), encoding="utf-8")
        _write_target_overlay(source, material_targets, overlay_path)
        context.record_stage(self.name, "review_required", [output, targets_path, prompt_path, overlay_path], file_sha256(source), [reason])


class DecompositionStage(Stage):
    name = "decomposition"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        source = context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png"
        image = open_rgba(source)
        if context.config.decomposition.provider == "vast_qwen_layered":
            remote_outputs = self._remote_outputs(context)
            if remote_outputs:
                context.record_stage(self.name, "complete", remote_outputs, file_sha256(source))
                return
            job_files = write_vast_job_package(context.run_dir, context.config)
            context.record_stage(
                self.name,
                "review_required",
                job_files,
                file_sha256(source),
                [
                    "GPU decomposition is required. Upload this run to Vast.ai, run vast/run_remote_decomposition.py, download outputs, then resume."
                ],
            )
            raise RuntimeError(
                "Decomposition provider is Vast.ai Qwen Layered. Remote artifacts are not present yet. "
                f"Prepared job package: {context.run_dir / 'vast'}"
            )
        if context.config.decomposition.provider == "local_fallback" and not context.config.execution.allow_local_fallback:
            raise RuntimeError("Local fallback decomposition is disabled by execution.allow_local_fallback=false.")
        client = QwenLayeredClient(
            context.config.decomposition.provider,
            context.config.decomposition.model,
            context.config.decomposition.layers,
        )
        layers = client.decompose(image)
        raw_dir = context.dir("decomposition") / "raw"
        mask_dir = context.dir("segmentation") / "masks"
        outputs: list[Path] = []
        metadata = []
        for index, layer in enumerate(layers):
            layer_path = raw_dir / f"layer_{index:02d}_{layer.id}.png"
            mask_path = mask_dir / f"{layer.id}_mask.png"
            save_png(layer.image, layer_path)
            save_png(layer.mask, mask_path)
            outputs.extend([layer_path, mask_path])
            metadata.append(
                {
                    "id": layer.id,
                    "name": layer.name,
                    "group": layer.group,
                    "z_index": layer.z_index,
                    "layer": str(layer_path.relative_to(context.run_dir)),
                    "mask": str(mask_path.relative_to(context.run_dir)),
                    "confidence": layer.confidence,
                    "warnings": layer.warnings,
                }
            )
        metadata_path = context.dir("decomposition") / "layers.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        sheet = context.dir("previews") / "layer_sheet.png"
        self._contact_sheet(layers, sheet)
        outputs.extend([metadata_path, sheet])
        context.record_stage(self.name, "complete", outputs, file_sha256(source))

    @staticmethod
    def _remote_outputs(context: PipelineContext) -> list[Path]:
        metadata = context.run_dir / "decomposition" / "layers.json"
        raw_dir = context.run_dir / "decomposition" / "raw"
        if not raw_dir.exists():
            return []
        layer_files = sorted(raw_dir.glob("*.png"))
        if not layer_files:
            return []
        if not metadata.exists():
            metadata_items = []
            for index, path in enumerate(layer_files):
                part_id = _sanitize_part_id(path.stem.replace(f"layer_{index:02d}_", ""))
                group = _guess_group(part_id)
                metadata_items.append(
                    {
                        "id": part_id,
                        "name": _title_name(part_id),
                        "group": group,
                        "z_index": 10 + index * 10,
                        "layer": str(path.relative_to(context.run_dir)),
                        "mask": "",
                        "confidence": 0.75,
                        "warnings": ["metadata inferred locally from Vast layer filename"],
                    }
                )
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(json.dumps(metadata_items, indent=2), encoding="utf-8")
        masks = []
        mask_dir = context.dir("segmentation") / "masks"
        data = json.loads(metadata.read_text(encoding="utf-8"))
        changed = False
        for item in data:
            layer_path = context.run_dir / item["layer"]
            mask_path = context.run_dir / (item.get("mask") or f"segmentation/masks/{item['id']}_mask.png")
            if not mask_path.exists():
                layer = open_rgba(layer_path)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                save_png(layer.getchannel("A"), mask_path)
            if not item.get("mask"):
                item["mask"] = str(mask_path.relative_to(context.run_dir))
                changed = True
            masks.append(mask_path)
        if changed:
            metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")
        sheet = context.dir("previews") / "layer_sheet.png"
        if not sheet.exists():
            layers = []
            for item in data:
                layer_image = open_rgba(context.run_dir / item["layer"])
                mask_image = open_rgba(context.run_dir / item["mask"]).getchannel("A") if (context.run_dir / item["mask"]).exists() else layer_image.getchannel("A")
                layers.append(
                    DecomposedLayer(
                        item["id"],
                        item["name"],
                        item["group"],
                        item["z_index"],
                        layer_image,
                        mask_image,
                        item.get("confidence", 0.75),
                        item.get("warnings", []),
                    )
                )
            DecompositionStage._contact_sheet(layers, sheet)
            layer_files.append(sheet)
        return [metadata, *layer_files, *masks]

    @staticmethod
    def _contact_sheet(layers: list[DecomposedLayer], path: Path) -> None:
        thumb_width = 220
        label_height = 24
        columns = 3
        rows = len(layers)
        sheet = Image.new("RGB", (thumb_width * columns, rows * (thumb_width + label_height)), "#f7f7f7")
        draw = ImageDraw.Draw(sheet)
        for row, layer in enumerate(layers):
            y = row * (thumb_width + label_height)
            backgrounds = [checkerboard(layer.image.size), Image.new("RGB", layer.image.size, "white"), Image.new("RGB", layer.image.size, "black")]
            for column, background in enumerate(backgrounds):
                preview = composite_on(layer.image, background)
                preview.thumbnail((thumb_width, thumb_width), Image.Resampling.LANCZOS)
                x = column * thumb_width + (thumb_width - preview.width) // 2
                sheet.paste(preview, (x, y + label_height))
            draw.text((6, y + 4), f"{layer.id}  {layer.group}  z={layer.z_index}", fill="#111111")
        save_png(sheet, path)


class ManifestStage(Stage):
    name = "manifest"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        quality = kwargs.get("quality") or context.run.info.quality
        source = context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png"
        image = open_rgba(source)
        metadata = json.loads((context.run_dir / "decomposition" / "layers.json").read_text(encoding="utf-8"))
        parts = []
        warnings = []
        if not metadata:
            raise ValueError("Decomposition metadata is empty; cannot build manifest.")
        for item in metadata:
            layer_path = context.run_dir / item["layer"]
            mask_path = context.run_dir / item["mask"]
            if not layer_path.exists():
                raise FileNotFoundError(f"Missing decomposition layer: {layer_path}")
            layer_image = open_rgba(layer_path)
            if layer_image.size != image.size:
                raise ValueError(f"Layer {layer_path} has size {layer_image.size}, expected {image.size}")
            item_warnings = list(item.get("warnings") or [])
            item_warnings.extend(validate_material_image(layer_path, image.size))
            bbox = alpha_bbox(layer_image)
            if bbox == [0, 0, 0, 0]:
                item_warnings.append("empty material")
            coverage = alpha_coverage(layer_image)
            warnings.extend([f"{item['id']}: {warning}" for warning in item_warnings])
            parts.append(
                ManifestPart(
                    id=item["id"],
                    name=item["name"],
                    group=item["group"],
                    parent=_default_parent(item["group"]),
                    source_layer=item["layer"],
                    mask=item["mask"],
                    material=f"materials/{item['id']}.png",
                    bbox=bbox,
                    z_index=item["z_index"],
                    deformable=item["group"] not in {"ROOT"},
                    physics_candidate=item["group"] in {"HAIR", "ACCESSORIES", "CLOTHES"},
                    physics_type=_physics_type(item["group"]),
                    validation=ValidationStatus(coverage=coverage, confidence=item["confidence"], warnings=item_warnings),
                )
            )
        manifest = Manifest(
            canvas=Canvas(width=image.width, height=image.height),
            source=str(next((context.run_dir / "source").glob("master_original.*")).relative_to(context.run_dir)),
            quality=quality,
            parts=sorted(parts, key=lambda part: part.z_index),
            recommended_parameters=_recommended_parameters(),
            warnings=warnings,
        )
        manifest_path = context.run_dir / "manifest.json"
        manifest.save(manifest_path)
        context.record_stage(self.name, "complete", [manifest_path], file_sha256(source), warnings)


class MaterialStage(Stage):
    name = "materials"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        manifest = Manifest.model_validate_json((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        outputs = []
        for part in manifest.parts:
            source = open_rgba(context.run_dir / part.source_layer)
            destination = context.run_dir / part.material
            save_png(source, destination)
            outputs.append(destination)
        reconstructed = Image.new("RGBA", (manifest.canvas.width, manifest.canvas.height), (0, 0, 0, 0))
        for part in sorted(manifest.parts, key=lambda item: item.z_index):
            reconstructed.alpha_composite(open_rgba(context.run_dir / part.material))
        reconstructed_path = context.dir("previews") / "reconstructed.png"
        diff_path = context.dir("previews") / "difference.png"
        save_png(reconstructed, reconstructed_path)
        source = open_rgba(context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png")
        save_png(difference_map(source, reconstructed), diff_path)
        outputs.extend([reconstructed_path, diff_path])
        context.record_stage(self.name, "complete", outputs)


class PsdStage(Stage):
    name = "psd"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        if not context.config.psd.enabled:
            context.record_stage(self.name, "skipped")
            return
        manifest = Manifest.model_validate_json((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        layers = [
            PsdLayer(_psd_name(part.name), open_rgba(context.run_dir / part.material))
            for part in sorted(manifest.parts, key=lambda item: item.z_index)
        ]
        output_dir = context.dir("psd")
        master = output_dir / "character_material_separation.psd"
        import_psd = output_dir / "character_import.psd"
        write_psd(master, (manifest.canvas.width, manifest.canvas.height), layers)
        write_psd(import_psd, (manifest.canvas.width, manifest.canvas.height), layers)
        if not is_valid_psd(master) or not is_valid_psd(import_psd):
            raise ValueError("PSD creation failed validation.")
        context.record_stage(self.name, "complete", [master, import_psd])


class ReportStage(Stage):
    name = "reports"

    def run(self, context: PipelineContext, **kwargs: Any) -> None:
        manifest = Manifest.model_validate_json((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = "\n".join(
            f"<tr><td>{html.escape(part.id)}</td><td>{html.escape(part.group)}</td><td>{part.z_index}</td>"
            f"<td>{part.validation.confidence:.2f}</td><td>{html.escape(', '.join(part.validation.warnings))}</td></tr>"
            for part in manifest.parts
        )
        low_confidence = sum(1 for part in manifest.parts if part.validation.confidence < context.config.validation.min_confidence)
        status = "READY_FOR_LIVE2D_REVIEW" if low_confidence == 0 else "REVIEW_REQUIRED"
        report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live2D Asset Compiler Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2328; }}
    h1 {{ margin-bottom: 0; }}
    img {{ max-width: 320px; border: 1px solid #d0d7de; background: #fff; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Live2D Asset Compiler Report</h1>
  <p class="status">Status: {status}</p>
  <p>Canvas: {manifest.canvas.width}x{manifest.canvas.height}<br>Materials: {len(manifest.parts)}<br>Low confidence: {low_confidence}</p>
  <div class="grid">
    <figure><img src="../previews/original.png"><figcaption>Original</figcaption></figure>
    <figure><img src="../previews/upscaled.png"><figcaption>Upscaled</figcaption></figure>
    <figure><img src="../previews/reconstructed.png"><figcaption>Reconstructed</figcaption></figure>
    <figure><img src="../previews/difference.png"><figcaption>Difference</figcaption></figure>
  </div>
  <h2>Layer Contact Sheet</h2>
  <p><img src="../previews/layer_sheet.png" style="max-width: 100%;"></p>
  <h2>Materials</h2>
  <table>
    <thead><tr><th>ID</th><th>Group</th><th>Z</th><th>Confidence</th><th>Warnings</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
        report_path = context.dir("reports") / "report.html"
        report_path.write_text(report, encoding="utf-8")
        context.record_stage(self.name, "complete", [report_path])


def _default_parent(group: str) -> str | None:
    return {
        "HAIR": "head",
        "EYES": "head",
        "MOUTH": "head",
        "HEAD": "root",
        "BODY": "root",
        "CLOTHES": "body",
        "ACCESSORIES": "root",
    }.get(group)


def _physics_type(group: str) -> str | None:
    return {
        "HAIR": "hair",
        "CLOTHES": "cloth",
        "ACCESSORIES": "accessory",
    }.get(group)


def _recommended_parameters() -> list[str]:
    return [
        "ParamAngleX",
        "ParamAngleY",
        "ParamAngleZ",
        "ParamEyeLOpen",
        "ParamEyeROpen",
        "ParamEyeBallX",
        "ParamEyeBallY",
        "ParamMouthOpenY",
        "ParamMouthForm",
    ]


def _material_targets_from_analysis(analysis: dict[str, Any], quality: str, source: Path | None = None) -> dict[str, Any]:
    parts = analysis.get("parts", []) if isinstance(analysis, dict) else []
    clean_parts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        confidence = float(part.get("confidence", 0.0) or 0.0)
        if confidence < 0.35:
            continue
        clean_parts.append(
            {
                "id": _sanitize_part_id(str(part.get("id") or "part")),
                "group": str(part.get("group") or "ROOT"),
                "bbox": part.get("bbox") or [0, 0, 0, 0],
                "depth": int(part.get("depth", 50) or 50),
                "deformable": bool(part.get("deformable", True)),
                "occluded": bool(part.get("occluded", False)),
                "needs_reconstruction": bool(part.get("needs_reconstruction", False)),
                "confidence": confidence,
            }
        )
    if not clean_parts:
        clean_parts = _default_material_targets(quality, source)
    return {
        "quality": quality,
        "parts": sorted(clean_parts, key=lambda item: item["depth"]),
        "notes": [
            "Targets are semantic guidance for Qwen Image Layered.",
            "Qwen may not obey per-layer semantics exactly; validation and refinement happen after remote output.",
        ],
    }


def _default_material_targets(quality: str, source: Path | None) -> list[dict[str, Any]]:
    if source and source.exists():
        canvas = open_rgba(source)
        character_box = _character_bbox(canvas)
    else:
        character_box = [0, 0, 1000, 1800]
    x, y, w, h = character_box
    head = _relative_bbox(character_box, 0.18, 0.00, 0.64, 0.31)
    face = _relative_bbox(head, 0.28, 0.24, 0.44, 0.40)
    eye_l = _relative_bbox(face, 0.08, 0.40, 0.30, 0.16)
    eye_r = _relative_bbox(face, 0.62, 0.40, 0.30, 0.16)
    mouth = _relative_bbox(face, 0.42, 0.64, 0.16, 0.07)
    targets = [
        ("hair_back", "HAIR", _expand_bbox(head, 0.35, 0.22), 20, True),
        ("body", "BODY", _relative_bbox(character_box, 0.20, 0.30, 0.60, 0.70), 35, False),
        ("clothes", "CLOTHES", _relative_bbox(character_box, 0.12, 0.40, 0.76, 0.58), 45, False),
        ("head_face", "HEAD", face, 60, False),
        ("eye_l", "EYES", eye_l, 80, False),
        ("eye_r", "EYES", eye_r, 81, False),
        ("mouth", "MOUTH", mouth, 86, False),
        ("hair_front", "HAIR", _relative_bbox(head, 0.00, 0.00, 1.00, 0.58), 110, True),
    ]
    if quality == "high":
        targets.extend(
            [
                ("brow_l", "BROWS", _relative_bbox(face, 0.08, 0.20, 0.30, 0.10), 82, False),
                ("brow_r", "BROWS", _relative_bbox(face, 0.62, 0.20, 0.30, 0.10), 83, False),
                ("accessories", "ACCESSORIES", _relative_bbox(character_box, 0.00, 0.00, 1.00, 0.60), 115, False),
            ]
        )
    return [
        {
            "id": part_id,
            "group": group,
            "bbox": bbox,
            "depth": depth,
            "deformable": group != "ROOT",
            "occluded": group in {"HAIR", "CLOTHES"},
            "needs_reconstruction": needs_reconstruction,
            "confidence": 0.5,
        }
        for part_id, group, bbox, depth, needs_reconstruction in targets
    ]


def _qwen_prompt(material_targets: dict[str, Any]) -> str:
    parts = material_targets.get("parts", [])
    part_text = ", ".join(f"{item['id']} ({item['group']})" for item in parts[:32])
    return (
        "Decompose this single front-facing anime character illustration into clean transparent RGBA layers "
        "for Live2D material preparation. Preserve the exact source identity, face, costume, colors, line art, "
        "geometry, and visible pixels. Do not redesign the character. Prefer coherent semantic materials over "
        "arbitrary color fragments. Useful Live2D targets include: "
        f"{part_text}. Output layers from back to front where possible."
    )


def _psd_name(name: str) -> str:
    return name.encode("ascii", errors="replace").decode("ascii")[:31]


def _sanitize_part_id(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return cleaned or "material"


def _title_name(part_id: str) -> str:
    return " ".join(piece.capitalize() for piece in part_id.split("_") if piece)


def _guess_group(part_id: str) -> str:
    if "hair" in part_id or "bang" in part_id:
        return "HAIR"
    if "eye" in part_id or "iris" in part_id or "pupil" in part_id:
        return "EYES"
    if "mouth" in part_id or "lip" in part_id or "tongue" in part_id or "teeth" in part_id:
        return "MOUTH"
    if "face" in part_id or "head" in part_id or "ear" in part_id:
        return "HEAD"
    if "cloth" in part_id or "dress" in part_id or "skirt" in part_id:
        return "CLOTHES"
    if "arm" in part_id or "hand" in part_id or "leg" in part_id or "body" in part_id:
        return "BODY"
    if "ribbon" in part_id or "chain" in part_id or "ornament" in part_id:
        return "ACCESSORIES"
    return "ROOT"


def _character_bbox(image: Image.Image) -> list[int]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        return [left, top, right - left, bottom - top]
    rgb = image.convert("RGB")
    white = Image.new("RGB", image.size, "white")
    diff = ImageOps.grayscale(ImageChops.difference(rgb, white))
    bbox = diff.point(lambda value: 255 if value > 18 else 0).getbbox()
    if not bbox:
        return [0, 0, image.width, image.height]
    left, top, right, bottom = bbox
    return [left, top, right - left, bottom - top]


def _relative_bbox(parent: list[int], left: float, top: float, width: float, height: float) -> list[int]:
    x, y, w, h = parent
    return [int(x + w * left), int(y + h * top), max(1, int(w * width)), max(1, int(h * height))]


def _expand_bbox(bbox: list[int], x_pad: float, y_pad: float) -> list[int]:
    x, y, w, h = bbox
    dx = int(w * x_pad)
    dy = int(h * y_pad)
    return [max(0, x - dx), max(0, y - dy), w + dx * 2, h + dy * 2]


def _write_target_overlay(source: Path, material_targets: dict[str, Any], path: Path) -> None:
    image = open_rgba(source)
    preview = image.copy()
    preview.thumbnail((900, 1400), Image.Resampling.LANCZOS)
    scale_x = preview.width / image.width
    scale_y = preview.height / image.height
    canvas = preview.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "EYES": (0, 190, 255, 220),
        "MOUTH": (255, 80, 120, 220),
        "HAIR": (160, 90, 255, 200),
        "HEAD": (255, 190, 60, 210),
        "BODY": (80, 210, 120, 180),
        "CLOTHES": (80, 120, 255, 180),
    }
    for part in material_targets.get("parts", []):
        x, y, w, h = part["bbox"]
        left = int(x * scale_x)
        top = int(y * scale_y)
        right = int((x + w) * scale_x)
        bottom = int((y + h) * scale_y)
        color = colors.get(part["group"], (255, 255, 255, 180))
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        draw.text((left + 4, top + 4), part["id"], fill=color)
    canvas.alpha_composite(overlay)
    save_png(canvas, path)
