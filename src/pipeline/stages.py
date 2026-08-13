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
        source = _analysis_source(context)
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
            openrouter_plan = self._openrouter_plan(client, source, output_dir, kwargs.get("quality") or context.run.info.quality)
            if openrouter_plan.get("parts"):
                analysis = {
                    "status": "openrouter_multi_pass",
                    "character": analysis.get("character", {}),
                    "parts": openrouter_plan["parts"],
                    "warnings": analysis.get("warnings", []) + openrouter_plan.get("warnings", []),
                }
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
        if context.config.openrouter.enabled and not warnings:
            try:
                qa = client.validate_target_overlay(overlay_path)
                (output_dir / "target_overlay_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                warnings.append(f"OpenRouter overlay QA failed: {exc}")
        context.record_stage(self.name, "complete", [output, targets_path, prompt_path, overlay_path], file_sha256(source), warnings)

    def write_fallback(self, context: PipelineContext, quality: str, reason: str) -> None:
        source = _analysis_source(context)
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

    def _openrouter_plan(self, client: OpenRouterClient, source: Path, output_dir: Path, quality: str) -> dict[str, Any]:
        image = open_rgba(source)
        fallback_targets = _default_material_targets(quality, source)
        face_part = next((item for item in fallback_targets if item["id"] == "head_face"), None)
        face_crop_path = output_dir / "face_crop_for_openrouter.png"
        crop_origin = [0, 0]
        crop_box: list[int] | None = None
        if face_part:
            crop_box = _expand_bbox(face_part["bbox"], 0.35, 0.35)
            crop_box = _clamp_bbox(crop_box, image.size)
            crop_origin = [crop_box[0], crop_box[1]]
            _crop_bbox(image, crop_box).save(face_crop_path)
        plan = client.plan_live2d_targets(source, face_crop_path if face_crop_path.exists() else None)
        parts, merge_warnings = _merge_openrouter_regions(
            plan=plan,
            face_crop_origin=crop_origin,
            image_size=image.size,
            character_box=_character_bbox(image),
            fallback_targets=fallback_targets,
            face_crop_box=crop_box,
        )
        plan_path = output_dir / "openrouter_plan.json"
        plan_path.write_text(json.dumps({**plan, "parts_full_canvas": parts}, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"parts": parts, "warnings": plan.get("warnings", []) + merge_warnings}


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
            reconstruction = item.get("reconstruction") or {}
            if reconstruction.get("required") and reconstruction.get("status") != "complete":
                item_warnings.append(f"hidden reconstruction {reconstruction.get('status', 'pending')}")
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
                    reconstruction=item.get("reconstruction") or {
                        "required": bool(item.get("needs_reconstruction", False)),
                        "status": "pending" if item.get("needs_reconstruction", False) else "not_required",
                    },
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
    fallback_parts = _default_material_targets(quality, source)
    fallback_by_id = {item["id"]: dict(item) for item in fallback_parts}
    merged = {item["id"]: dict(item) for item in fallback_parts}
    image_size: tuple[int, int] | None = None
    character_box: list[int] | None = None
    rejected = 0
    accepted = 0
    if source and source.exists():
        canvas = open_rgba(source)
        image_size = canvas.size
        character_box = _character_bbox(canvas)
    for part in parts:
        if not isinstance(part, dict):
            continue
        confidence = _safe_float(part.get("confidence", 0.0), 0.0)
        if confidence < 0.35:
            rejected += 1
            continue
        item = _normalize_region({**part, "confidence": confidence})
        item = _canonicalize_region(item, fallback_by_id)
        if item["id"] not in merged:
            rejected += 1
            continue
        if image_size and character_box:
            selected_bbox = _select_best_bbox_variant(
                item,
                image_size,
                character_box,
                fallback_by_id,
            )
            if not selected_bbox:
                rejected += 1
                continue
            item["bbox"] = selected_bbox
            if not _part_bbox_is_reasonable(item, image_size, character_box, fallback_by_id):
                rejected += 1
                continue
        merged[item["id"]] = item
        accepted += 1
    clean_parts = _ordered_material_parts(merged, fallback_parts)
    notes = [
        "Targets are semantic guidance for Qwen Image Layered.",
        "Qwen may not obey per-layer semantics exactly; validation and refinement happen after remote output.",
        "OpenRouter boxes are geometry-checked; unsafe boxes are replaced by deterministic fallback targets.",
    ]
    if parts:
        notes.append(f"OpenRouter target filter accepted {accepted} candidate(s), rejected {rejected} candidate(s).")
    return {
        "quality": quality,
        "parts": sorted(clean_parts, key=lambda item: item["depth"]),
        "notes": notes,
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
                ("accessories", "ACCESSORIES", _relative_bbox(character_box, 0.12, 0.02, 0.76, 0.38), 115, False),
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
        "geometry, and visible pixels. Do not redesign the character. Create materials that are useful for rigging, "
        "not a flat screenshot split into arbitrary fragments. Prefer coherent semantic materials over arbitrary "
        "color fragments. Preserve transparent alpha. Mark occluded/covered materials so a later reconstruction "
        "pass can complete hidden pixels behind hair, clothes, arms, and accessories. Useful Live2D targets include: "
        f"{part_text}. Output layers from back to front where possible."
    )


def _psd_name(name: str) -> str:
    return name.encode("ascii", errors="replace").decode("ascii")[:31]


def _analysis_source(context: PipelineContext) -> Path:
    neural = context.run_dir / "upscale" / "master_neural_2x.png"
    if neural.exists():
        return neural
    return context.run_dir / "upscale" / f"master_{context.config.upscale.scale}x.png"


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
        if w <= 0 or h <= 0:
            continue
        left = int(x * scale_x)
        top = int(y * scale_y)
        right = int((x + w) * scale_x)
        bottom = int((y + h) * scale_y)
        color = colors.get(part["group"], (255, 255, 255, 180))
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        draw.text((left + 4, top + 4), part["id"], fill=color)
    canvas.alpha_composite(overlay)
    save_png(canvas, path)


def _clamp_bbox(bbox: list[int], size: tuple[int, int]) -> list[int]:
    x, y, w, h = bbox
    width, height = size
    left = max(0, min(width, x))
    top = max(0, min(height, y))
    right = max(0, min(width, x + w))
    bottom = max(0, min(height, y + h))
    return [left, top, max(0, right - left), max(0, bottom - top)]


def _crop_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


def _merge_openrouter_regions(
    plan: dict[str, Any],
    face_crop_origin: list[int],
    image_size: tuple[int, int],
    character_box: list[int],
    fallback_targets: list[dict[str, Any]],
    face_crop_box: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    full_regions = plan.get("full", {}).get("regions", []) if isinstance(plan.get("full"), dict) else []
    face_regions = plan.get("face_crop", {}).get("regions", []) if isinstance(plan.get("face_crop"), dict) else []
    fallback_by_id = {item["id"]: dict(item) for item in fallback_targets}
    merged: dict[str, dict[str, Any]] = {item["id"]: dict(item) for item in fallback_targets}
    warnings: list[str] = []
    accepted = 0
    rejected = 0
    for region in full_regions:
        if not _valid_region(region):
            rejected += 1
            continue
        item = _canonicalize_region(_normalize_region(region), fallback_by_id)
        if item["id"] not in fallback_by_id or item["id"] in _face_detail_ids():
            rejected += 1
            continue
        selected_bbox = _select_best_bbox_variant(item, image_size, character_box, fallback_by_id)
        if not selected_bbox:
            rejected += 1
            continue
        item["bbox"] = selected_bbox
        if not _part_bbox_is_reasonable(item, image_size, character_box, fallback_by_id):
            rejected += 1
            continue
        merged[item["id"]] = _stabilize_known_part(item, fallback_by_id[item["id"]])
        accepted += 1
    for region in face_regions:
        if not _valid_region(region):
            rejected += 1
            continue
        item = _canonicalize_region(_normalize_region(region), fallback_by_id)
        if item["id"] not in fallback_by_id or item["id"] not in _face_detail_ids():
            rejected += 1
            continue
        selected_bbox = _select_best_bbox_variant(
            item,
            image_size,
            character_box,
            fallback_by_id,
            face_crop_origin=face_crop_origin,
            face_crop_box=face_crop_box,
        )
        if not selected_bbox:
            rejected += 1
            continue
        item["bbox"] = selected_bbox
        if not _part_bbox_is_reasonable(item, image_size, character_box, fallback_by_id):
            rejected += 1
            continue
        merged[item["id"]] = _stabilize_known_part(item, fallback_by_id[item["id"]])
        accepted += 1
    if full_regions or face_regions:
        warnings.append(f"OpenRouter region filter accepted {accepted} candidate(s), rejected {rejected} candidate(s).")
    return _ordered_material_parts(merged, fallback_targets), warnings


def _valid_region(region: Any) -> bool:
    return isinstance(region, dict) and isinstance(region.get("bbox"), list) and len(region["bbox"]) == 4 and region.get("id")


def _normalize_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _sanitize_part_id(str(region.get("id") or "part")),
        "group": _normalize_group(str(region.get("group") or "ROOT")),
        "bbox": [_safe_int(value, 0) for value in region.get("bbox", [0, 0, 1, 1])[:4]],
        "depth": _safe_int(region.get("depth", 50), 50),
        "deformable": bool(region.get("deformable", True)),
        "occluded": bool(region.get("occluded", False)),
        "needs_reconstruction": bool(region.get("needs_reconstruction", False)),
        "confidence": _safe_float(region.get("confidence", 0.5), 0.5),
        "reasoning": str(region.get("reasoning", "")),
    }


def _normalize_group(group: str) -> str:
    upper = group.upper()
    aliases = {
        "FACE": "HEAD",
        "EYE": "EYES",
        "EYEBROW": "BROWS",
        "EYEBROWS": "BROWS",
        "BROW": "BROWS",
        "LIPS": "MOUTH",
        "LIP": "MOUTH",
        "CLOTHING": "CLOTHES",
        "CLOTH": "CLOTHES",
        "ACCESSORY": "ACCESSORIES",
        "ORNAMENT": "ACCESSORIES",
    }
    normalized = aliases.get(upper, upper)
    if normalized in {"ROOT", "HEAD", "EYES", "BROWS", "MOUTH", "HAIR", "BODY", "CLOTHES", "ACCESSORIES", "EFFECTS"}:
        return normalized
    return "ROOT"


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _face_detail_ids() -> set[str]:
    return {"eye_l", "eye_r", "brow_l", "brow_r", "mouth", "head_face"}


def _canonicalize_region(item: dict[str, Any], fallback_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    canonical = dict(item)
    part_id = _sanitize_part_id(str(canonical.get("id") or "part"))
    group = _normalize_group(str(canonical.get("group") or _guess_group(part_id)))
    text = f"{part_id} {canonical.get('reasoning', '')}".lower()
    direct_aliases = {
        "face": "head_face",
        "head": "head_face",
        "head_face": "head_face",
        "left_eye": "eye_l",
        "eye_left": "eye_l",
        "l_eye": "eye_l",
        "right_eye": "eye_r",
        "eye_right": "eye_r",
        "r_eye": "eye_r",
        "left_brow": "brow_l",
        "left_eyebrow": "brow_l",
        "brow_left": "brow_l",
        "right_brow": "brow_r",
        "right_eyebrow": "brow_r",
        "brow_right": "brow_r",
        "lips": "mouth",
        "lip": "mouth",
        "mouth": "mouth",
        "torso": "body",
        "upper_body": "body",
        "dress": "clothes",
        "costume": "clothes",
        "outfit": "clothes",
        "clothing": "clothes",
        "cloth": "clothes",
        "accessory": "accessories",
        "ornaments": "accessories",
        "ornament": "accessories",
    }
    if part_id in direct_aliases:
        part_id = direct_aliases[part_id]
    elif part_id not in fallback_by_id:
        if group == "EYES":
            part_id = "eye_r" if _mentions_right(text) else "eye_l"
        elif group == "BROWS":
            part_id = "brow_r" if _mentions_right(text) else "brow_l"
        elif group == "MOUTH":
            part_id = "mouth"
        elif group == "HEAD":
            part_id = "head_face"
        elif group == "BODY":
            part_id = "body"
        elif group == "CLOTHES":
            part_id = "clothes"
        elif group == "ACCESSORIES":
            part_id = "accessories"
        elif group == "HAIR":
            if "back" in text or "rear" in text or "behind" in text:
                part_id = "hair_back"
            else:
                part_id = "hair_front"
    if part_id in fallback_by_id:
        group = str(fallback_by_id[part_id]["group"])
    canonical["id"] = part_id
    canonical["group"] = group
    return canonical


def _mentions_right(text: str) -> bool:
    tokens = {token for token in text.replace("-", "_").split("_") if token}
    return "right" in tokens or "r" in tokens or "right" in text


def _select_best_bbox_variant(
    item: dict[str, Any],
    image_size: tuple[int, int],
    character_box: list[int],
    fallback_by_id: dict[str, dict[str, Any]],
    face_crop_origin: list[int] | None = None,
    face_crop_box: list[int] | None = None,
) -> list[int] | None:
    best_bbox: list[int] | None = None
    best_score = -1.0
    for candidate in _bbox_candidates(item["bbox"], image_size, face_crop_origin, face_crop_box):
        clamped = _clamp_bbox(candidate, image_size)
        if clamped[2] <= 0 or clamped[3] <= 0:
            continue
        test_item = {**item, "bbox": clamped}
        if not _part_bbox_is_reasonable(test_item, image_size, character_box, fallback_by_id):
            continue
        score = _bbox_candidate_score(test_item, character_box, fallback_by_id)
        if score > best_score:
            best_score = score
            best_bbox = clamped
    return best_bbox


def _bbox_candidates(
    bbox: list[int],
    image_size: tuple[int, int],
    face_crop_origin: list[int] | None = None,
    face_crop_box: list[int] | None = None,
) -> list[list[int]]:
    raw = [_safe_int(value, 0) for value in bbox[:4]]
    variants: list[list[int]] = []

    def add(candidate: list[int]) -> None:
        if candidate[2] <= 0 or candidate[3] <= 0:
            return
        if candidate not in variants:
            variants.append(candidate)

    add(raw)
    x, y, a, b = raw
    if a > x and b > y:
        add([x, y, a - x, b - y])

    seed_variants = list(variants)
    for seed in seed_variants:
        add(_scale_bbox(seed, 2.0))
        add(_scale_bbox(seed, 0.5))
        if _looks_like_unit_box(seed):
            width, height = image_size
            add([int(seed[0] * width), int(seed[1] * height), int(seed[2] * width), int(seed[3] * height)])
        if _looks_like_1000_grid(seed):
            width, height = image_size
            add(_scale_bbox_xy(seed, width / 1000.0, height / 1000.0))

    if face_crop_origin and face_crop_box:
        crop_width, crop_height = face_crop_box[2], face_crop_box[3]
        for seed in seed_variants:
            add([seed[0] + face_crop_origin[0], seed[1] + face_crop_origin[1], seed[2], seed[3]])
            add([seed[0] * 2 + face_crop_origin[0], seed[1] * 2 + face_crop_origin[1], seed[2] * 2, seed[3] * 2])
            if _looks_like_1000_grid(seed):
                scaled = _scale_bbox_xy(seed, crop_width / 1000.0, crop_height / 1000.0)
                add([scaled[0] + face_crop_origin[0], scaled[1] + face_crop_origin[1], scaled[2], scaled[3]])
            if _looks_like_unit_box(seed):
                add(
                    [
                        int(seed[0] * crop_width + face_crop_origin[0]),
                        int(seed[1] * crop_height + face_crop_origin[1]),
                        int(seed[2] * crop_width),
                        int(seed[3] * crop_height),
                    ]
                )
    return variants


def _scale_bbox(bbox: list[int], scale: float) -> list[int]:
    return [int(round(value * scale)) for value in bbox]


def _scale_bbox_xy(bbox: list[int], scale_x: float, scale_y: float) -> list[int]:
    return [
        int(round(bbox[0] * scale_x)),
        int(round(bbox[1] * scale_y)),
        int(round(bbox[2] * scale_x)),
        int(round(bbox[3] * scale_y)),
    ]


def _looks_like_1000_grid(bbox: list[int]) -> bool:
    return max(abs(value) for value in bbox) <= 1000 and min(bbox[2], bbox[3]) > 1


def _looks_like_unit_box(bbox: list[int]) -> bool:
    return all(0 <= value <= 1 for value in bbox)


def _part_bbox_is_reasonable(
    item: dict[str, Any],
    image_size: tuple[int, int],
    character_box: list[int],
    fallback_by_id: dict[str, dict[str, Any]],
) -> bool:
    bbox = item["bbox"]
    part_id = str(item.get("id") or "")
    group = _normalize_group(str(item.get("group") or "ROOT"))
    width, height = image_size
    if bbox[2] < 4 or bbox[3] < 4:
        return False
    if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > width or bbox[1] + bbox[3] > height:
        return False
    bbox_area = _bbox_area(bbox)
    canvas_area = max(1, width * height)
    if bbox_area / canvas_area > 0.72:
        return False
    inside_character = _inside_fraction(bbox, character_box)
    if inside_character < (0.08 if group in {"HAIR", "ACCESSORIES"} else 0.18):
        return False

    fallback = fallback_by_id.get(part_id)
    if not fallback:
        return False

    fallback_bbox = fallback["bbox"]
    fallback_area = max(1, _bbox_area(fallback_bbox))
    area_ratio = bbox_area / fallback_area
    if part_id in {"eye_l", "eye_r", "brow_l", "brow_r", "mouth"}:
        face = fallback_by_id.get("head_face", {}).get("bbox", fallback_bbox)
        face_guard = _clamp_bbox(_expand_bbox(face, 0.65, 0.65), image_size)
        if _inside_fraction(bbox, face_guard) < 0.65:
            return False
        if part_id.startswith("eye") and not (0.10 <= area_ratio <= 7.50):
            return False
        if part_id.startswith("brow") and not (0.05 <= area_ratio <= 8.00):
            return False
        if part_id == "mouth" and not (0.04 <= area_ratio <= 10.00):
            return False
        cx, cy = _bbox_center(bbox)
        face_x, face_y, face_w, face_h = face
        face_center_x = face_x + face_w / 2
        if part_id.endswith("_l") and cx > face_center_x + face_w * 0.12:
            return False
        if part_id.endswith("_r") and cx < face_center_x - face_w * 0.12:
            return False
        if part_id == "mouth" and not (face_y + face_h * 0.40 <= cy <= face_y + face_h * 0.88):
            return False
        return True

    limits = {
        "hair_back": (0.12, 4.50),
        "hair_front": (0.08, 3.50),
        "head_face": (0.16, 4.00),
        "body": (0.16, 4.25),
        "clothes": (0.16, 4.25),
        "accessories": (0.03, 5.00),
    }.get(part_id, (0.10, 4.00))
    if not (limits[0] <= area_ratio <= limits[1]):
        return False
    guard_pad = 0.75 if part_id in {"hair_back", "accessories"} else 0.45
    guard = _clamp_bbox(_expand_bbox(fallback_bbox, guard_pad, guard_pad), image_size)
    center = _bbox_center(bbox)
    if part_id not in {"hair_back", "accessories"} and not _point_in_bbox(center, guard):
        return False
    min_inside = 0.30 if part_id in {"hair_back", "accessories"} else 0.50
    if _inside_fraction(bbox, guard) < min_inside and _inside_fraction(fallback_bbox, bbox) < 0.15:
        return False
    return True


def _bbox_candidate_score(
    item: dict[str, Any],
    character_box: list[int],
    fallback_by_id: dict[str, dict[str, Any]],
) -> float:
    bbox = item["bbox"]
    fallback = fallback_by_id.get(item["id"], {})
    fallback_bbox = fallback.get("bbox", character_box)
    area_ratio = _bbox_area(bbox) / max(1, _bbox_area(fallback_bbox))
    ratio_score = max(0.0, 1.0 - abs(1.0 - min(area_ratio, 1 / max(area_ratio, 0.0001))))
    return (
        _iou(bbox, fallback_bbox) * 3.0
        + _inside_fraction(bbox, _expand_bbox(fallback_bbox, 0.80, 0.80)) * 2.0
        + _inside_fraction(bbox, character_box)
        + ratio_score
        + _safe_float(item.get("confidence"), 0.5)
    )


def _ordered_material_parts(merged: dict[str, dict[str, Any]], fallback_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = []
    for fallback in fallback_targets:
        item = dict(merged.get(fallback["id"], fallback))
        item = _stabilize_known_part(item, fallback)
        ordered.append(item)
    return ordered


def _stabilize_known_part(item: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    stable = dict(item)
    stable["id"] = fallback["id"]
    stable["group"] = fallback["group"]
    stable["depth"] = fallback["depth"]
    stable["deformable"] = fallback["deformable"]
    stable["occluded"] = fallback["occluded"]
    stable["needs_reconstruction"] = bool(stable.get("needs_reconstruction", fallback.get("needs_reconstruction", False)))
    stable["confidence"] = max(_safe_float(stable.get("confidence"), 0.5), _safe_float(fallback.get("confidence"), 0.5))
    return stable


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2]) * max(0, bbox[3])


def _intersection_area(a: list[int], b: list[int]) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0, right - left) * max(0, bottom - top)


def _inside_fraction(child: list[int], parent: list[int]) -> float:
    return _intersection_area(child, parent) / max(1, _bbox_area(child))


def _iou(a: list[int], b: list[int]) -> float:
    intersection = _intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / max(1, union)


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2


def _point_in_bbox(point: tuple[float, float], bbox: list[int]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]
