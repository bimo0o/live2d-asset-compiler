from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


class OpenRouterClient:
    """Cloud reasoning adapter.

    Heavy vision/semantic planning belongs to OpenRouter, not the local CPU.
    This adapter intentionally hides HTTP details from pipeline stages.
    """

    def __init__(self, enabled: bool, api_key_env: str, vision_model: str, image_model: str) -> None:
        self.enabled = enabled
        self.api_key_env = api_key_env
        self.vision_model = vision_model
        self.image_model = image_model

    def analyze_character(self, image: Path) -> dict[str, Any]:
        width, height = _image_size(image)
        prompt = (
            "You are analyzing a 2D anime character illustration specifically for Live2D "
            "material separation. Return strict JSON only. Identify visually present "
            "components that should potentially become independent deformable materials. "
            f"The image is exactly {width}x{height} pixels. Use ONLY this full-image pixel "
            "coordinate system with origin at the top-left corner. Every bbox must be inside "
            f"0<=x<{width}, 0<=y<{height}. Do not invent missing objects. "
            "For each component include id, group, bbox [x,y,w,h], depth, deformable, "
            "occluded, needs_reconstruction, confidence, "
            "and short reasoning. Also include character type, pose, full_body, and background."
        )
        return self._vision_json(image, prompt, _character_analysis_schema(), "live2d_character_analysis")

    def plan_live2d_targets(self, full_image: Path, face_crop: Path | None = None) -> dict[str, Any]:
        full_width, full_height = _image_size(full_image)
        full_prompt = (
            "You are a senior Live2D material separation artist. Analyze the full anime character image. "
            f"The image is exactly {full_width}x{full_height} pixels. "
            "Return strict JSON. Identify the character silhouette, head, face, hair groups, clothes, arms, hands, "
            "legs, and accessories. Use pixel coordinates in this exact full image. Do not use a 0-1000 normalized "
            "grid and do not use coordinates from a resized preview. For the full-image pass, prefer these canonical "
            "ids only: hair_back, hair_front, body, clothes, accessories, head_face. Do NOT output eye, eyebrow, or "
            "mouth boxes in the full-image pass; those are handled by a face crop. Be conservative: if unsure, mark "
            "confidence low rather than guessing."
        )
        full_plan = self._vision_json(full_image, full_prompt, _region_plan_schema(), "full_body_live2d_plan")
        face_plan: dict[str, Any] = {"regions": [], "warnings": []}
        if face_crop and face_crop.exists():
            crop_width, crop_height = _image_size(face_crop)
            face_prompt = (
                "Analyze this FACE CROP for Live2D material separation. Return strict JSON. Locate exact visible "
                f"left eye, right eye, eyebrows, mouth, nose, and face contour. The crop is exactly "
                f"{crop_width}x{crop_height} pixels. Use CROP pixel coordinates only, with crop top-left as (0,0). "
                "Use these canonical ids only: eye_l, eye_r, brow_l, brow_r, mouth, head_face. "
                "The mouth box must tightly cover the visible mouth only, not collar, chin shadow, or clothing. "
                "Eye boxes must tightly cover visible eyeball/iris/lashes only, not hair."
            )
            face_plan = self._vision_json(face_crop, face_prompt, _region_plan_schema(), "face_detail_live2d_plan")
        return {"full": full_plan, "face_crop": face_plan, "warnings": []}

    def validate_target_overlay(self, overlay: Path) -> dict[str, Any]:
        prompt = (
            "Review this Live2D target overlay. Return strict JSON. Check whether eye_l, eye_r, and mouth boxes "
            "are tightly aligned to the actual eyes and mouth. Do not be polite. Mark any box that touches hair, "
            "collar, clothing, or misses the intended feature."
        )
        return self._vision_json(overlay, prompt, _overlay_validation_schema(), "live2d_overlay_validation")

    def analyze_layer(self, image: Path, layer: Path) -> dict[str, Any]:
        raise NotImplementedError("OpenRouter layer analysis starts in Phase 2.")

    def validate_result(self, images: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("OpenRouter QA starts in Phase 2.")

    def _vision_json(self, image: Path, prompt: str, schema: dict[str, Any], schema_name: str = "live2d_json") -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "character": {}, "parts": [], "warnings": ["OpenRouter disabled"]}
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"OpenRouter is required, but {self.api_key_env} is not set.")

        image_data = _image_data_url(image)
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {
                "require_parameters": True,
            },
        }
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.live2d-compiler",
                "X-Title": "One-Click AI Live2D Asset Compiler",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: HTTP {error.code}: {detail}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"OpenRouter response did not contain message content: {body}") from exc
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenRouter returned empty content: {body}")
        try:
            return _normalize_response(json.loads(_extract_json_text(content)), schema_name)
        except json.JSONDecodeError as exc:
            preview = content[:1000]
            raise RuntimeError(f"OpenRouter returned non-JSON content for {schema_name}: {preview}") from exc


def _image_data_url(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _character_analysis_schema() -> dict[str, Any]:
    part = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "group": {
                "type": "string",
                "enum": ["ROOT", "HEAD", "EYES", "BROWS", "MOUTH", "HAIR", "BODY", "CLOTHES", "ACCESSORIES", "EFFECTS"],
            },
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "depth": {"type": "integer"},
            "deformable": {"type": "boolean"},
            "occluded": {"type": "boolean"},
            "needs_reconstruction": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": [
            "id",
            "group",
            "bbox",
            "depth",
            "deformable",
            "occluded",
            "needs_reconstruction",
            "confidence",
            "reasoning",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "character": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "pose": {"type": "string"},
                    "full_body": {"type": "boolean"},
                    "background": {"type": "string"},
                },
                "required": ["type", "pose", "full_body", "background"],
                "additionalProperties": False,
            },
            "parts": {"type": "array", "items": part},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["character", "parts", "warnings"],
        "additionalProperties": False,
    }


def _region_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "coordinate_space": {"type": "string"},
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "group": {
                            "type": "string",
                            "enum": ["ROOT", "HEAD", "EYES", "BROWS", "MOUTH", "HAIR", "BODY", "CLOTHES", "ACCESSORIES", "EFFECTS"],
                        },
                        "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                        "depth": {"type": "integer"},
                        "deformable": {"type": "boolean"},
                        "occluded": {"type": "boolean"},
                        "needs_reconstruction": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "group",
                        "bbox",
                        "depth",
                        "deformable",
                        "occluded",
                        "needs_reconstruction",
                        "confidence",
                        "reasoning",
                    ],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["coordinate_space", "regions", "warnings"],
        "additionalProperties": False,
    }


def _overlay_validation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "overall_status": {"type": "string", "enum": ["ok", "needs_fix", "bad"]},
            "checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": ["ok", "too_large", "too_small", "missed", "wrong_object"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["id", "status", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overall_status", "checks", "warnings"],
        "additionalProperties": False,
    }


def _extract_json_text(content: str) -> str:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_response(data: dict[str, Any], schema_name: str) -> dict[str, Any]:
    if schema_name == "live2d_character_analysis":
        character = data.get("character") or {
            "type": data.get("character_type", "anime_character"),
            "pose": data.get("pose", "unknown"),
            "full_body": bool(data.get("full_body", False)),
            "background": data.get("background", "unknown"),
        }
        parts = data.get("parts") or data.get("components") or data.get("regions") or []
        return {"character": character, "parts": [_normalize_part(part) for part in parts], "warnings": data.get("warnings", [])}
    if schema_name in {"full_body_live2d_plan", "face_detail_live2d_plan"}:
        regions = data.get("regions") or data.get("parts") or data.get("components") or []
        return {
            "coordinate_space": data.get("coordinate_space", "image_pixels"),
            "regions": [_normalize_part(part) for part in regions],
            "warnings": data.get("warnings", []),
        }
    return data


def _normalize_part(part: dict[str, Any]) -> dict[str, Any]:
    group = str(part.get("group") or "ROOT").upper()
    group_aliases = {
        "HAIR_B": "HAIR",
        "HAIR": "HAIR",
        "FACE": "HEAD",
        "EYE": "EYES",
        "EYES": "EYES",
        "MOUTH": "MOUTH",
        "CLOTHING": "CLOTHES",
        "CLOTHES": "CLOTHES",
        "ACCESSORY": "ACCESSORIES",
    }
    bbox = part.get("bbox") or part.get("box") or [0, 0, 1, 1]
    return {
        "id": str(part.get("id") or part.get("name") or "part"),
        "group": group_aliases.get(group, group if group in {"ROOT", "HEAD", "EYES", "BROWS", "MOUTH", "HAIR", "BODY", "CLOTHES", "ACCESSORIES", "EFFECTS"} else "ROOT"),
        "bbox": [int(value) for value in bbox[:4]],
        "depth": int(part.get("depth", 50) or 50),
        "deformable": bool(part.get("deformable", True)),
        "occluded": bool(part.get("occluded", False)),
        "needs_reconstruction": bool(part.get("needs_reconstruction", False)),
        "confidence": float(part.get("confidence", 0.5) or 0.5),
        "reasoning": str(part.get("reasoning") or part.get("short_reasoning") or ""),
    }
