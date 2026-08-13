from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
        prompt = (
            "You are analyzing a 2D anime character illustration specifically for Live2D "
            "material separation. Return strict JSON only. Identify visually present "
            "components that should potentially become independent deformable materials. "
            "Do not invent missing objects. For each component include id, group, bbox "
            "[x,y,w,h], depth, deformable, occluded, needs_reconstruction, confidence, "
            "and short reasoning. Also include character type, pose, full_body, and background."
        )
        return self._vision_json(image, prompt, _character_analysis_schema())

    def analyze_layer(self, image: Path, layer: Path) -> dict[str, Any]:
        raise NotImplementedError("OpenRouter layer analysis starts in Phase 2.")

    def validate_result(self, images: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("OpenRouter QA starts in Phase 2.")

    def _vision_json(self, image: Path, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
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
                    "name": "live2d_character_analysis",
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

        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return json.loads(content)


def _image_data_url(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
