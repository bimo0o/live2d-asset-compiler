from __future__ import annotations

from pathlib import Path

from PIL import Image


def alpha_coverage(image: Image.Image) -> float:
    alpha = image.convert("RGBA").getchannel("A")
    histogram = alpha.histogram()
    visible = sum(histogram[1:])
    return visible / float(image.width * image.height)


def validate_material_image(path: Path, expected_size: tuple[int, int]) -> list[str]:
    warnings: list[str] = []
    image = Image.open(path).convert("RGBA")
    if image.size != expected_size:
        raise ValueError(f"Layer {path} has size {image.size}, expected {expected_size}")
    coverage = alpha_coverage(image)
    if coverage <= 0:
        raise ValueError(f"Layer {path} has no visible pixels")
    if coverage > 0.96:
        warnings.append("alpha coverage is almost full canvas; possible opaque background")
    if coverage < 0.0005:
        warnings.append("alpha coverage is extremely tiny; possible failed layer")
    return warnings

