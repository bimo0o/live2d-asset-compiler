from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops, ImageFilter, ImageOps

from src.utils.images import apply_mask, clean_alpha, color_distance_mask, subtract_masks


@dataclass(frozen=True)
class DecomposedLayer:
    id: str
    name: str
    group: str
    z_index: int
    image: Image.Image
    mask: Image.Image
    confidence: float
    warnings: list[str]


class QwenLayeredClient:
    """Interface for Qwen Image Layered.

    Phase 1 ships a local deterministic fallback. The public method is kept
    narrow so a real Qwen implementation can replace the fallback later.
    """

    def __init__(self, provider: str, model: str, layer_count: int) -> None:
        self.provider = provider
        self.model = model
        self.layer_count = layer_count

    def decompose(self, image: Image.Image) -> list[DecomposedLayer]:
        if self.provider != "local_fallback":
            raise NotImplementedError(f"Provider is not wired yet: {self.provider}")
        return LocalFallbackDecomposer(self.layer_count).decompose(image)


class LocalFallbackDecomposer:
    def __init__(self, layer_count: int) -> None:
        self.layer_count = layer_count

    def decompose(self, image: Image.Image) -> list[DecomposedLayer]:
        rgba = image.convert("RGBA")
        character_mask = self._character_mask(rgba)
        width, height = rgba.size

        regions = [
            ("hair_back", "Hair Back", "HAIR", 20, (0.00, 0.00, 1.00, 0.58), 0.62),
            ("body", "Body", "BODY", 40, (0.18, 0.42, 0.82, 1.00), 0.66),
            ("clothes", "Clothes", "CLOTHES", 55, (0.12, 0.48, 0.88, 1.00), 0.60),
            ("head_face", "Face", "HEAD", 70, (0.22, 0.12, 0.78, 0.55), 0.68),
            ("eye_l", "Eye L", "EYES", 90, (0.27, 0.24, 0.48, 0.39), 0.58),
            ("eye_r", "Eye R", "EYES", 91, (0.52, 0.24, 0.73, 0.39), 0.58),
            ("mouth", "Mouth", "MOUTH", 96, (0.38, 0.36, 0.62, 0.48), 0.54),
            ("hair_front", "Hair Front", "HAIR", 120, (0.06, 0.00, 0.94, 0.50), 0.62),
        ]
        selected = regions[: max(3, min(self.layer_count, len(regions)))]

        layers: list[DecomposedLayer] = []
        used_mask = Image.new("L", rgba.size, 0)
        for index, (part_id, name, group, z_index, relative_box, confidence) in enumerate(selected):
            region_mask = self._relative_box_mask((width, height), relative_box)
            mask = ImageChops.multiply(character_mask, region_mask)
            if group == "EYES":
                dark = self._dark_mask(rgba)
                mask = ImageChops.multiply(mask, dark.filter(ImageFilter.MaxFilter(9)))
            elif group == "MOUTH":
                dark = self._dark_mask(rgba)
                mask = ImageChops.multiply(mask, dark.filter(ImageFilter.MaxFilter(7)))
            elif part_id == "hair_front":
                dark = self._dark_mask(rgba)
                mask = ImageChops.lighter(ImageChops.multiply(mask, dark.filter(ImageFilter.MaxFilter(15))), mask.point(lambda v: int(v * 0.45)))
            elif part_id == "clothes":
                non_skin = self._non_skin_mask(rgba)
                mask = ImageChops.multiply(mask, non_skin)

            mask = clean_alpha(mask)
            if index < len(selected) - 1 and group not in {"EYES", "MOUTH", "HAIR"}:
                mask = subtract_masks(mask, used_mask.point(lambda value: 128 if value > 12 else 0))
            used_mask = ImageChops.lighter(used_mask, mask)
            layer = apply_mask(rgba, mask)
            warnings = []
            if not mask.getbbox():
                warnings.append("empty fallback layer")
            layers.append(DecomposedLayer(part_id, name, group, z_index, layer, mask, confidence, warnings))

        remaining = subtract_masks(character_mask, used_mask.point(lambda value: 255 if value > 12 else 0))
        if remaining.getbbox():
            layers.insert(
                0,
                DecomposedLayer(
                    "base_remainder",
                    "Base Remainder",
                    "ROOT",
                    10,
                    apply_mask(rgba, remaining),
                    remaining,
                    0.5,
                    ["fallback layer for pixels not covered by heuristic regions"],
                ),
            )
        return layers

    @staticmethod
    def _character_mask(image: Image.Image) -> Image.Image:
        alpha = image.getchannel("A")
        if alpha.getbbox():
            return clean_alpha(alpha)
        white_mask = color_distance_mask(image, (255, 255, 255), 72)
        return ImageOps.invert(white_mask).filter(ImageFilter.MaxFilter(3))

    @staticmethod
    def _relative_box_mask(size: tuple[int, int], box: tuple[float, float, float, float]) -> Image.Image:
        width, height = size
        left = int(width * box[0])
        top = int(height * box[1])
        right = int(width * box[2])
        bottom = int(height * box[3])
        mask = Image.new("L", size, 0)
        mask.paste(255, (left, top, right, bottom))
        return mask

    @staticmethod
    def _dark_mask(image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image.convert("RGB"))
        return gray.point(lambda value: 255 if value < 120 else 0)

    @staticmethod
    def _non_skin_mask(image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        mask = Image.new("L", image.size, 0)
        pixels = rgb.load()
        out = mask.load()
        width, height = image.size
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                skin_like = r > 135 and g > 80 and b > 60 and r > b and r >= g
                out[x, y] = 0 if skin_like else 255
        return mask

