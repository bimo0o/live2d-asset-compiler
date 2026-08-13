from __future__ import annotations

from PIL import Image

from src.utils.images import resize_lanczos


class FaithfulUpscaler:
    def __init__(self, scale: int = 2) -> None:
        self.scale = scale

    def upscale(self, image: Image.Image) -> Image.Image:
        return resize_lanczos(image, self.scale)

