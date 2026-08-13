from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def open_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def ensure_srgb_rgba(image: Image.Image) -> Image.Image:
    return image.convert("RGBA")


def has_visible_pixels(image: Image.Image) -> bool:
    alpha = image.getchannel("A")
    return alpha.getbbox() is not None


def alpha_bbox(image: Image.Image) -> list[int]:
    bbox = image.getchannel("A").getbbox()
    if not bbox:
        return [0, 0, 0, 0]
    left, top, right, bottom = bbox
    return [left, top, right - left, bottom - top]


def checkerboard(size: tuple[int, int], tile: int = 24) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "#ffffff")
    draw = ImageDraw.Draw(image)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = "#d8d8d8" if (x // tile + y // tile) % 2 else "#f4f4f4"
            draw.rectangle((x, y, min(x + tile, width), min(y + tile, height)), fill=color)
    return image


def composite_on(image: Image.Image, background: str | Image.Image) -> Image.Image:
    if isinstance(background, str):
        base = Image.new("RGBA", image.size, background)
    else:
        base = background.convert("RGBA")
    base.alpha_composite(image)
    return base.convert("RGB")


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")


def feather_mask(mask: Image.Image, radius: float = 0.4) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def clean_alpha(alpha: Image.Image) -> Image.Image:
    mask = alpha.point(lambda value: 255 if value > 12 else 0)
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return feather_mask(mask, 0.35)


def color_distance_mask(image: Image.Image, target: tuple[int, int, int], tolerance: int) -> Image.Image:
    rgb = image.convert("RGB")
    r_target, g_target, b_target = target
    mask = Image.new("L", image.size, 0)
    pixels = rgb.load()
    out = mask.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            distance = abs(r - r_target) + abs(g - g_target) + abs(b - b_target)
            if distance <= tolerance:
                out[x, y] = 255
    return mask


def subtract_masks(base: Image.Image, subtract: Image.Image) -> Image.Image:
    return ImageChops.subtract(base.convert("L"), subtract.convert("L"))


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    layer = image.copy().convert("RGBA")
    layer.putalpha(mask.convert("L"))
    return layer


def resize_lanczos(image: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return image.copy()
    width, height = image.size
    return image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)


def difference_map(a: Image.Image, b: Image.Image) -> Image.Image:
    left = a.convert("RGBA")
    right = b.convert("RGBA").resize(left.size)
    diff = ImageChops.difference(left, right)
    alpha = diff.convert("L")
    enhanced = ImageOps.autocontrast(alpha)
    return Image.merge("RGBA", (enhanced, Image.new("L", a.size, 0), Image.new("L", a.size, 0), Image.new("L", a.size, 255)))

