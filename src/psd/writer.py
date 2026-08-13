from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PsdLayer:
    name: str
    image: Image.Image


def write_psd(path: Path, size: tuple[int, int], layers: list[PsdLayer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    flattened = Image.new("RGBA", size, (0, 0, 0, 0))
    for layer in layers:
        flattened.alpha_composite(layer.image.convert("RGBA"))

    with path.open("wb") as file:
        file.write(_header(width, height, channels=4))
        file.write(struct.pack(">I", 0))
        file.write(struct.pack(">I", 0))
        layer_info = _layer_info(size, layers)
        file.write(struct.pack(">I", len(layer_info)))
        file.write(layer_info)
        file.write(_image_data(flattened))


def _header(width: int, height: int, channels: int) -> bytes:
    return b"8BPS" + struct.pack(">H6sHIIHH", 1, b"\0" * 6, channels, height, width, 8, 3)


def _layer_info(size: tuple[int, int], layers: list[PsdLayer]) -> bytes:
    records = bytearray()
    pixel_data = bytearray()
    records.extend(struct.pack(">h", len(layers)))
    for layer in layers:
        image = layer.image.convert("RGBA")
        bbox = image.getchannel("A").getbbox() or (0, 0, 0, 0)
        left, top, right, bottom = bbox
        cropped = image.crop(bbox) if bbox != (0, 0, 0, 0) else Image.new("RGBA", (0, 0), (0, 0, 0, 0))
        channel_payloads = _layer_channel_payloads(cropped)
        records.extend(struct.pack(">llllH", top, left, bottom, right, 4))
        for channel_id, payload in channel_payloads:
            records.extend(struct.pack(">hI", channel_id, len(payload)))
        records.extend(b"8BIM")
        records.extend(b"norm")
        records.extend(bytes([255, 0, 0, 0]))
        extra = _layer_extra(layer.name)
        records.extend(struct.pack(">I", len(extra)))
        records.extend(extra)
        for _channel_id, payload in channel_payloads:
            pixel_data.extend(payload)
    payload = records + pixel_data
    return struct.pack(">I", len(payload)) + payload + struct.pack(">I", 0)


def _layer_extra(name: str) -> bytes:
    mask_data = struct.pack(">I", 0)
    blending_ranges = struct.pack(">I", 0)
    encoded = name.encode("ascii", errors="replace")[:255]
    name_data = bytes([len(encoded)]) + encoded
    while len(name_data) % 4:
        name_data += b"\0"
    return mask_data + blending_ranges + name_data


def _layer_channel_payloads(image: Image.Image) -> list[tuple[int, bytes]]:
    if image.size == (0, 0):
        empty = struct.pack(">H", 0)
        return [(0, empty), (1, empty), (2, empty), (-1, empty)]
    r, g, b, a = image.split()
    return [
        (0, _raw_channel(r)),
        (1, _raw_channel(g)),
        (2, _raw_channel(b)),
        (-1, _raw_channel(a)),
    ]


def _raw_channel(channel: Image.Image) -> bytes:
    return struct.pack(">H", 0) + channel.tobytes()


def _image_data(image: Image.Image) -> bytes:
    r, g, b, a = image.convert("RGBA").split()
    return struct.pack(">H", 0) + r.tobytes() + g.tobytes() + b.tobytes() + a.tobytes()


def is_valid_psd(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 64:
        return False
    with path.open("rb") as file:
        return file.read(4) == b"8BPS"
