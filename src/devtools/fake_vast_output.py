from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Create fake Vast outputs for local resume smoke tests.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    source = Image.open(run_dir / "upscale" / "master_2x.png").convert("RGBA")
    targets_path = run_dir / "analysis" / "material_targets.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8")) if targets_path.exists() else {"parts": []}
    parts = targets.get("parts") or _fallback_parts()

    raw_dir = run_dir / "decomposition" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    width, height = source.size
    for index, part in enumerate(parts[:8]):
        mask = Image.new("L", source.size, 0)
        left, top, right, bottom = _box_for(index, width, height)
        mask.paste(255, (left, top, right, bottom))
        layer = source.copy()
        layer.putalpha(mask)
        part_id = part["id"]
        layer_path = raw_dir / f"layer_{index:02d}_{part_id}.png"
        layer.save(layer_path)
        metadata.append(
            {
                "id": part_id,
                "name": _title(part_id),
                "group": part.get("group", "ROOT"),
                "z_index": int(part.get("depth", 10 + index * 10)),
                "layer": str(layer_path.relative_to(run_dir)),
                "mask": "",
                "confidence": float(part.get("confidence", 0.5)),
                "warnings": ["fake Vast output for local smoke testing"],
            }
        )
    (run_dir / "decomposition" / "layers.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


def _fallback_parts() -> list[dict]:
    return [{"id": f"part_{index:02d}", "group": "ROOT", "depth": 10 + index * 10, "confidence": 0.5} for index in range(8)]


def _box_for(index: int, width: int, height: int) -> tuple[int, int, int, int]:
    boxes = [
        (0.0, 0.0, 1.0, 0.35),
        (0.1, 0.28, 0.9, 0.65),
        (0.2, 0.45, 0.8, 1.0),
        (0.25, 0.08, 0.75, 0.38),
        (0.25, 0.18, 0.48, 0.32),
        (0.52, 0.18, 0.75, 0.32),
        (0.35, 0.30, 0.65, 0.42),
        (0.0, 0.0, 1.0, 1.0),
    ]
    left, top, right, bottom = boxes[index % len(boxes)]
    return int(left * width), int(top * height), int(right * width), int(bottom * height)


def _title(part_id: str) -> str:
    return " ".join(piece.capitalize() for piece in part_id.split("_") if piece)


if __name__ == "__main__":
    raise SystemExit(main())

