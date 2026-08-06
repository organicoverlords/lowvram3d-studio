"""Build an aligned baseline/candidate render comparison sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_manifest(directory: Path) -> dict:
    path = directory / "render_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    order = payload.get("view_order")
    if not isinstance(order, list) or not order:
        raise ValueError(f"invalid render manifest: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Baseline 2048")
    parser.add_argument("--candidate-label", default="Surface-owned candidate")
    args = parser.parse_args()

    baseline_manifest = load_manifest(args.baseline_dir)
    candidate_manifest = load_manifest(args.candidate_dir)
    baseline_order = list(baseline_manifest["view_order"])
    candidate_order = list(candidate_manifest["view_order"])
    if baseline_order != candidate_order:
        raise ValueError("baseline and candidate view manifests differ")

    rows: list[tuple[str, Image.Image, Image.Image]] = []
    for name in baseline_order:
        baseline_path = args.baseline_dir / name
        candidate_path = args.candidate_dir / name
        if not baseline_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(f"comparison input missing: {name}")
        rows.append((Path(name).stem, Image.open(baseline_path).convert("RGB"),
                     Image.open(candidate_path).convert("RGB")))

    tile_width = max(max(left.width, right.width) for _, left, right in rows)
    tile_height = max(max(left.height, right.height) for _, left, right in rows)
    header = 68
    row_label_width = 150
    row_gap = 32
    sheet = Image.new(
        "RGB",
        (row_label_width + tile_width * 2, header + len(rows) * (tile_height + row_gap)),
        (24, 26, 30),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((row_label_width + 10, 18), args.baseline_label, fill=(245, 245, 245))
    draw.text((row_label_width + tile_width + 10, 18), args.candidate_label, fill=(245, 245, 245))

    for index, (label, baseline, candidate) in enumerate(rows):
        y = header + index * (tile_height + row_gap)
        draw.text((10, y + tile_height // 2 - 8), label, fill=(235, 235, 235))
        left = ImageOps.pad(baseline, (tile_width, tile_height), color=(24, 26, 30))
        right = ImageOps.pad(candidate, (tile_width, tile_height), color=(24, 26, 30))
        sheet.paste(left, (row_label_width, y))
        sheet.paste(right, (row_label_width + tile_width, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
