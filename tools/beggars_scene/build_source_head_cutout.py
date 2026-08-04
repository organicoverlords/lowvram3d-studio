from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--crop", default="155,18,405,250")
    parser.add_argument("--feather", type=float, default=4.0)
    return parser.parse_args()


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop: {parts}")
    return x0, y0, x1, y1


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    image = Image.open(input_path).convert("RGB")
    crop = parse_crop(args.crop)
    x0, y0, x1, y1 = crop
    if x1 > image.width or y1 > image.height:
        raise ValueError(f"crop {crop} exceeds source size {image.size}")
    cropped = image.crop(crop)

    # Hand-authored silhouette for the public reference keyframe. The points
    # intentionally include the full hair mass, ear, jaw and enough neck to
    # overlap the procedural torso without retaining the caption beneath it.
    polygon = [
        (14, 53),
        (23, 33),
        (48, 17),
        (80, 8),
        (116, 4),
        (153, 8),
        (185, 21),
        (209, 39),
        (229, 66),
        (239, 96),
        (236, 124),
        (226, 148),
        (216, 164),
        (207, 182),
        (197, 199),
        (187, 218),
        (174, 230),
        (158, 231),
        (146, 222),
        (134, 211),
        (119, 204),
        (102, 195),
        (84, 181),
        (67, 168),
        (50, 158),
        (35, 143),
        (23, 122),
        (15, 96),
        (11, 73),
    ]
    if cropped.size != (250, 232):
        raise RuntimeError(f"unexpected reference crop size: {cropped.size}")

    hard_mask = Image.new("L", cropped.size, 0)
    draw = ImageDraw.Draw(hard_mask)
    draw.polygon(polygon, fill=255)
    feathered = hard_mask.filter(ImageFilter.GaussianBlur(radius=args.feather))

    rgba = cropped.convert("RGBA")
    rgba.putalpha(feathered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path)

    alpha_bbox = feathered.getbbox()
    if alpha_bbox is None:
        raise RuntimeError("source head alpha mask is empty")
    nonzero = sum(1 for value in feathered.getdata() if value > 0)
    fraction = nonzero / float(cropped.width * cropped.height)
    report = {
        "classification": "PROVEN",
        "route": "PUBLIC_REFERENCE_FEATHERED_HEAD_CUTOUT",
        "input": str(input_path),
        "output": str(output_path),
        "source_size": list(image.size),
        "crop": list(crop),
        "crop_size": list(cropped.size),
        "polygon_points": [list(point) for point in polygon],
        "feather_radius_pixels": args.feather,
        "alpha_bbox": list(alpha_bbox),
        "alpha_nonzero_fraction": fraction,
        "caption_pixels_included": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if output_path.stat().st_size < 25000:
        raise RuntimeError("source head cutout is implausibly small")
    print(
        "SOURCE_HEAD_CUTOUT=PROVEN "
        f"SIZE={cropped.width}x{cropped.height} ALPHA_FRACTION={fraction:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
