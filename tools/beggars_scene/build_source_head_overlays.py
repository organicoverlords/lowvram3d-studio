from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


VARIANTS = [
    {"label": "scale_185", "scale": 1.85, "x": 400, "y": 70},
    {"label": "scale_200", "scale": 2.00, "x": 380, "y": 48},
    {"label": "scale_215", "scale": 2.15, "x": 360, "y": 28},
    {"label": "left_350", "scale": 2.00, "x": 350, "y": 48},
    {"label": "right_410", "scale": 2.00, "x": 410, "y": 48},
    {"label": "high_030", "scale": 2.00, "x": 380, "y": 30},
    {"label": "low_070", "scale": 2.00, "x": 380, "y": 70},
]
SELECTED_LABEL = "scale_200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cutout_path = Path(args.cutout).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    if not cutout_path.is_file():
        raise FileNotFoundError(cutout_path)
    if args.width <= 0 or args.height <= 0:
        raise ValueError("overlay dimensions must be positive")

    cutout = Image.open(cutout_path).convert("RGBA")
    if cutout.getchannel("A").getbbox() is None:
        raise RuntimeError("source head cutout has no visible alpha")
    output_dir.mkdir(parents=True, exist_ok=True)

    receipts: list[dict[str, object]] = []
    for variant in VARIANTS:
        scale = float(variant["scale"])
        scaled_size = (
            max(1, int(round(cutout.width * scale))),
            max(1, int(round(cutout.height * scale))),
        )
        resized = cutout.resize(scaled_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 0))
        x_value = int(variant["x"])
        y_value = int(variant["y"])
        if x_value < 0 or y_value < 0:
            raise ValueError(f"negative overlay location: {variant}")
        if x_value + resized.width > args.width or y_value + resized.height > args.height:
            raise ValueError(f"overlay exceeds frame: {variant}, size={scaled_size}")
        canvas.alpha_composite(resized, (x_value, y_value))
        output_path = output_dir / f"source_head_overlay_{variant['label']}.png"
        canvas.save(output_path)
        if output_path.stat().st_size < 25000:
            raise RuntimeError(f"overlay is implausibly small: {output_path}")
        receipts.append(
            {
                **variant,
                "file": output_path.name,
                "scaled_size": list(scaled_size),
                "bytes": output_path.stat().st_size,
            }
        )
        print(
            "SOURCE_HEAD_OVERLAY=PROVEN "
            f"LABEL={variant['label']} SCALE={scale:.2f} X={x_value} Y={y_value}"
        )

    selected = next(item for item in receipts if item["label"] == SELECTED_LABEL)
    report = {
        "classification": "PROVEN",
        "route": "FULL_FRAME_RGBA_HEAD_OVERLAY",
        "cutout": str(cutout_path),
        "frame_size": [args.width, args.height],
        "selected_label": SELECTED_LABEL,
        "selected": selected,
        "variants": receipts,
        "caption_pixels_included": False,
        "background_is_transparent": True,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SOURCE_HEAD_OVERLAYS=PROVEN COUNT={len(receipts)} SELECTED={SELECTED_LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
