from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


VARIANTS = [
    {
        "label": "bridge185_low",
        "head_scale": 1.85,
        "head_x": 400,
        "head_y": 80,
        "bridge_bbox": [490, 300, 800, 590],
        "bridge_color": [22, 10, 9, 255],
    },
    {
        "label": "bridge185",
        "head_scale": 1.85,
        "head_x": 400,
        "head_y": 70,
        "bridge_bbox": [500, 285, 790, 575],
        "bridge_color": [22, 10, 9, 255],
    },
    {
        "label": "bridge200",
        "head_scale": 2.00,
        "head_x": 380,
        "head_y": 48,
        "bridge_bbox": [495, 290, 810, 585],
        "bridge_color": [22, 10, 9, 255],
    },
    {
        "label": "bridge200_low",
        "head_scale": 2.00,
        "head_x": 380,
        "head_y": 65,
        "bridge_bbox": [495, 310, 810, 600],
        "bridge_color": [22, 10, 9, 255],
    },
    {
        "label": "bridge185_warm",
        "head_scale": 1.85,
        "head_x": 400,
        "head_y": 78,
        "bridge_bbox": [485, 300, 805, 595],
        "bridge_color": [31, 14, 11, 255],
    },
    {
        "label": "bridge185_narrow",
        "head_scale": 1.85,
        "head_x": 400,
        "head_y": 80,
        "bridge_bbox": [515, 300, 775, 585],
        "bridge_color": [20, 9, 8, 255],
    },
]
SELECTED_LABEL = "bridge185_low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-plate", required=True)
    parser.add_argument("--head-cutout", required=True)
    parser.add_argument("--source-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def make_bridge(
    size: tuple[int, int],
    bbox: list[int],
    color: list[int],
) -> Image.Image:
    left, top, right, bottom = bbox
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(
        [
            (left + 70, top),
            (right - 70, top),
            (right, bottom),
            (left, bottom),
        ],
        fill=255,
    )
    draw.ellipse((left + 45, top - 45, right - 45, top + 130), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=18))
    bridge = Image.new("RGBA", size, tuple(color))
    bridge.putalpha(mask)
    return bridge


def compose_variant(
    body: Image.Image,
    head: Image.Image,
    variant: dict,
) -> Image.Image:
    bridge = make_bridge(body.size, variant["bridge_bbox"], variant["bridge_color"])
    result = Image.alpha_composite(body, bridge)
    scale = float(variant["head_scale"])
    resized = head.resize(
        (
            max(1, int(round(head.width * scale))),
            max(1, int(round(head.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    overlay = Image.new("RGBA", body.size, (0, 0, 0, 0))
    overlay.alpha_composite(
        resized,
        (int(variant["head_x"]), int(variant["head_y"])),
    )
    return Image.alpha_composite(result, overlay)


def main() -> int:
    args = parse_args()
    body_path = Path(args.body_plate).resolve()
    head_path = Path(args.head_cutout).resolve()
    reference_path = Path(args.source_reference).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (body_path, head_path, reference_path):
        if not required.is_file():
            raise SystemExit(f"V17 input missing: {required}")

    body = Image.open(body_path).convert("RGBA")
    head = Image.open(head_path).convert("RGBA")
    if head.getchannel("A").getbbox() is None:
        raise RuntimeError("V17 head cutout has no alpha")

    rendered: list[dict] = []
    selected_path: Path | None = None
    for variant in VARIANTS:
        image = compose_variant(body, head, variant)
        output_path = output_dir / f"bridge_composite_{variant['label']}.png"
        image.save(output_path)
        if output_path.stat().st_size < 50000:
            raise RuntimeError(f"V17 composite is implausibly small: {output_path}")
        rendered.append({**variant, "render": output_path.name, "bytes": output_path.stat().st_size})
        if variant["label"] == SELECTED_LABEL:
            selected_path = output_path

    if selected_path is None:
        raise RuntimeError("V17 selected variant was not generated")
    hero_path = output_dir / "hero_bridge_composite.png"
    hero_path.write_bytes(selected_path.read_bytes())

    items: list[tuple[Path, str]] = [
        (reference_path, "PUBLIC REFERENCE"),
        (body_path, "BLENDER BODY PLATE"),
    ]
    items.extend(
        (output_dir / item["render"], f"V17 {item['label']}") for item in rendered
    )
    items.append((hero_path, "V17 SELECTED"))

    tiles: list[Image.Image] = []
    for path, label in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (640, 404), "black")
        tile.paste(image, ((640 - image.width) // 2, 0))
        ImageDraw.Draw(tile).text((12, 374), label, fill="white")
        tiles.append(tile)
    rows = (len(tiles) + 1) // 2
    sheet = Image.new("RGB", (1280, rows * 404), "black")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 640, (index // 2) * 404))
    sheet_path = output_dir / "contact_sheet.png"
    sheet.save(sheet_path, quality=95)

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "machine_status": "PROVEN",
        "visual_status": "NOT_PROVEN",
        "route": "BLENDER_BODY_PLATE_PLUS_FEATHERED_HEAD_AND_SOFT_ROBE_BRIDGE_V17",
        "selected_label": SELECTED_LABEL,
        "hero": hero_path.name,
        "contact_sheet": sheet_path.name,
        "variants": rendered,
    }
    (output_dir / "v17_visual_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "BEGGARS_FACEVERSE_V17_BRIDGE_COMPOSITE=PROVEN "
        f"VARIANTS={len(rendered)} SELECTED={SELECTED_LABEL}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
