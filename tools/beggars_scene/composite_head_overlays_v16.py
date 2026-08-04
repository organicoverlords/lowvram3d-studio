from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-plate", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--overlay-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected", default="scale_200")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    body_path = Path(args.body_plate).resolve()
    overlay_dir = Path(args.overlay_dir).resolve()
    report_path = Path(args.overlay_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (body_path, overlay_dir, report_path):
        if not required.exists():
            raise SystemExit(f"V16 compositing input missing: {required}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    base = Image.open(body_path).convert("RGBA")
    variants: list[dict] = []
    selected_path: Path | None = None
    for variant in report["variants"]:
        label = str(variant["label"])
        overlay_path = overlay_dir / str(variant["file"])
        overlay = Image.open(overlay_path).convert("RGBA")
        if overlay.size != base.size:
            raise RuntimeError(
                f"Overlay size mismatch for {label}: {overlay.size} != {base.size}"
            )
        result = Image.alpha_composite(base, overlay)
        output_path = output_dir / f"external_composite_{label}.png"
        result.save(output_path)
        if output_path.stat().st_size < 50000:
            raise RuntimeError(f"V16 composite too small: {output_path}")
        variants.append({**variant, "render": output_path.name, "bytes": output_path.stat().st_size})
        if label == args.selected:
            selected_path = output_path

    if selected_path is None:
        raise RuntimeError(f"Selected V16 variant was not rendered: {args.selected}")
    hero_path = output_dir / "hero_external_composite.png"
    hero_path.write_bytes(selected_path.read_bytes())

    source_path = output_dir / "projected_keyframe_031.png"
    comparison_items: list[tuple[Path, str]] = []
    if source_path.is_file():
        comparison_items.append((source_path, "PUBLIC REFERENCE"))
    comparison_items.append((body_path, "BLENDER BODY PLATE"))
    for variant in variants:
        comparison_items.append((output_dir / variant["render"], f"V16 {variant['label']}"))
    comparison_items.append((hero_path, "V16 SELECTED"))

    tiles: list[Image.Image] = []
    for path, label in comparison_items:
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

    visual_report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "machine_status": "PROVEN",
        "visual_status": "NOT_PROVEN",
        "route": "BLENDER_BODY_PLATE_PLUS_PIL_RGBA_HEAD_V16",
        "selected_label": args.selected,
        "body_plate": body_path.name,
        "hero": hero_path.name,
        "contact_sheet": sheet_path.name,
        "variants": variants,
    }
    (output_dir / "v16_visual_report.json").write_text(
        json.dumps(visual_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "BEGGARS_FACEVERSE_V16_EXTERNAL_COMPOSITE=PROVEN "
        f"VARIANTS={len(variants)} SELECTED={args.selected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
