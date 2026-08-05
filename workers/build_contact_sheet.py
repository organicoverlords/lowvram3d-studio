"""Build a labelled contact sheet from a Blender diagnostic render manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.render_dir / "render_manifest.json").read_text(encoding="utf-8"))
    names = manifest["view_order"]
    images = [Image.open(args.render_dir / name).convert("RGB") for name in names]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images) + 34
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 3 + 50), (24, 26, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{manifest['label']} | {manifest['input_sha256'][:16]} | {manifest['blender_version']}", fill=(240, 240, 240))
    for index, (name, image) in enumerate(zip(names, images, strict=True)):
        row, column = divmod(index, 3)
        x = column * tile_w
        y = 50 + row * tile_h
        sheet.paste(ImageOps.pad(image, (tile_w, tile_h - 34), color=(24, 26, 30)), (x, y))
        draw.text((x + 8, y + tile_h - 28), Path(name).stem, fill=(240, 240, 240))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
