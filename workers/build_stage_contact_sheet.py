"""One image showing what each meaningful pipeline stage actually produced.

The evidence directory holds dozens of PNGs, most of which nobody should look
at -- a mask per region means a picture of the sky. What matters is the chain
for one asset: the photograph, the region it was cut from, the pixels the
generator was conditioned on, the geometry that came back, and the appearance
finally applied. Five pictures, in order, with the numbers that qualify them.

This exists because every serious defect this project found late was visible in
an image the whole time and nobody had assembled them side by side. A phantom
building, a "tree line" that was one tree, a texture on the wrong half of the
mesh, and a crop that was 81% padding were all diagnosed in minutes once the
right two pictures were adjacent.

    py -3.12 workers/build_stage_contact_sheet.py --evidence evidence/scenes/x \\
        --asset architecture_house_025 --out stages.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROW_HEIGHT = 260
PAD = 14
LABEL_HEIGHT = 34
BACKGROUND = (238, 238, 238)


def _load(path: Path):
    from PIL import Image

    if path and Path(path).is_file():
        return Image.open(path).convert("RGB")
    return None


def _fit(image, height):
    from PIL import Image

    if image is None:
        return None
    scale = height / image.height
    return image.resize((max(1, int(image.width * scale)), height),
                        Image.LANCZOS)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(evidence: Path, asset: str, source_image: Path | None):
    """The stages worth looking at, in pipeline order, each with its numbers."""
    asset_dir = evidence / "generated_assets" / asset
    manifest = _read_json(evidence / "generated_asset_manifest.json")
    entry = next((a for a in manifest.get("assets", [])
                  if a.get("asset_id") == asset), {})
    # The manifest embeds a copy of the projection receipt taken when the asset
    # was generated. Re-projecting an existing mesh rewrites the standalone
    # receipt and leaves that copy stale, so the sheet reported None for every
    # coverage figure while the real numbers sat in the file next to it. Prefer
    # the standalone receipt and fall back to the embedded copy.
    projection = _read_json(
        asset_dir / f"{asset}_textured.projection.json") or entry.get(
            "texture_projection", {})

    depth = _read_json(evidence / "depth_reconstruction_receipt.json")
    region = next((r for r in depth.get("segmentation", {}).get("regions", [])
                   if r.get("id") == asset), {})

    stages = [
        ("1. source photograph",
         source_image or entry.get("crop_png"),
         f"{Path(source_image).name if source_image else ''}"),
        ("2. semantic regions",
         evidence / "depth" / f"{evidence.parent.name}_segmentation.png",
         "SegFormer ADE20K"),
        ("3. this region's mask",
         evidence / "depth" / "region_masks" / f"{asset}.png",
         f"{region.get('semantic_label', '?')} "
         f"confidence {region.get('confidence', '?')}"),
        ("4. conditioning crop",
         asset_dir / "crop.png",
         f"{entry.get('crop_size_px')} -> {entry.get('conditioning_size_px')} "
         f"square, matte {entry.get('matte', '?')}"),
        ("5. geometry (unlit)",
         asset_dir / "preview.png",
         f"octree {entry.get('octree_resolution')} · "
         f"{entry.get('raw_triangles')} -> {entry.get('triangles')} tris · "
         f"{entry.get('mesh_bodies')} bodies"),
        ("6. appearance",
         asset_dir / "textured_views.png",
         f"observed {projection.get('observed_face_fraction')} · "
         f"mirrored {projection.get('mirrored_face_fraction')} · "
         f"flat {projection.get('flat_filled_face_fraction')}"),
    ]
    return [(title, Path(path) if path else None, note)
            for title, path, note in stages]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--row-height", type=int, default=ROW_HEIGHT)
    args = parser.parse_args(argv)

    from PIL import Image, ImageDraw

    evidence = Path(args.evidence)
    stages = collect(evidence, args.asset,
                     Path(args.source_image) if args.source_image else None)

    rows = []
    for title, path, note in stages:
        panel = _fit(_load(path), args.row_height)
        rows.append((title, panel, note, path))

    present = [r for r in rows if r[1] is not None]
    if not present:
        raise SystemExit(f"no stage images found under {evidence}")

    width = max(r[1].width for r in present) + PAD * 2
    height = sum(r[1].height + LABEL_HEIGHT + PAD for r in present) + PAD
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    y = PAD
    for title, panel, note, _ in present:
        draw.text((PAD, y), title, fill=(0, 0, 0))
        draw.text((PAD, y + 15), note, fill=(90, 90, 90))
        sheet.paste(panel, (PAD, y + LABEL_HEIGHT))
        y += panel.height + LABEL_HEIGHT + PAD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    receipt = {
        "schema_version": "stage_contact_sheet_v1",
        "classification": "PROVEN",
        "asset": args.asset,
        "out": str(out),
        "stages_present": [t for t, p, _, _ in rows if p is not None],
        "stages_missing": [t for t, p, _, _ in rows if p is None],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
