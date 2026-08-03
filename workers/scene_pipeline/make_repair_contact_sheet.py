"""Create the bounded Castlegrounds source-mesh comparison sheet."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
OUT = ROOT / "source_mesh_repair_contact_sheet.png"


def main() -> None:
    items = [
        ("source", ROOT / "source_rgb_512.png"),
        ("official MoGe", ROOT / "official_moge_exact_cull_off.png"),
        ("balanced old", ROOT / "balanced_010_exact2_cull_off.png"),
        ("strict winding", ROOT / "strict_005_winding_exact_cull_on.png"),
        ("permissive winding", ROOT / "permissive_020_winding_exact_cull_on.png"),
        ("adaptive conservative", ROOT / "adaptive_conservative_winding_exact_cull_on.png"),
        ("adaptive balanced", ROOT / "adaptive_balanced_winding_exact_cull_on.png"),
        ("adaptive coverage", ROOT / "adaptive_coverage_winding_exact_cull_on.png"),
        ("VITS 768", ROOT / "moge_comparison/vits_768/vits_768_winding_exact_cull_on.png"),
        ("VITB 512", ROOT / "moge_comparison/vitb_512/vitb_512_winding_exact_cull_on.png"),
        ("VITB 640", ROOT / "moge_comparison/vitb_640/vitb_640_winding_exact_cull_on.png"),
        ("missing overlay", ROOT / "blender_missing_pixel_overlay.png"),
    ]
    tile_w, tile_h = 256, 216
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 3), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w, tile_h - 24), Image.Resampling.LANCZOS)
        x = (index % 4) * tile_w
        y = (index // 4) * tile_h
        sheet.paste(image, (x + (tile_w - image.width) // 2, y + 22))
        draw.text((x + 5, y + 4), label, fill=(255, 255, 255))
    sheet.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
