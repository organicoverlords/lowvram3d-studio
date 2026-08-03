"""Create a labeled visual evidence sheet without changing scene assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
REPO_PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def main() -> None:
    names = ["source_rgb", "blender_source", "blender_left", "blender_right", "blender_forward", "blender_elevated", "blender_rear", "depth_vis", "normal_vis"]
    cells = []
    for name in names:
        path = ROOT / (name + ".png")
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 320))
        cell = Image.new("RGB", (500, 360), (22, 28, 36))
        cell.paste(image, ((500 - image.width) // 2, 28))
        ImageDraw.Draw(cell).text((12, 8), name, fill=(240, 240, 240))
        cells.append(cell)
    sheet = Image.new("RGB", (1500, ((len(cells) + 2) // 3) * 360), (12, 16, 22))
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 3) * 500, (index // 3) * 360))
    external = ROOT / "contact_sheet.png"
    repo = REPO_PROOF / "contact_sheet.png"
    sheet.save(external)
    repo.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(repo)


if __name__ == "__main__":
    main()
