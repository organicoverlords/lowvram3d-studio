"""Matched detail crops from two high-res clay renders."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "workers")
import json
import tempfile

from PIL import Image, ImageDraw

import render_asset_views as r

SIZE = 1800
MESHES = [("res 512  146k", "evidence/compare/boat/stage6_s1006.glb"),
          ("res 1024 298k", "evidence/compare/boat/stage6_1024_s1006.glb")]
VIEW = sys.argv[1] if len(sys.argv) > 1 else "profile"
# Fractional (x0, y0, x1, y1) boxes in the square tile.
REGIONS = {
    # Fractions of the rendered CONTENT box, not the tile. The subject does
    # not fill the tile and the two meshes do not fill it identically, so tile
    # fractions crop different parts of each boat -- which is how the first
    # attempt cropped empty background.
    "profile": [("paddle wheel", (0.00, 0.45, 0.22, 1.00)),
                ("mid decks + railings", (0.34, 0.20, 0.68, 0.62)),
                ("bow ornament", (0.76, 0.15, 1.00, 0.70))],
    "end_plus": [("arch + fan", (0.10, 0.00, 0.90, 0.40)),
                 ("tier windows", (0.05, 0.35, 0.95, 0.80))],
}[VIEW]

scratch = Path(tempfile.mkdtemp(prefix="zoom-"))
script = scratch / "r.py"
script.write_text(r.SCRIPT, encoding="utf-8")
jobs = [{"mesh": str(Path(p).resolve()), "prefix": str(scratch / f"m{i}"),
         "views": [[VIEW, *r.VIEWS[VIEW]]]} for i, (_, p) in enumerate(MESHES)]
payload = json.dumps({"size": SIZE, "align": True, "half": False,
                      "clay": True, "jobs": jobs})
subprocess.run([str(r.BLENDER), "-b", "--python", str(script), "--", payload],
               capture_output=True, text=True)

CROP = 460
sheet = Image.new("RGB", (CROP * len(REGIONS), (CROP + 26) * 2), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for row, (label, _) in enumerate(MESHES):
    tile = Image.open(scratch / f"m{row}_{VIEW}.png").convert("RGB")
    import numpy as np
    a = np.asarray(tile).astype(int)
    # Background is the flat world grey; content is anything that differs.
    bg = np.bincount(a.reshape(-1, 3)[:, 0], minlength=256).argmax()
    solid = (np.abs(a - a[0, 0]).sum(axis=2) > 10)
    ys, xs = np.nonzero(solid)
    cx0, cx1, cy0, cy1 = xs.min(), xs.max(), ys.min(), ys.max()
    cw, ch = cx1 - cx0, cy1 - cy0
    draw.text((6, row * (CROP + 26) + 8), label, fill=(0, 0, 0))
    for col, (name, (x0, y0, x1, y1)) in enumerate(REGIONS):
        box = (int(cx0 + x0 * cw), int(cy0 + y0 * ch),
               int(cx0 + x1 * cw), int(cy0 + y1 * ch))
        sheet.paste(tile.crop(box).resize((CROP, CROP), Image.LANCZOS),
                    (col * CROP, row * (CROP + 26) + 26))
        if row == 0:
            draw.text((col * CROP + 8, 8), name, fill=(110, 110, 110))
sheet.save(sys.argv[2])
print("WROTE", sys.argv[2])
