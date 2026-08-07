"""Reflow a one-row view sheet into a grid, and drop the empty margins.

The renderer lays its views out in a single row. At seven views that sheet is
4340x646, and anything displaying it at 2000px wide shrinks each tile to about
285px -- small enough that I twice claimed a defect was fixed from a thumbnail
when zooming in showed it was not.

Two changes fix it without re-rendering. Reflowing to two rows roughly doubles
the displayed tile size for the same pixels. And each tile is mostly backdrop,
because the camera frames the whole object including its widest view, so
cropping every tile to the union of the subject bounding boxes recovers the rest.
The union rather than per-tile boxes keeps the scale honest across views.

    py workers/reflow_view_sheet.py --sheet views.png --views 7 --columns 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

#: Rows of the sheet occupied by the renderer's filename/view label strip.
LABEL_FRACTION = 0.045
#: Padding kept around the union subject box, as a fraction of its larger side.
MARGIN = 0.04


def subject_box(tile: np.ndarray) -> tuple[int, int, int, int] | None:
    flat = tile.reshape(-1, tile.shape[-1])[:, 0]
    backdrop = np.bincount(flat.astype(int)).argmax()
    mask = np.abs(tile.astype(int) - backdrop).sum(-1) > 18
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def reflow(sheet: Path, views: int, columns: int, out: Path) -> dict:
    image = Image.open(sheet).convert("RGB")
    width, height = image.size
    tile_width = width // views
    label = int(height * LABEL_FRACTION)

    tiles = [image.crop((i * tile_width, label, (i + 1) * tile_width, height))
             for i in range(views)]

    boxes = [b for b in (subject_box(np.asarray(t)) for t in tiles) if b]
    if not boxes:
        raise SystemExit("NO_SUBJECT_FOUND")
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad = int(max(x1 - x0, y1 - y0) * MARGIN)
    crop = (max(0, x0 - pad), max(0, y0 - pad),
            min(tiles[0].size[0], x1 + pad), min(tiles[0].size[1], y1 + pad))

    cropped = [t.crop(crop) for t in tiles]
    cw, ch = cropped[0].size
    rows = (views + columns - 1) // columns
    backdrop = tuple(int(v) for v in np.asarray(cropped[0])[0, 0])
    grid = Image.new("RGB", (cw * columns, ch * rows), backdrop)
    for i, tile in enumerate(cropped):
        grid.paste(tile, ((i % columns) * cw, (i // columns) * ch))
    grid.save(out)
    return {
        "schema": "lowvram3d_view_reflow_v1",
        "sheet_in": str(sheet),
        "sheet_out": str(out),
        "views": views,
        "grid": [rows, columns],
        "tile_before": [tile_width, height],
        "tile_after": [cw, ch],
        "size_out": list(grid.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    out = args.out or args.sheet.with_name(args.sheet.stem + "_grid.png")
    import json
    print(json.dumps(reflow(args.sheet, args.views, args.columns, out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
