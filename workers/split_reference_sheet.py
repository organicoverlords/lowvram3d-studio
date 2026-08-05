"""Cut a multi-panel reference sheet into its separate drawings.

A character sheet holds several orthographic elevations on one plate -- front,
side, back, a deck plan, a cutaway, and a strip of component studies. Each is a
separate drawing that happens to share a canvas, and for texturing purposes the
useful ones are worth far more than any generated view: they are real authored
reference for exactly the faces a multiview adapter has to invent.

Panels are found rather than hardcoded. Everything that is not the plate is
labelled, tiny specks are dropped, and the remaining components are merged when
their bounding boxes overlap horizontally and vertically -- a single drawing is
usually several disconnected strokes (a flag above a mast, a detached railing),
so connected components alone would shatter it.

What comes out is a directory of cropped PNGs plus a report of where each came
from. Naming them is a judgement about content and is left to the caller.

    py -3.12 workers/split_reference_sheet.py --sheet SHEET.png --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Fraction of the sheet below which a component is speckle, not a drawing.
MIN_PANEL_AREA_FRACTION = 0.0008

#: Gap in pixels within which two components are considered the same drawing.
MERGE_GAP = 26


def _merge(boxes, gap):
    """Union boxes that are within `gap` of each other, until stable."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                overlap_x = a[0] - gap <= b[2] and b[0] - gap <= a[2]
                overlap_y = a[1] - gap <= b[3] and b[1] - gap <= a[3]
                if overlap_x and overlap_y:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP)
    parser.add_argument("--pad", type=int, default=6)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    sheet = Image.open(args.sheet).convert("RGB")
    rgb = np.asarray(sheet)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    plate = np.median(np.concatenate([
        rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
        rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)]), axis=0)
    ink = np.linalg.norm(rgb.astype(np.float32) - plate, axis=2) > args.tolerance

    labels, count = ndimage.label(ink, structure=np.ones((3, 3)))
    if not count:
        raise SystemExit("NO_PANELS_FOUND")
    areas = ndimage.sum(ink, labels, range(1, count + 1))
    floor = ink.size * MIN_PANEL_AREA_FRACTION
    slices = ndimage.find_objects(labels)
    boxes = [[s[1].start, s[0].start, s[1].stop - 1, s[0].stop - 1]
             for index, s in enumerate(slices) if areas[index] >= floor]
    boxes = _merge(boxes, args.merge_gap)
    # Reading order: top to bottom, then left to right, with a generous row
    # tolerance so panels whose tops differ slightly still count as one row.
    boxes.sort(key=lambda b: (round(b[1] / 120), b[0]))

    panels = []
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        x0 = max(0, x0 - args.pad)
        y0 = max(0, y0 - args.pad)
        x1 = min(rgb.shape[1] - 1, x1 + args.pad)
        y1 = min(rgb.shape[0] - 1, y1 + args.pad)
        crop = sheet.crop((x0, y0, x1 + 1, y1 + 1))
        path = output / f"panel_{index:02d}.png"
        crop.save(path)
        panels.append({
            "index": index,
            "file": path.name,
            "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "size": [int(x1 - x0 + 1), int(y1 - y0 + 1)],
            "ink_fraction": round(float(ink[y0:y1 + 1, x0:x1 + 1].mean()), 4),
        })

    report = {
        "schema_version": "reference_sheet_split_v1",
        "sheet": str(Path(args.sheet).resolve()),
        "output": str(output.resolve()),
        "plate_rgb": [float(v) for v in plate],
        "panels_found": len(panels),
        "panels": panels,
        "naming": "positional only; identifying which panel is which view is a "
                  "judgement about content and is not made here",
    }
    report_path = Path(args.report) if args.report else output / "panels.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
