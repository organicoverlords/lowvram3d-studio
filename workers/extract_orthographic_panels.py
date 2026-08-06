"""Cut FRONT / SIDE / BACK panels out of a reference sheet at a COMMON scale.

The first multiview boat run conditioned `hunyuan3d-dit-v2-mv-turbo` on three
panels that had each been tight-cropped to their own ink and squared to 512
independently. That is the obvious thing to do and it is wrong. The FRONT panel
is 280x461 on the sheet and the SIDE panel is 551x460 -- the same object at the
same height -- but squaring each to its own bounding box makes the wide one
shrink to fit and the narrow one grow. Measured against the sheet, the three
panels came out at 1.111, 0.937 and 1.256 pixels per sheet pixel, a spread of
1.34, and their baselines landed on rows 512, 471 and 511 of a 512 canvas. The
model was handed three mutually inconsistent framings of one object and
reconciled them into a terraced average -- sheer gone, bow rake gone, paddle
wheel gone.

Orthographic elevations of one object share exactly one dimension: **height**.
Not width, not diagonal, not bounding-box area. So the panels are normalised on
ink height to a shared pixel-per-unit, padded rather than rescaled to fill, and
laid on a common baseline. A deck level then lands at the same row in FRONT and
in SIDE, which is the only property the multiview conditioner can actually use.

`workers/split_reference_sheet.py` already finds and crops the panels on a
plate, and it is the better tool for that half of the job -- it merges the
disconnected strokes of one drawing, which plain component labelling does not.
What it does not do, and what cost the first multiview run, is decide the scale
across panels. That is all this file adds. Regions come from the operator or
from that splitter's report; the tool snaps to the drawing inside each one, so
captions like "FRONT" and "SIDE" fall out for free.

    py workers/extract_orthographic_panels.py --sheet sheet.png --list

    py workers/extract_orthographic_panels.py --sheet sheet.png --out-dir cond \
       --panel front=661,13,941,474 --panel left=967,13,1518,473 \
       --panel back=47,479,337,886

View names are the tags `MVImageProcessorV2` expects: front, left, back, right,
where left means the object turned 90 degrees clockwise seen from above. Which
physical elevation a given SIDE panel is remains the operator's call; the sheet
does not say.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage as ndi

#: Anything darker than this on any channel is drawing rather than paper. The
#: sheets in this project are white-on-white JPEG-ish exports whose "white" sits
#: around 250-254, so the threshold has to be loose enough to survive that and
#: tight enough not to swallow the page.
INK_THRESHOLD = 235

#: Components below this fraction of the region's largest are captions, stray
#: registration marks and JPEG speckle, not part of the subject.
MIN_COMPONENT_FRACTION = 0.02

#: A row counts toward the subject's BODY only if it is at least this wide
#: relative to the panel's widest row. Masts, flags, spires, antennae and raised
#: staffs are a few pixels across and reach far above everything else, and no
#: two elevations of one object draw them at the same proportion. Normalising on
#: total ink height therefore matches flagpoles rather than hulls: on the Lucky
#: Drown sheet it squashed the side elevation's length by about 45 percent and
#: turned a long low riverboat into a wedding cake -- length:height 1.12 against
#: the 1.74 the target has. Measuring the body instead is what makes the shared
#: scale mean anything.
BODY_WIDTH_FRACTION = 0.15

DEFAULT_CANVAS = 512
DEFAULT_MARGIN = 0.06

VIEW_TAGS = ("front", "left", "back", "right")


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=2) < INK_THRESHOLD


def clean_region(mask: np.ndarray) -> np.ndarray:
    """Drop everything that is not the subject: captions, speckle, rules."""
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return mask
    sizes = np.asarray(ndi.sum(mask, labels, range(1, count + 1)))
    keep = np.nonzero(sizes >= sizes.max() * MIN_COMPONENT_FRACTION)[0] + 1
    return np.isin(labels, keep)


def body_height(mask: np.ndarray) -> int:
    """Vertical extent of the subject's body, ignoring thin protrusions."""
    per_row = mask.sum(axis=1)
    body = np.nonzero(per_row >= per_row.max() * BODY_WIDTH_FRACTION)[0]
    if body.size < 2:
        return int(mask.shape[0])
    return int(body[-1] - body[0] + 1)


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.nonzero(mask.any(axis=1))[0]
    columns = np.nonzero(mask.any(axis=0))[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def list_components(sheet: Path, limit: int) -> list[dict]:
    rgb = np.asarray(Image.open(sheet).convert("RGB"))
    mask = ink_mask(rgb)
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = np.asarray(ndi.sum(mask, labels, range(1, count + 1)))
    found = []
    for rank, index in enumerate(np.argsort(sizes)[::-1][:limit]):
        rows, columns = ndi.find_objects(labels == index + 1)[0]
        found.append({
            "rank": rank,
            "pixels": int(sizes[index]),
            "box": [int(columns.start), int(rows.start),
                    int(columns.stop), int(rows.stop)],
            "width": int(columns.stop - columns.start),
            "height": int(rows.stop - rows.start),
        })
    return found


def run(sheet: Path, regions: dict, out_dir: Path,
        canvas: int = DEFAULT_CANVAS, margin: float = DEFAULT_MARGIN,
        mirrors: dict | None = None) -> dict:
    rgb = np.asarray(Image.open(sheet).convert("RGB"))

    # Snap each supplied region to the drawing inside it, and record the true
    # ink extent. Everything downstream is derived from these six numbers.
    panels = {}
    for tag, (x0, y0, x1, y1) in regions.items():
        window = clean_region(ink_mask(rgb[y0:y1, x0:x1]))
        if not window.any():
            raise SystemExit(f"no ink found in the region given for '{tag}'")
        wx0, wy0, wx1, wy1 = tight_box(window)
        panels[tag] = {
            "box": [x0 + wx0, y0 + wy0, x0 + wx1, y0 + wy1],
            "mask": window[wy0:wy1, wx0:wx1],
            "rgb": rgb[y0 + wy0:y0 + wy1, x0 + wx0:x0 + wx1],
        }

    available = canvas * (1.0 - 2.0 * margin)

    # One shared BODY height for every view, chosen so that no panel -- masts
    # and all -- overflows the canvas once that scale is applied. This is the
    # whole point of the file: the scale is a property of the SET, never of the
    # individual panel.
    for panel in panels.values():
        panel["body"] = body_height(panel["mask"])
    shared_body = min(
        min(available * p["body"] / p["mask"].shape[0],          # full height fits
            available * p["body"] / p["mask"].shape[1])          # full width fits
        for p in panels.values())
    baseline = int(round(canvas - (canvas - available) / 2.0))

    receipt_panels = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, panel in panels.items():
        height, width = panel["mask"].shape
        scale = shared_body / panel["body"]
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

        crop = Image.fromarray(panel["rgb"]).resize(target, Image.LANCZOS)
        alpha = Image.fromarray((panel["mask"] * 255).astype(np.uint8)).resize(
            target, Image.LANCZOS)
        crop.putalpha(alpha)

        sheet_out = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
        sheet_out.paste(crop, ((canvas - target[0]) // 2, baseline - target[1]), crop)
        path = out_dir / f"{tag}.png"
        sheet_out.save(path)

        receipt_panels[tag] = {
            "path": str(path),
            "sheet_box": panel["box"],
            "sheet_ink_size": [int(width), int(height)],
            "sheet_body_height": int(panel["body"]),
            "scale": round(float(scale), 5),
            "placed_size": list(target),
        }

    # A laterally symmetric subject has a right elevation that is exactly the
    # mirror of its left, so for a hull, a fuselage or a facade the fourth view
    # is free and costs no new authored art. It is opt-in because asserting
    # that symmetry on a subject that lacks it is how a pipeline invents
    # geometry: the mirror is a claim about the object, not about the image.
    for target, source in (mirrors or {}).items():
        if source not in receipt_panels:
            raise SystemExit(f"cannot mirror '{source}' into '{target}': not extracted")
        mirrored = Image.open(receipt_panels[source]["path"]).transpose(
            Image.FLIP_LEFT_RIGHT)
        path = out_dir / f"{target}.png"
        mirrored.save(path)
        receipt_panels[target] = {
            **receipt_panels[source], "path": str(path),
            "mirrored_from": source,
        }

    scales = [p["scale"] for p in receipt_panels.values()]
    return {
        "schema": "lowvram3d_orthographic_panels_v1",
        "sheet": str(sheet),
        "canvas": canvas,
        "margin": margin,
        "shared_body_height_px": round(float(shared_body), 2),
        "baseline_row": baseline,
        # The number the previous run got wrong. At a common scale the panels
        # differ only by how far the sheet's own draughting drifted; the old
        # per-panel squaring produced ratios near 2.
        "scale_spread": round(max(scales) / min(scales), 4),
        "panels": receipt_panels,
    }


def parse_panel(argument: str) -> tuple[str, tuple[int, int, int, int]]:
    tag, _, box = argument.partition("=")
    tag = tag.strip().lower()
    if tag not in VIEW_TAGS:
        raise SystemExit(f"unknown view tag '{tag}'; expected one of {VIEW_TAGS}")
    parts = [int(v) for v in box.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--panel {tag} needs x0,y0,x1,y1")
    return tag, tuple(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--panel", action="append", default=[],
                        help="tag=x0,y0,x1,y1 approximate region on the sheet")
    parser.add_argument("--mirror", action="append", default=[],
                        help="tag=source, e.g. right=left for a symmetric subject")
    parser.add_argument("--canvas", type=int, default=DEFAULT_CANVAS)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--list", action="store_true",
                        help="print the largest ink components and exit")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(list_components(args.sheet, args.limit), indent=2))
        return 0
    if not args.panel or args.out_dir is None:
        raise SystemExit("need --out-dir and at least one --panel (or --list)")

    regions = dict(parse_panel(p) for p in args.panel)
    mirrors = {}
    for spec in args.mirror:
        target, _, source = spec.partition("=")
        target, source = target.strip().lower(), source.strip().lower()
        if target not in VIEW_TAGS or source not in VIEW_TAGS:
            raise SystemExit(f"--mirror needs two view tags from {VIEW_TAGS}")
        mirrors[target] = source
    result = run(args.sheet, regions, args.out_dir, args.canvas, args.margin,
                 mirrors=mirrors)
    (args.out_dir / "panels.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
