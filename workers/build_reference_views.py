"""Put authored reference drawings where generated views would have gone.

MV-Adapter invents whatever the conditioning image does not show. On the
riverboat that produced a glass canopy and a deck full of machinery that exist
nowhere in the source, and two independent reviews called them out. No colour
correction touches that, and neither generation-side knob available here fixed
it: raising the reference conditioning scale left the invented structure
unchanged, and classifier-free guidance cannot run in this pipeline at all.

A reference sheet solves it by removing the need to invent. Front, side, back
and deck-plan elevations are real drawings of the real subject, so where one
exists it should replace the generated view outright rather than be blended
with it.

The mapping from drawing to view index is the caller's judgement and is passed
in explicitly, because a sheet labels its panels by the *boat's* anatomy -- bow,
stern, port -- while the control bundle labels its cameras by the *mesh's*
canonical axes. Those two vocabularies do not line up, and on this asset the
contract's "front" and "rear" are the two long sides. Guessing that mapping from
labels is how a texture ends up rotated a quarter turn.

Registration is bounding box to bounding box, the same anisotropic fit the
photograph gets. A drawing and a mesh silhouette will not agree exactly -- the
drawing was made by a person and the mesh by a generator working from one
photograph -- so the achieved IoU is measured and reported per view, and a
drawing that registers badly is worth knowing about before it is baked.

    py -3.12 workers/build_reference_views.py --controls DIR --output DIR \\
        --assign 1=panels/panel_01.png --assign 2=panels/panel_02.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Below this the drawing and the silhouette disagree enough that baking it
#: would smear detail onto the wrong geometry.
MIN_REGISTRATION_IOU = 0.55


def _bbox(mask):
    import numpy as np

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y = np.where(rows)[0]
    x = np.where(cols)[0]
    return int(x[0]), int(y[0]), int(x[-1]), int(y[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fallback-views", default="",
                        help="Directory of generated views used for any index "
                             "with no reference drawing assigned.")
    parser.add_argument("--assign", action="append", default=[],
                        metavar="INDEX=PATH",
                        help="Reference drawing for a view index. Repeatable.")
    parser.add_argument("--mirror", default="",
                        help="Comma separated view indices whose drawing should "
                             "be flipped horizontally before registration.")
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    from pipeline_matte import key_alpha

    controls = Path(args.controls).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((controls / "camera_contract.json").read_text(encoding="utf-8"))
    entries = sorted(contract["views"], key=lambda view: int(view["index"]))

    mirror = {int(v) for v in args.mirror.split(",") if v.strip()}
    assigned = {}
    for item in args.assign:
        index, _, path = item.partition("=")
        assigned[int(index)] = Path(path)

    fallback = Path(args.fallback_views) if args.fallback_views else None

    results = []
    for entry in entries:
        index = int(entry["index"])
        name = str(entry["proven_semantic"])
        target = output / f"view_{index}_{name}.png"

        mask_path = controls / str(entry["control_mask_filename"])
        silhouette = np.asarray(Image.open(mask_path).convert("L")) > 127
        size = silhouette.shape[0]
        silhouette_box = _bbox(silhouette)
        if silhouette_box is None:
            raise SystemExit(f"EMPTY_SILHOUETTE:{index}")

        if index not in assigned:
            if fallback is None:
                raise SystemExit(f"NO_SOURCE_FOR_VIEW:{index}")
            source = next(fallback.glob(f"view_{index}_*.png"))
            Image.open(source).convert("RGB").resize(
                (size, size), Image.LANCZOS).save(target)
            results.append({"index": index, "name": name, "source": "generated",
                            "file": source.name})
            continue

        drawing = Image.open(assigned[index]).convert("RGB")
        if index in mirror:
            drawing = drawing.transpose(Image.FLIP_LEFT_RIGHT)
        alpha, _ = key_alpha(np.asarray(drawing), args.tolerance)
        drawing_mask = alpha > 127
        drawing_box = _bbox(drawing_mask)
        if drawing_box is None:
            raise SystemExit(f"EMPTY_DRAWING:{index}")

        # Crop to the drawing's own extent, then stretch that extent onto the
        # silhouette's. Anisotropic on purpose: the drawing's aspect is the
        # artist's, the silhouette's is the mesh's, and the mesh is what will be
        # textured.
        dx0, dy0, dx1, dy1 = drawing_box
        cropped = drawing.crop((dx0, dy0, dx1 + 1, dy1 + 1))
        cropped_alpha = Image.fromarray(alpha[dy0:dy1 + 1, dx0:dx1 + 1], "L")

        sx0, sy0, sx1, sy1 = silhouette_box
        width = sx1 - sx0 + 1
        height = sy1 - sy0 + 1
        fitted = cropped.resize((width, height), Image.LANCZOS)
        fitted_alpha = cropped_alpha.resize((width, height), Image.LANCZOS)

        # Composite the drawing over the generated view for this index, not over
        # a flat plate.
        #
        # The bake scatters by triangle id and barycentric coordinate and never
        # tests whether the pixel it samples is subject or background. So any
        # part of the mesh silhouette the drawing fails to cover would have the
        # plate colour baked directly onto the model -- and the drawings register
        # at 0.75 to 0.87 IoU, so between a seventh and a quarter of each
        # silhouette is uncovered. That is exactly the pale grey blotching that
        # showed up on the upper decks and stern of the first reference bake.
        #
        # Falling back to the generated view instead means those texels get
        # plausible synthesis rather than flat grey. Worse than a drawing, much
        # better than a hole, and the receipt records how much of each view came
        # from which source so the distinction is never lost.
        if fallback is not None:
            candidates = list(fallback.glob(f"view_{index}_*.png"))
            if candidates:
                base = Image.open(candidates[0]).convert("RGB").resize(
                    (size, size), Image.LANCZOS)
            else:
                base = Image.new("RGB", (size, size), (128, 128, 128))
        else:
            base = Image.new("RGB", (size, size), (128, 128, 128))

        base.paste(fitted, (sx0, sy0), fitted_alpha)
        base.save(target)

        placed = np.zeros_like(silhouette)
        placed[sy0:sy1 + 1, sx0:sx1 + 1] = np.asarray(fitted_alpha) > 127
        union = np.logical_or(placed, silhouette).sum()
        intersection = np.logical_and(placed, silhouette).sum()
        iou = float(intersection) / float(max(union, 1))

        covered = np.logical_and(placed, silhouette).sum()
        results.append({
            "index": index, "name": name, "source": "reference_drawing",
            "file": str(assigned[index]),
            "mirrored": index in mirror,
            "registration_iou": round(iou, 4),
            "drawing_bbox": list(drawing_box),
            "silhouette_bbox": list(silhouette_box),
            "acceptable": bool(iou >= MIN_REGISTRATION_IOU),
            # What fraction of this view's visible surface the drawing actually
            # covers. The remainder falls back to generated pixels, so this is
            # the honest split between authored and synthesised for the view.
            "silhouette_covered_by_drawing": round(
                float(covered) / float(max(silhouette.sum(), 1)), 4),
            "uncovered_falls_back_to": ("generated view" if fallback is not None
                                        else "flat plate"),
        })

    drawings = [r for r in results if r["source"] == "reference_drawing"]
    poor = [r for r in drawings if not r["acceptable"]]
    report = {
        "schema_version": "reference_view_build_v1",
        "classification": "PROVEN" if not poor else "REGISTRATION_BELOW_FLOOR",
        "controls": str(controls),
        "output": str(output),
        "resolution": int(size),
        "views_from_reference": len(drawings),
        "views_from_generation": len(results) - len(drawings),
        "min_registration_iou": MIN_REGISTRATION_IOU,
        "poorly_registered": [r["index"] for r in poor],
        "views": results,
    }
    report_path = Path(args.report) if args.report else output / "reference_views.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not poor else 2


if __name__ == "__main__":
    raise SystemExit(main())
