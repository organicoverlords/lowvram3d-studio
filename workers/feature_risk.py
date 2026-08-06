"""Predict which features the generator will lose, before generating anything.

The generator decides occupancy on a 32^3 grid before it refines anything. With
a 512 px input that is one structural cell every 16 px of frame, and a feature
that cannot claim a cell is not built -- no amount of `--res` recovers it,
because raising `--res` leaves the occupancy grid at 32 and only sharpens what
the grid already claimed. Measured: a res-512 run logs 1901 active voxels and a
res-1024 run of the same subject logs 1921.

Thickness alone does not predict survival, which is the part that took four
subjects to work out:

    hanging strings   1 px wide, ~150 px long   survived
    pendants          15-23 px across           blobbed into identical stubs
    portcullis bars   ~3 px                     vanished
    crenellations     ~6 px, tightly repeated   merged into a flat parapet

A 1 px cord beats a 20 px pendant. The reason is *run length*: a long feature
claims cells along its length even when it is thinner than one cell, while a
compact prop has one chance at one cell and loses it. So the predictor needs
both numbers.

Method. Morphologically open the alpha with a disc of the structural-cell
radius. Whatever the opening removes is thinner than a cell -- that is the
at-risk set, by construction and without any knowledge of the subject. Label
those regions, and for each measure:

    thickness   2x the maximum distance-to-background inside the region
    run length  the region's skeleton length, i.e. how far it extends

then classify:

    thickness >= 1 cell                      -> safe
    thin, but run length >= RUN_CELLS cells  -> likely survives as a strand
    thin and compact                         -> likely lost

The output is a ranked list of what will be lost and a suggested crop for a
detail pass, because the only real fix is to reframe so the feature is larger
relative to the subject. Nothing here knows what a pendant, a bar or a
crenellation is; it measures geometry against a grid.

    py feature_risk.py --image crop512.png --overlay risk.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The occupancy grid, fixed at 32 regardless of --res. See module docstring.
STRUCTURAL_GRID = 32

#: A thin region is predicted to survive if it runs at least this many cells.
#: The strings span roughly 9-10 cells and survived; the pendants span about
#: one and did not. 2.0 sits between them, nearer the failure side, because a
#: false "safe" is more costly than a false "at risk".
RUN_CELLS = 2.0

#: Regions smaller than this fraction of subject area are ignored as texture
#: noise or matte fringe rather than features anyone would miss.
MIN_REGION_FRACTION = 2e-4

#: Padding applied to a suggested detail-pass crop, as a fraction of its size.
CROP_PADDING = 0.35


def run_length(region, distance):
    """How far a region extends along itself, in pixels.

    Area divided by mean thickness. For a ribbon of roughly constant width that
    is its centreline length, and unlike a bounding-box diagonal it stays
    correct when the strand curves or hangs at an angle -- which every cord on a
    hanging mobile does. It also needs no skeletonisation, so there is no
    scikit-image dependency for one function.
    """
    import numpy as np

    area = float(region.sum())
    mean_thickness = float(distance[region].mean()) * 2.0
    return area / max(mean_thickness, 1.0)


def suggest_crops(lost, size, max_crops=4, min_magnification=1.6):
    """Cluster lost features into square detail-pass crops.

    Agglomerative by centroid distance: merge the two nearest clusters while
    the merged crop still magnifies enough to be worth a pass. A crop that only
    magnifies 1.2x reproduces the failure it was meant to fix, so crops below
    `min_magnification` are dropped rather than reported -- an honest empty list
    beats a suggestion that cannot work.
    """
    import numpy as np

    if not lost:
        return []
    boxes = [f["bbox"] for f in lost]
    weights = [f["area_px"] for f in lost]
    clusters = [{"box": list(b), "weight": w} for b, w in zip(boxes, weights)]

    def merged(a, b):
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def span_of(box):
        return max(box[2] - box[0], box[3] - box[1])

    while len(clusters) > max_crops:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                box = merged(clusters[i]["box"], clusters[j]["box"])
                cost = span_of(box)
                if best is None or cost < best[0]:
                    best = (cost, i, j, box)
        _, i, j, box = best
        clusters[i] = {"box": box,
                       "weight": clusters[i]["weight"] + clusters[j]["weight"]}
        clusters.pop(j)

    out = []
    for cluster in sorted(clusters, key=lambda c: -c["weight"]):
        box = cluster["box"]
        pad = int(span_of(box) * CROP_PADDING) + 4
        crop = [max(box[0] - pad, 0), max(box[1] - pad, 0),
                min(box[2] + pad, size), min(box[3] + pad, size)]
        span = max(span_of(crop), 1)
        magnification = size / span
        if magnification < min_magnification:
            continue
        out.append({
            "crop_px": crop,
            "crop_normalised": [round(v / size, 4) for v in crop],
            "lost_area_px": int(cluster["weight"]),
            "magnification": round(magnification, 2),
            "structural_cells_gained": round(magnification, 1),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="The prepared square RGBA input, as fed to the "
                             "generator.")
    parser.add_argument("--grid", type=int, default=STRUCTURAL_GRID)
    parser.add_argument("--run-cells", type=float, default=RUN_CELLS)
    parser.add_argument("--overlay", default="",
                        help="Write a PNG marking at-risk geometry, so the "
                             "prediction can be checked by eye.")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    image = Image.open(args.image).convert("RGBA")
    alpha = np.asarray(image)[..., 3] > 8
    if not alpha.any():
        raise SystemExit("NO_ALPHA: matte the image first")

    size = alpha.shape[0]
    cell = size / args.grid
    radius = max(int(round(cell / 2.0)), 1)

    # Distance to background: half-thickness at every interior pixel.
    distance = ndimage.distance_transform_edt(alpha)

    # Opening with a disc of the cell radius. What survives is at least a cell
    # thick; what it removes is the at-risk set.
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = (xx ** 2 + yy ** 2) <= radius ** 2
    opened = ndimage.binary_opening(alpha, structure=disc)
    at_risk = alpha & ~opened

    # The dual problem, and the one that actually explains the pendants.
    #
    # Opening finds solid geometry too thin to claim a cell. But a feature can
    # be thick enough to survive as *mass* and still lose its *shape*: the
    # pendants came through as identical stubs because the hole through one and
    # the concavity of another are negative-space features about 6-8 px across,
    # under one 16 px cell. Closing with the same disc fills exactly those, so
    # whatever the closing adds is detail the generator will smooth away.
    #
    # This correctly spares the staff ring, whose hole is ~30 px and which did
    # survive with a real hole through it.
    closed = ndimage.binary_closing(alpha, structure=disc)
    fill_risk = closed & ~alpha

    labels, count = ndimage.label(at_risk)
    subject_area = float(alpha.sum())
    findings = []
    for index in range(1, count + 1):
        region = labels == index
        area = int(region.sum())
        if area / subject_area < MIN_REGION_FRACTION:
            continue
        thickness = float(distance[region].max() * 2.0)
        run = run_length(region, distance)
        ys, xs = np.nonzero(region)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        thickness_cells = thickness / cell
        run_cells = run / cell
        if thickness_cells >= 1.0:
            verdict = "safe"
        elif run_cells >= args.run_cells:
            verdict = "strand"      # thin but long: claims cells along its run
        else:
            verdict = "at_risk"     # thin and compact: no cell to claim
        findings.append({
            "bbox": bbox, "area_px": area,
            "thickness_px": round(thickness, 1),
            "thickness_cells": round(thickness_cells, 2),
            "run_px": round(run, 1),
            "run_cells": round(run_cells, 2),
            "verdict": verdict,
        })

    # Negative-space findings: holes and concavities that will fill in.
    fill_labels, fill_count = ndimage.label(fill_risk)
    for index in range(1, fill_count + 1):
        region = fill_labels == index
        area = int(region.sum())
        if area / subject_area < MIN_REGION_FRACTION:
            continue
        gap = float(ndimage.distance_transform_edt(region).max() * 2.0)
        ys, xs = np.nonzero(region)
        findings.append({
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1],
            "area_px": area,
            "thickness_px": round(gap, 1),
            "thickness_cells": round(gap / cell, 2),
            "run_px": round(run_length(region, ndimage.distance_transform_edt(region)), 1),
            "run_cells": None,
            # The mass survives; the shape does not. Distinct from at_risk,
            # where nothing is built at all.
            "verdict": "shape_lost",
        })

    order = {"at_risk": 0, "shape_lost": 1, "strand": 2, "safe": 3}
    findings.sort(key=lambda f: (order.get(f["verdict"], 9), -f["area_px"]))
    lost = [f for f in findings
            if f["verdict"] in ("at_risk", "shape_lost")]

    # A detail pass is the only real remedy: reframing so the feature is larger
    # relative to the structural grid. Raising --res does not work -- the
    # occupancy grid stays at 32 either way.
    #
    # One bounding box over every lost feature is useless when they are spread
    # across the subject: on the shaman it returns the whole frame at 1.0x
    # magnification, which is the run that just failed. So cluster them and
    # propose several crops, each of which is worth a pass only if it actually
    # magnifies.
    suggestions = suggest_crops(lost, size)

    receipt = {
        "schema_version": "feature_risk_v1",
        "image": str(Path(args.image).resolve()),
        "input_size": size,
        "structural_grid": args.grid,
        "structural_cell_px": round(cell, 2),
        "regions_examined": len(findings),
        "at_risk": sum(1 for f in findings if f["verdict"] == "at_risk"),
        "shape_lost": sum(1 for f in findings if f["verdict"] == "shape_lost"),
        "strands": sum(1 for f in findings if f["verdict"] == "strand"),
        "findings": findings[:args.top],
        "detail_passes": suggestions,
    }

    if args.overlay:
        rgb = np.asarray(image.convert("RGB")).copy()
        strand = np.zeros_like(alpha)
        risky = np.zeros_like(alpha)
        for index in range(1, count + 1):
            region = labels == index
            if region.sum() / subject_area < MIN_REGION_FRACTION:
                continue
            thickness = distance[region].max() * 2.0
            if thickness / cell >= 1.0:
                continue
            run = run_length(region, distance)
            (strand if run / cell >= args.run_cells else risky)[region] = True
        rgb[strand] = [40, 190, 255]    # blue: thin but long, expected to live
        rgb[risky] = [255, 40, 40]      # red: thin and compact, expected to die
        # Yellow: mass survives, shape does not -- holes and concavities that
        # will fill in. This is the pendant failure, which neither of the other
        # two classes catches.
        rgb[fill_risk] = [255, 220, 40]
        Image.fromarray(rgb).save(args.overlay)

    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
