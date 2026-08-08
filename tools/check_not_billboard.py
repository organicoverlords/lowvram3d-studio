"""Fail a generated mesh that is a billboard rather than a solid.

TRELLIS can collapse a subject onto two crossed flat planes and still exit 0
with `success: true`, `geometry_decoded: true` and a finalizer note listing
every stage. The greentree at res 512 did exactly that: 146,326 faces, a
0.78 x 1.00 x 0.77 bounding box, a 1024 atlas, a plausible receipt -- and two
intersecting cardboard panels. Nothing in the receipt distinguishes it from a
real asset, because every field it records was true.

Three measurements do distinguish it, and none of them needs a render:

  fill      mesh volume over bounding-box volume. A solid body sits in the
            percent-to-tens-of-percent range; the greentree was 1.0%.

  area      surface area over the area of the crossed quads that the bounding
            box would allow. Two double-sided planes spanning the box give
            2 * 2 * w * h; the greentree scored 0.996 of that. A solid body
            cannot approach 1.0 without being those planes.

  spread    the fraction of vertices falling in the two busiest bins of a
            20-bin histogram along each axis. A plane normal to X puts every
            vertex it owns at one X, so the crossed pair scored 55% on X and
            55% on Z against an even 10% expectation.

`fill` alone is not enough: a thin shell of a real subject is also mostly
empty. It is the combination -- low fill AND near-quad area AND concentrated
spread on two axes -- that is only produced by a billboard.

Exit 1 on failure so a shell script with `set -e` stops before spending forty
minutes painting cardboard.

    py tools/check_not_billboard.py --mesh out.glb --receipt gate.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

# Measured, not guessed. The first cut of this file guessed and was wrong in a
# way worth recording, because the mistake is easy to repeat: `quad_ratio` was
# written as an upper bound on the theory that a billboard has "too much area".
# It does not. A billboard has *exactly* the area of the crossed quads, so its
# ratio sits at 1.0, and every solid sits ABOVE it. A ceiling of 0.75 was below
# every real asset in the project and fired on all of them.
#
#   asset                 fill     quad_ratio   worst spread   truth
#   greentree TRELLIS     0.0101   0.998        0.554          cardboard
#   seal diver TRELLIS    0.0253   1.752        0.178          solid
#   moss titan TRELLIS    0.0414   2.236        0.196          solid
#   greentree Mini Turbo  0.1712   2.853        0.231          solid
#
# So the billboard signature is quad_ratio NEAR one, not above a threshold.
# `fill` alone cannot carry the test either: the seal diver is a genuine solid
# at 0.025 because flippers, ropes, an anchor and a swinging lantern inflate the
# bounding box far beyond the body inside it.
FILL_FLOOR = 0.020
QUAD_NEAR_ONE = 1.30
SPREAD_CEILING = 0.35


def measure(mesh: trimesh.Trimesh) -> dict:
    extents = mesh.bounds[1] - mesh.bounds[0]
    box_volume = float(np.prod(extents))
    fill = float(mesh.volume) / box_volume if box_volume > 0 else 0.0

    # The largest crossed pair the box admits: two double-sided quads, each
    # spanning the long axis and one of the other two.
    order = np.argsort(extents)[::-1]
    long_side = extents[order[0]]
    quad_area = 2.0 * 2.0 * long_side * (extents[order[1]] + extents[order[2]]) / 2.0
    quad_ratio = float(mesh.area) / quad_area if quad_area > 0 else 0.0

    spreads = []
    for axis in range(3):
        counts, _ = np.histogram(mesh.vertices[:, axis], bins=20)
        top_two = np.sort(counts)[-2:].sum() / len(mesh.vertices)
        spreads.append(float(top_two))

    return {
        "extents": [round(float(e), 4) for e in extents],
        "faces": int(len(mesh.faces)),
        "volume": round(float(mesh.volume), 6),
        "area": round(float(mesh.area), 4),
        "fill": round(fill, 4),
        "quad_ratio": round(quad_ratio, 4),
        "spread_per_axis": [round(s, 4) for s in spreads],
        "worst_spread": round(max(spreads), 4),
    }


# A decode that keeps only part of the subject is a different failure from a
# billboard, and the three measures above cannot see it: the part that survives
# is genuinely solid, so fill, quad_ratio and spread all read healthy. The tree
# city decoded its root mass, trunk, walkway and doorways beautifully and simply
# omitted the entire canopy -- 60% of the subject gone, gate passed.
#
# The matte knows what the subject's proportions actually are. Comparing the
# decoded mesh's upright aspect against the silhouette's catches truncation for
# the cost of reading one PNG.
ASPECT_TOLERANCE = 0.45


def silhouette_aspect(matte_path: str) -> float | None:
    """Height / width of the matte's opaque region, or None if unreadable."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        alpha = np.array(Image.open(matte_path).convert("RGBA"))[..., 3]
    except Exception as error:
        print(f"  (matte unreadable: {error})", flush=True)
        return None
    rows, columns = np.where(alpha > 128)
    if len(rows) < 16:
        return None
    height = float(rows.max() - rows.min() + 1)
    width = float(columns.max() - columns.min() + 1)
    return height / width if width > 0 else None


def check_truncation(mesh, matte_path: str) -> tuple[dict, str | None]:
    """Compare the mesh's upright aspect with the source silhouette's.

    The generated mesh is normalised to a unit box and is Y-up, so its height is
    extents[1] and its footprint is the larger of the two horizontal extents.
    A subject that lost its top reads much squatter than its own photograph.
    """
    source_aspect = silhouette_aspect(matte_path)
    if source_aspect is None:
        return {"silhouette_aspect": None, "mesh_aspect": None}, None

    extents = mesh.bounds[1] - mesh.bounds[0]
    footprint = max(float(extents[0]), float(extents[2]))
    mesh_aspect = float(extents[1]) / footprint if footprint > 0 else 0.0

    detail = {"silhouette_aspect": round(source_aspect, 4),
              "mesh_aspect": round(mesh_aspect, 4),
              "aspect_ratio_of_ratios": round(mesh_aspect / source_aspect, 4)
              if source_aspect else None}

    # Only flag SHORTER than the source. A mesh taller than its silhouette is
    # usually a framing difference, not a loss.
    if mesh_aspect < source_aspect * (1.0 - ASPECT_TOLERANCE):
        return detail, (f"mesh aspect {mesh_aspect:.2f} against silhouette "
                        f"{source_aspect:.2f} -- the decode is far squatter "
                        f"than the subject, so part of it is missing")
    return detail, None


def verdict(stats: dict) -> tuple[bool, list[str]]:
    reasons = []
    if stats["fill"] < FILL_FLOOR:
        reasons.append(f"fill {stats['fill']:.4f} < {FILL_FLOOR} "
                       f"(mesh encloses almost no volume)")
    if stats["quad_ratio"] < QUAD_NEAR_ONE:
        reasons.append(f"quad_ratio {stats['quad_ratio']:.4f} < {QUAD_NEAR_ONE} "
                       f"(surface area is exactly that of crossed flat panels)")
    # Two concentrated axes is the billboard signature; one is a flat base.
    concentrated = [s for s in stats["spread_per_axis"] if s > SPREAD_CEILING]
    if len(concentrated) >= 2:
        reasons.append(f"spread {stats['spread_per_axis']} "
                       f"(vertices pile onto planes on two axes)")
    # Any single symptom can occur on a legitimately thin asset. Two together
    # have only ever been produced by a collapse.
    return len(reasons) >= 2, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--matte",
                        help="the subject's matte PNG; enables the truncation "
                             "check, which compares the decoded mesh's upright "
                             "aspect against the source silhouette's")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)

    loaded = trimesh.load(args.mesh)
    mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else loaded

    stats = measure(mesh)
    failed, reasons = verdict(stats)

    truncated = None
    if args.matte and Path(args.matte).is_file():
        detail, truncated = check_truncation(mesh, args.matte)
        stats.update(detail)
        stats["truncated"] = bool(truncated)
    stats["billboard"] = failed
    stats["reasons"] = reasons
    stats["mesh"] = str(Path(args.mesh).resolve())

    for key in ("faces", "extents", "fill", "quad_ratio", "spread_per_axis"):
        print(f"  {key:16} {stats[key]}", flush=True)

    if args.receipt:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if truncated:
        print(f"  silhouette_aspect  {stats.get('silhouette_aspect')}", flush=True)
        print(f"  mesh_aspect        {stats.get('mesh_aspect')}", flush=True)
        print(f"TRUNCATION_WARNING: {truncated}", flush=True)

    if failed:
        print("BILLBOARD_ABORT: this mesh is flat panels, not a solid",
              flush=True)
        for reason in reasons:
            print(f"  - {reason}", flush=True)
        return 1
    print("  not a billboard -- continuing", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
