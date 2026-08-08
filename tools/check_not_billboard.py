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
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)

    loaded = trimesh.load(args.mesh)
    mesh = loaded.to_mesh() if hasattr(loaded, "to_mesh") else loaded

    stats = measure(mesh)
    failed, reasons = verdict(stats)
    stats["billboard"] = failed
    stats["reasons"] = reasons
    stats["mesh"] = str(Path(args.mesh).resolve())

    for key in ("faces", "extents", "fill", "quad_ratio", "spread_per_axis"):
        print(f"  {key:16} {stats[key]}", flush=True)

    if args.receipt:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(stats, indent=2), encoding="utf-8")

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
