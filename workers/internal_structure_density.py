"""How much internal structure a mesh actually carries, per unit of silhouette.

`feature_edge_f1.py` compares a candidate to a teacher. That is the right shape
of question when a teacher exists, but the teacher available for this subject
gives trellis512 0.411 and gives a completely different object -- an ornate
tower -- 0.395. A 0.016 margin cannot rank two candidates; it can barely tell
the right object from the wrong one. So for "is res-1024 structurally better
than res-512" the F1 route has no power, and asking it anyway would be
answering with a number that does not know.

This measures each mesh on its own terms instead. Same depth-and-normal
buffers, same internal-edge definition (occlusion steps plus creases, outer
silhouette excluded), but no comparison: just how much internal feature edge
exists per covered pixel, and how sharp it is.

    melted       few internal edges, low density        -- detail never formed
    crisp        many internal edges, coherent          -- decks, railings, panels
    noisy        many internal edges, high fragmentation -- invented lumps

Density alone cannot separate crisp from noisy, so fragmentation is reported
next to it: the number of connected edge components per unit of edge length. A
deck line is one long component; the same edge budget spent on speckle is
hundreds of short ones. Both numbers are needed, and neither is a verdict on
its own -- this is a supporting measure for a clay render, not a replacement.

Face count deliberately earns nothing here. Depth and normal passes are used
rather than shaded RGB precisely so that a denser mesh does not score higher for
being denser.

    py workers/internal_structure_density.py --mesh a.glb --mesh b.glb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from feature_edge_f1 import (feature_edges, load_normalised,  # noqa: E402
                             render_depth_normal)

#: Same six directions for every mesh, so density is comparable across meshes.
#: Fixed, not fitted: a per-mesh camera search would let a mesh pick the views
#: that flatter it.
DIRECTIONS = [
    ("front", (0.0, 0.0, 1.0)),
    ("back", (0.0, 0.0, -1.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("top", (0.0, 1.0, 0.0)),
    ("three_quarter", (0.7, 0.35, 0.7)),
]
UP = (0.0, 1.0, 0.0)


def components(edges: np.ndarray) -> int:
    """Connected components of the edge mask, 8-connected."""
    from scipy import ndimage
    _, count = ndimage.label(edges, structure=np.ones((3, 3), dtype=int))
    return int(count)


def measure(path: Path, size: int) -> dict:
    vertices, faces = load_normalised(path)
    views = []
    for name, forward in DIRECTIONS:
        depth, normals, covered = render_depth_normal(
            vertices, faces, np.array(forward, dtype=np.float64),
            np.array(UP, dtype=np.float64), size)
        edges, _ = feature_edges(depth, normals, covered)
        edge_pixels = int(edges.sum())
        covered_pixels = int(covered.sum())
        parts = components(edges) if edge_pixels else 0
        views.append({
            "view": name,
            "covered_pixels": covered_pixels,
            "edge_pixels": edge_pixels,
            # Internal edge length per unit of visible surface. Scale-free
            # because both meshes are normalised to a unit box first.
            "density": round(edge_pixels / max(covered_pixels, 1), 4),
            "components": parts,
            # Short components mean speckle; long ones mean deck lines.
            "pixels_per_component": round(edge_pixels / max(parts, 1), 2),
        })

    density = float(np.mean([v["density"] for v in views]))
    per_component = float(np.mean([v["pixels_per_component"] for v in views]))
    return {
        "mesh": str(path),
        "faces": int(len(faces)),
        "views": views,
        "mean_density": round(density, 4),
        "mean_pixels_per_component": round(per_component, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    results = [measure(m, args.size) for m in args.mesh]
    payload = {
        "schema": "lowvram3d_internal_structure_v1",
        "size": args.size,
        "results": results,
        "note": ("supporting measure only; density separates melted from "
                 "detailed, fragmentation separates crisp from speckled, and "
                 "neither is a verdict without a clay render"),
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n",
                            encoding="utf-8")
    for r in results:
        print("%-46s faces %8d  density %.4f  px/comp %6.2f"
              % (Path(r["mesh"]).name, r["faces"], r["mean_density"],
                 r["mean_pixels_per_component"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
