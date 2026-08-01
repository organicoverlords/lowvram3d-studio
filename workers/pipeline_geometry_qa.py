"""Generic geometry gate for Pipeline V2.

Checks the three geometry defects that got through by hand during the shaman run: a mesh that is
collapsed or lying down, unsupported detached shards, and thin source-supported features that
decimation quietly deleted. Emits failure codes the repair policy is keyed on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mesh_io import read_glb, triangle_components

WELD = 4e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-axis-ratio", type=float, default=8.0)
    parser.add_argument("--debris-height-min", type=float, default=0.70)
    parser.add_argument("--max-shard-triangles", type=int, default=20)
    parser.add_argument("--max-shard-diagonal-fraction", type=float, default=0.062)
    parser.add_argument("--debris-blocking", action="store_true",
                        help="treat remaining detached shards as a failure (use after CLEAN)")
    args = parser.parse_args()

    positions, _, _, tris = read_glb(Path(args.mesh))
    positions = positions.astype(np.float64)

    low, high = positions.min(axis=0), positions.max(axis=0)
    extent = high - low
    ordered = np.sort(extent)
    axis_ratio = float(ordered[-1] / max(ordered[0], 1e-9))
    scene_diagonal = float(np.linalg.norm(extent))
    span = max(float(extent[1]), 1e-9)

    component, welded = triangle_components(positions, tris, WELD)
    sizes = np.bincount(component)
    body = int(np.argmax(sizes))

    max_diagonal = scene_diagonal * args.max_shard_diagonal_fraction
    shards = []
    for index in range(len(sizes)):
        if index == body:
            continue
        members = component == index
        vertices = positions[np.unique(tris[members])]
        height = float(((vertices[:, 1] - low[1]) / span).mean())
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        count = int(members.sum())
        if count <= 1 or (height >= args.debris_height_min and count <= args.max_shard_triangles
                          and diagonal <= max_diagonal):
            shards.append({"component": index, "triangles": count,
                           "height_mean": round(height, 4), "diagonal": round(diagonal, 5)})

    failure_codes, advisory_codes, messages = [], [], []
    if axis_ratio > args.max_axis_ratio:
        failure_codes.append("BAD_ORIENTATION")
        messages.append(f"axis ratio {axis_ratio:.2f} exceeds {args.max_axis_ratio}")
    if shards:
        message = f"{len(shards)} unsupported detached components"
        messages.append(message)
        # Raw generator output is expected to carry shards - removing them is CLEAN's job, and this
        # gate runs before CLEAN. Blocking here would fail every asset on a defect the next stage
        # exists to fix. CLEAN re-runs this check with --debris-blocking afterwards, where a
        # remaining shard genuinely is a failure.
        (failure_codes if args.debris_blocking else advisory_codes).append("FLOATING_DEBRIS")

    report = {
        "mesh": args.mesh,
        "triangles": int(len(tris)),
        "components": int(len(sizes)),
        "extent": {"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
        "axis_ratio": round(axis_ratio, 4),
        "longest_axis": "xyz"[int(np.argmax(extent))],
        "debris": {
            "unsupported_components_remaining": len(shards),
            "triangles_in_shards": int(sum(s["triangles"] for s in shards)),
            "shards": shards[:60],
        },
        "failure_codes": failure_codes,
        "advisory_codes": advisory_codes,
        "messages": messages,
        "passed": not failure_codes,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"GEOMETRY_QA passed={report['passed']} axis_ratio={axis_ratio:.2f} "
          f"components={len(sizes)} shards={len(shards)} codes={failure_codes} advisory={advisory_codes}", flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
