"""Drop unsupported detached shards, writing a derived copy and never touching the input.

The output carries the same vertices and, where present, the same UVs, minus whole triangles - so
every surviving triangle's texture coordinate is byte-identical and an atlas built against the
original still registers exactly.

Only detached components qualify, and only small ones above a height threshold. Anything attached
to the main body is never a candidate, which is what separates "unsupported shard" from
"intentional protrusion" without a judgement call about shape: a twig growing out of the mesh is
part of the body component, a floating triangle is not. Large or low detached parts - hanging
ornaments, cords, held props - are excluded by the height and diagonal limits.

Single-triangle islands are removed wherever they sit. One triangle cannot represent a branch, a
charm or a cord; it is always generator debris.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components, write_glb

WELD = 4e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--height-min", type=float, default=0.70)
    parser.add_argument("--max-triangles", type=int, default=20)
    parser.add_argument("--max-diagonal-fraction", type=float, default=0.062)
    args = parser.parse_args()

    positions, normals, uv, tris = read_glb(Path(args.input))
    component, welded = triangle_components(positions, tris, WELD)

    low, high = positions.min(axis=0), positions.max(axis=0)
    span = max(float(high[1] - low[1]), 1e-9)
    scene_diagonal = float(np.linalg.norm(high - low))
    max_diagonal = scene_diagonal * args.max_diagonal_fraction

    sizes = np.bincount(component)
    body = int(np.argmax(sizes))

    removed, kept = [], []
    drop = np.zeros(len(tris), bool)
    for index in range(len(sizes)):
        members = component == index
        count = int(members.sum())
        vertices = positions[np.unique(tris[members])]
        height = float(((vertices[:, 1] - low[1]) / span).mean())
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        record = {"component": index, "triangles": count,
                  "height_mean": round(height, 4), "diagonal": round(diagonal, 5)}
        if index == body:
            record["verdict"] = "kept: main body (carries the attached antler twigs)"
            kept.append(record)
            continue
        if count <= 1:
            record["verdict"] = "removed: single-triangle island cannot be a real feature"
        elif height >= args.height_min and count <= args.max_triangles and diagonal <= max_diagonal:
            record["verdict"] = "removed: detached shard at/above the antlers"
        else:
            reasons = []
            if height < args.height_min:
                reasons.append("below the head region")
            if count > args.max_triangles:
                reasons.append("too many triangles for a shard")
            if diagonal > max_diagonal:
                reasons.append("too large - reads as a cord or ornament")
            record["verdict"] = "kept: " + ", ".join(reasons)
            kept.append(record)
            continue
        drop |= members
        removed.append(record)

    survivors = tris[~drop]
    used = np.unique(survivors)
    remap = np.full(len(positions), -1, np.int64)
    remap[used] = np.arange(len(used))

    # A mesh straight out of the generator has no UVs yet; this stage runs before unwrapping as
    # often as after it, so carry them through only when they exist.
    kept_uv = uv[used] if uv is not None else None
    write_glb(Path(args.output), positions[used], normals[used], kept_uv, remap[survivors])

    # The whole point of dropping triangles rather than editing them: the UVs that survive must be
    # bit-identical, or an atlas built against the original no longer registers.
    check_positions, _, check_uv, _ = read_glb(Path(args.output))
    uv_identical = None if kept_uv is None else bool(np.array_equal(check_uv, kept_uv))
    positions_identical = bool(np.array_equal(check_positions, positions[used]))

    report = {
        "input": args.input,
        "output": args.output,
        "height_min": args.height_min,
        "max_triangles": args.max_triangles,
        "max_diagonal": round(max_diagonal, 5),
        "components_total": int(len(sizes)),
        "components_removed": len(removed),
        "triangles_before": int(len(tris)),
        "triangles_after": int(len(survivors)),
        "triangles_removed": int(drop.sum()),
        "triangles_removed_percent": round(float(drop.sum() / len(tris) * 100), 4),
        "uv_bit_identical": uv_identical,
        "positions_bit_identical": positions_identical,
        "removed": sorted(removed, key=lambda r: -r["triangles"]),
        "kept_non_body": sorted([k for k in kept if k["component"] != body],
                                key=lambda r: -r["triangles"]),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DEBRIS_STRIP removed {len(removed)} components / {int(drop.sum())} triangles "
        f"({report['triangles_removed_percent']}%), kept {len(report['kept_non_body'])} detached parts, "
        f"uv_identical={uv_identical}",
        flush=True,
    )


if __name__ == "__main__":
    main()
