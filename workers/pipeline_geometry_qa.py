"""Generic geometry gate for Pipeline V2.

Checks orientation/collapse and detached debris. Post-LOD verification uses the original source
silhouette, because low-poly cords and pendants may legitimately collapse to one triangle. A tiny
LOD component blocks only when it is also high/outboard and absent from the source in both mirrored
and non-mirrored registration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components
from source_support import component_position, component_support, load_support_context

WELD = 4e-4
SOURCE_SUPPORT_MIN = 0.18
OUTBOARD_MIN = 0.14
VERY_HIGH_MIN = 0.78


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-axis-ratio", type=float, default=8.0)
    parser.add_argument("--debris-height-min", type=float, default=0.70)
    parser.add_argument("--max-shard-triangles", type=int, default=20)
    parser.add_argument("--max-shard-diagonal-fraction", type=float, default=0.062)
    parser.add_argument(
        "--debris-blocking",
        action="store_true",
        help="treat remaining detached shards as a failure",
    )
    parser.add_argument(
        "--support-reference-mesh",
        default="",
        help="score source support in this mesh's frame instead of the graded mesh's own",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    positions, _, _, tris = read_glb(mesh_path)
    positions = positions.astype(np.float64)
    low, high = positions.min(axis=0), positions.max(axis=0)
    extent = high - low
    ordered = np.sort(extent)
    axis_ratio = float(ordered[-1] / max(ordered[0], 1e-9))
    scene_diagonal = float(np.linalg.norm(extent))
    legacy_span = max(float(extent[1]), 1e-9)

    labels, _ = triangle_components(positions, tris, WELD)
    sizes = np.bincount(labels)
    body = int(np.argmax(sizes))
    max_diagonal = scene_diagonal * args.max_shard_diagonal_fraction
    # Support is scored by projecting a component into a frame derived from percentiles of the
    # mesh's own positions. That makes the score depend on which *other* components exist: strip
    # nineteen of them and the frame shifts, re-scoring every survivor. A cleaner and the gate that
    # verifies it then disagree about the same triangles - measured here as support 0.375 -> 0.125,
    # 0.5 -> 0.0, 1.0 -> 0.0 - and the repair can never reach a fixed point. Grading in the
    # pre-cleanup frame keeps the two in the same coordinate system.
    support_positions = positions
    if args.support_reference_mesh:
        reference = Path(args.support_reference_mesh)
        if reference.is_file():
            support_positions = read_glb(reference)[0].astype(np.float64)
    context = load_support_context(mesh_path, support_positions)
    shards: list[dict] = []
    preserved_small: list[dict] = []

    for index, size_value in enumerate(sizes):
        if index == body:
            continue
        members = labels == index
        vertices = positions[np.unique(tris[members])]
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        count = int(size_value)
        tiny = count <= args.max_shard_triangles and diagonal <= max_diagonal
        record: dict = {
            "component": index,
            "triangles": count,
            "diagonal": round(diagonal, 6),
        }

        if context is not None:
            position = component_position(context, vertices)
            support = component_support(context, positions, tris, members)
            height = float(position["height_mean"])
            lateral = float(position["lateral_mean"])
            high_or_outboard = height >= args.debris_height_min and (
                lateral >= OUTBOARD_MIN or height >= VERY_HIGH_MIN
            )
            unsupported = float(support["support"]) < SOURCE_SUPPORT_MIN
            record.update({
                "height_mean": round(height, 5),
                "lateral_mean": round(lateral, 5),
                "source_support": support,
            })
            is_shard = tiny and high_or_outboard and unsupported
            if not is_shard and tiny:
                record["preservation_reason"] = (
                    "small LOD feature retained: source-supported or not high/outboard"
                )
                preserved_small.append(record)
        else:
            height = float(((vertices[:, 1] - low[1]) / legacy_span).mean())
            record["height_mean"] = round(height, 5)
            is_shard = count <= 1 or (
                height >= args.debris_height_min
                and count <= args.max_shard_triangles
                and diagonal <= max_diagonal
            )
        if is_shard:
            shards.append(record)

    failure_codes: list[str] = []
    advisory_codes: list[str] = []
    messages: list[str] = []
    if axis_ratio > args.max_axis_ratio:
        failure_codes.append("BAD_ORIENTATION")
        messages.append(f"axis ratio {axis_ratio:.2f} exceeds {args.max_axis_ratio}")
    if shards:
        messages.append(f"{len(shards)} unsupported detached components")
        (failure_codes if args.debris_blocking else advisory_codes).append("FLOATING_DEBRIS")

    report = {
        "mesh": str(mesh_path),
        "triangles": int(len(tris)),
        "components": int(len(sizes)),
        "extent": {"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
        "axis_ratio": round(axis_ratio, 4),
        "longest_axis": "xyz"[int(np.argmax(extent))],
        "debris": {
            "policy": "source_supported_post_lod" if context is not None else "legacy_pre_lod",
            "source_aware": context is not None,
            "source_path": None if context is None else str(context.source_path),
            "unsupported_components_remaining": len(shards),
            "triangles_in_shards": int(sum(row["triangles"] for row in shards)),
            "shards": shards[:60],
            "preserved_small_components": preserved_small[:100],
            "preserved_small_component_count": len(preserved_small),
            "source_support_min": SOURCE_SUPPORT_MIN,
            "outboard_min": OUTBOARD_MIN,
            "very_high_min": VERY_HIGH_MIN,
        },
        "failure_codes": failure_codes,
        "advisory_codes": advisory_codes,
        "messages": messages,
        "passed": not failure_codes,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"GEOMETRY_QA passed={report['passed']} axis_ratio={axis_ratio:.2f} "
        f"components={len(sizes)} shards={len(shards)} preserved_small={len(preserved_small)} "
        f"policy={report['debris']['policy']} codes={failure_codes}",
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
