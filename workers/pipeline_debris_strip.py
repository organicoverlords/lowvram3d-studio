"""Drop unsupported detached shards into a derived GLB.

For pre-LOD generator cleanup, the established conservative component policy is retained. For a
post-LOD mesh, face count alone is never evidence of debris: decimation can reduce a valid cord,
leaf or pendant to one triangle. Post-LOD removal therefore requires all of:

* detached from the dominant welded component;
* tiny by face count and world-space diagonal;
* high/outboard relative to the subject;
* unsupported by the original source silhouette in both mirrored and non-mirrored registration.

Surviving positions and UV coordinates are carried byte-identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components, write_glb
from source_support import (component_position, component_support,
                            component_surface_separation, load_support_context)

WELD = 4e-4
SOURCE_SUPPORT_MIN = 0.18
OUTBOARD_MIN = 0.14
VERY_HIGH_MIN = 0.78
MIN_SEPARATION_FRACTION = 0.005
MAX_RELATIVE_COMPONENT_FACES = 0.002
MAX_RELATIVE_COMPONENT_DIAGONAL = 0.075


def _bbox_gap(low_a: np.ndarray, high_a: np.ndarray,
              low_b: np.ndarray, high_b: np.ndarray) -> float:
    gap = np.maximum(np.maximum(low_a - high_b, low_b - high_a), 0.0)
    return float(np.linalg.norm(gap))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--height-min", type=float, default=0.70)
    parser.add_argument("--max-triangles", type=int, default=20)
    parser.add_argument("--max-diagonal-fraction", type=float, default=0.062)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--support-reference-mesh", default="")
    parser.add_argument("--up-axis", default="")
    parser.add_argument("--front-axis", default="")
    parser.add_argument("--right-axis", default="")
    parser.add_argument("--allow-source-mirror", action="store_true")
    parser.add_argument("--protected-components", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    positions, normals, uv, tris = read_glb(input_path)
    positions = positions.astype(np.float64)
    component, _ = triangle_components(positions, tris, WELD)
    context = load_support_context(
        input_path, positions,
        source_image=Path(args.source_image) if args.source_image else None,
        support_reference_mesh=(Path(args.support_reference_mesh)
                                if args.support_reference_mesh else None),
        up_axis=args.up_axis or None, front_axis=args.front_axis or None,
        right_axis=args.right_axis or None,
        allow_source_mirror=True if args.allow_source_mirror else None,
    )

    low, high = positions.min(axis=0), positions.max(axis=0)
    scene_diagonal = float(np.linalg.norm(high - low))
    max_diagonal = scene_diagonal * args.max_diagonal_fraction
    legacy_span = max(float(high[1] - low[1]), 1e-9)
    protected_components = {
        int(value) for value in str(args.protected_components).split(",") if value.strip()
    }

    sizes = np.bincount(component)
    body = int(np.argmax(sizes))
    body_vertices = positions[np.unique(tris[component == body])]
    body_low, body_high = body_vertices.min(axis=0), body_vertices.max(axis=0)
    removed: list[dict] = []
    kept: list[dict] = []
    drop = np.zeros(len(tris), bool)

    for index, size_value in enumerate(sizes):
        members = component == index
        count = int(size_value)
        vertices = positions[np.unique(tris[members])]
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        record: dict = {
            "component": index,
            "triangles": count,
            "diagonal": round(diagonal, 6),
        }
        if index == body:
            record["verdict"] = "kept: dominant attached body"
            kept.append(record)
            continue

        surface_separation = component_surface_separation(body_vertices, vertices)
        separated = surface_separation > scene_diagonal * MIN_SEPARATION_FRACTION
        small_relative = (count / max(len(tris), 1) <= MAX_RELATIVE_COMPONENT_FACES
                          or diagonal <= scene_diagonal * MAX_RELATIVE_COMPONENT_DIAGONAL)
        tiny = count <= args.max_triangles and diagonal <= max_diagonal
        if context is not None:
            position = component_position(context, vertices)
            support = component_support(context, positions, tris, members)
            height = float(position["height_mean"])
            lateral = float(position["lateral_mean"])
            source_score = float(support["support"])
            high_or_outboard = height >= args.height_min and (
                lateral >= OUTBOARD_MIN or height >= VERY_HIGH_MIN
            )
            unsupported = source_score < SOURCE_SUPPORT_MIN
            record.update({
                "height_mean": round(height, 5),
                "lateral_mean": round(lateral, 5),
                "source_support": support,
                "bbox_separation": round(_bbox_gap(vertices.min(axis=0), vertices.max(axis=0),
                                                     body_low, body_high), 6),
                "surface_separation": round(surface_separation, 6),
                "relative_face_fraction": round(count / max(len(tris), 1), 8),
                "relative_diagonal_fraction": round(diagonal / max(scene_diagonal, 1e-9), 8),
            })
            protected = index in protected_components
            remove = (not protected and separated and small_relative and unsupported
                      and (high_or_outboard or source_score <= 0.0))
            if remove:
                record["verdict"] = (
                    "removed: detached small component lacks direct/mirrored source-silhouette support"
                )
            else:
                reasons = []
                if protected:
                    reasons.append("manifest protected")
                if not separated:
                    reasons.append("close to dominant body")
                if not small_relative:
                    reasons.append("not small relative to mesh")
                if not high_or_outboard:
                    reasons.append("not high/outboard")
                if not unsupported:
                    reasons.append("supported by source silhouette")
                record["verdict"] = "kept: " + ", ".join(reasons or ["ambiguous"])
        else:
            # Legacy pre-LOD cleanup. It runs before source-aware decimation and is preserved to
            # avoid changing previously proven high-master cleanup semantics.
            height = float(((vertices[:, 1] - low[1]) / legacy_span).mean())
            record["height_mean"] = round(height, 5)
            remove = count <= 1 or (
                height >= args.height_min
                and count <= args.max_triangles
                and diagonal <= max_diagonal
            )
            if remove:
                record["verdict"] = (
                    "removed: legacy pre-LOD detached singleton/shard policy"
                )
            else:
                record["verdict"] = "kept: legacy pre-LOD policy found no shard evidence"

        if remove:
            drop |= members
            removed.append(record)
        else:
            kept.append(record)

    survivors = tris[~drop]
    if not len(survivors):
        raise RuntimeError("debris policy would remove the entire mesh")
    used = np.unique(survivors)
    remap = np.full(len(positions), -1, np.int64)
    remap[used] = np.arange(len(used))
    kept_uv = uv[used] if uv is not None else None
    output_path = Path(args.output)
    write_glb(output_path, positions[used], normals[used], kept_uv, remap[survivors])

    check_positions, _, check_uv, _ = read_glb(output_path)
    uv_identical = None if kept_uv is None else bool(np.array_equal(check_uv, kept_uv))
    positions_identical = bool(np.array_equal(check_positions, positions[used]))
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "policy": "explicit_source_supported" if context is not None else "legacy_pre_lod",
        "source_aware": context is not None,
        "source_path": None if context is None else str(context.source_path),
        "height_min": args.height_min,
        "outboard_min": OUTBOARD_MIN,
        "very_high_min": VERY_HIGH_MIN,
        "source_support_min": SOURCE_SUPPORT_MIN,
        "max_triangles": args.max_triangles,
        "max_diagonal": round(max_diagonal, 6),
        "min_separation_fraction": MIN_SEPARATION_FRACTION,
        "max_relative_component_faces": MAX_RELATIVE_COMPONENT_FACES,
                "max_relative_component_diagonal": MAX_RELATIVE_COMPONENT_DIAGONAL,
        "protected_components": sorted(protected_components),
        "components_total": int(len(sizes)),
        "components_removed": len(removed),
        "triangles_before": int(len(tris)),
        "triangles_after": int(len(survivors)),
        "triangles_removed": int(drop.sum()),
        "triangles_removed_percent": round(float(drop.sum() / len(tris) * 100), 6),
        "uv_bit_identical": uv_identical,
        "positions_bit_identical": positions_identical,
        "removed": sorted(removed, key=lambda row: -row["triangles"]),
        "kept_non_body": sorted(
            [row for row in kept if row["component"] != body],
            key=lambda row: -row["triangles"],
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DEBRIS_STRIP policy={report['policy']} removed={len(removed)} components/"
        f"{int(drop.sum())} triangles ({report['triangles_removed_percent']}%) "
        f"kept_detached={len(report['kept_non_body'])} uv_identical={uv_identical}",
        flush=True,
    )


if __name__ == "__main__":
    main()
