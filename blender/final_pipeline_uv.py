"""UV unwrap shaman LOD0 and validate it with exact positive-area overlap.

The acceptance gate deliberately does not rasterise every triangle into a 4K occupancy grid.
That approach is both slow on a 220k-triangle mesh and mathematically unsound as an overlap test:
shared chart edges and sub-texel boundary rounding distort the result. Atlas utilisation is
computed from analytic triangle area, while bake safety is decided only by exact positive-area
triangle intersections from ``lowvram3d.uv_overlap``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lowvram3d.uv_overlap import positive_area_uv_overlaps
from lowvram3d.anchor_provenance import AnchorProvenanceError, geometry_sha256, load_anchor_provenance, provenance_record


def weld(obj: bpy.types.Object, merge_distance: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.object.mode_set(mode="OBJECT")


def unwrap(obj: bpy.types.Object, angle_limit: float, island_margin: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=angle_limit,
        island_margin=island_margin,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(
        rotate=True,
        margin=island_margin,
        shape_method="CONCAVE",
    )
    bpy.ops.object.mode_set(mode="OBJECT")


def triangle_uvs(obj: bpy.types.Object) -> tuple[np.ndarray, int]:
    mesh = obj.data
    layer = mesh.uv_layers.active
    if layer is None:
        raise RuntimeError("LOD0 has no active UV layer after unwrap")

    loops = np.asarray([tuple(data.uv) for data in layer.data], dtype=np.float64)
    triangles: list[np.ndarray] = []
    non_triangles = 0
    for polygon in mesh.polygons:
        indices = list(polygon.loop_indices)
        if len(indices) != 3:
            non_triangles += 1
            continue
        triangles.append(loops[indices])

    if non_triangles:
        raise RuntimeError(
            f"LOD0 must be triangulated before UV validation; found {non_triangles} non-triangles"
        )
    if not triangles:
        raise RuntimeError("LOD0 contains no triangles")
    return np.asarray(triangles, dtype=np.float64), len(loops)


def uv_metrics(
    obj: bpy.types.Object,
    resolution: int,
    *,
    max_overlap_texels: float,
    overlap_timeout_seconds: float,
    max_candidate_pairs: int,
    min_utilisation: float,
) -> dict:
    triangles, loop_count = triangle_uvs(obj)
    finite = bool(np.isfinite(triangles).all())

    flat = triangles.reshape((-1, 2))
    umin, vmin = flat.min(axis=0)
    umax, vmax = flat.max(axis=0)

    signed_twice_area = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 2, 0] - triangles[:, 0, 0])
        * (triangles[:, 1, 1] - triangles[:, 0, 1])
    )
    triangle_area = np.abs(signed_twice_area) * 0.5
    analytic_area_fraction = float(triangle_area.sum())

    exact = positive_area_uv_overlaps(
        triangles,
        resolution,
        timeout_seconds=overlap_timeout_seconds,
        max_candidate_pairs=max_candidate_pairs,
    )
    exact_dict = exact.as_dict()

    overlap_area = float(exact.positive_overlap_total_area_uv)
    effective_unique_area = max(0.0, analytic_area_fraction - overlap_area)
    estimated_covered_texels = effective_unique_area * float(resolution * resolution)

    errors: list[str] = []
    warnings: list[str] = []
    if not finite:
        errors.append("non-finite UV coordinates present")
    if exact.errors:
        errors.extend(str(error) for error in exact.errors)
    if exact.out_of_bounds_triangle_count:
        errors.append(
            f"{exact.out_of_bounds_triangle_count} UV triangles are outside the atlas"
        )
    if exact.degenerate_uv_triangle_count:
        errors.append(
            f"{exact.degenerate_uv_triangle_count} UV triangles are degenerate"
        )
    if exact.positive_overlap_total_texels_equivalent > max_overlap_texels:
        errors.append(
            "positive-area UV overlap exceeds budget: "
            f"{exact.positive_overlap_total_texels_equivalent:.6f} > {max_overlap_texels:.6f} texels"
        )
    if effective_unique_area < min_utilisation:
        warnings.append(
            "atlas utilisation is below the preferred packing target: "
            f"{effective_unique_area * 100.0:.2f}% < {min_utilisation * 100.0:.2f}%"
        )

    overlap_fraction = overlap_area / max(analytic_area_fraction, 1e-12)
    return {
        "resolution": resolution,
        "uv_finite": finite,
        "uv_min": [float(umin), float(vmin)],
        "uv_max": [float(umax), float(vmax)],
        "uv_out_of_bounds": bool(exact.out_of_bounds_triangle_count),
        "triangles": int(len(triangles)),
        "uv_loops": int(loop_count),
        "degenerate_uv_triangles": int(exact.degenerate_uv_triangle_count),
        "analytic_uv_area_fraction": analytic_area_fraction,
        "atlas_utilisation": effective_unique_area,
        "estimated_covered_texels": estimated_covered_texels,
        "estimated_overlap_fraction": overlap_fraction,
        "positive_overlap_pair_count": int(exact.positive_overlap_pair_count),
        "positive_overlap_total_area_uv": overlap_area,
        "positive_overlap_total_texels_equivalent": float(
            exact.positive_overlap_total_texels_equivalent
        ),
        "positive_overlap_max_area_uv": float(exact.positive_overlap_max_area_uv),
        "exact_overlap": exact_dict,
        "max_overlap_texels": max_overlap_texels,
        "minimum_preferred_utilisation": min_utilisation,
        "gate_passed": not errors,
        "manual_review_required": bool(warnings),
        "errors": errors,
        "warnings": warnings,
        "deprecated_metrics": {
            "area_to_covered_ratio": None,
            "raster_collision_overlap": None,
            "reason": (
                "Removed from acceptance: analytic-area/raster-coverage ratios and raster "
                "collision counts conflate discretisation/shared edges with real overlap."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--angle-limit", type=float, default=1.15192)  # 66 degrees
    parser.add_argument("--island-margin", type=float, default=0.003)
    parser.add_argument("--merge-distance", type=float, default=1e-4)
    parser.add_argument("--max-overlap-texels", type=float, default=1.0)
    parser.add_argument("--overlap-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-candidate-pairs", type=int, default=10_000_000)
    parser.add_argument("--min-utilisation", type=float, default=0.35)
    parser.add_argument("--anchor-receipt", required=True)
    parser.add_argument("--expected-source-sha256", default="")
    args = parser.parse_args(argv_after_double_dash())

    try:
        _receipt, receipt_hash, anchor_ids = load_anchor_provenance(
            args.anchor_receipt, expected_source_sha256=args.expected_source_sha256 or None
        )
    except AnchorProvenanceError as exc:
        save_json(args.report, {"gate_passed": False, "failure_codes": [exc.code], "failure_detail": exc.detail})
        raise SystemExit(2)

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError(f"No mesh in {args.input}")
    if len(objects) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active or objects[0]

    weld(obj, args.merge_distance)
    source_vertices = np.asarray([tuple(v.co) for v in obj.data.vertices], dtype=np.float64)
    source_triangles = np.asarray(
        [[int(vertices[0]), int(vertices[i]), int(vertices[i + 1])]
         for polygon in obj.data.polygons
         for vertices in [[vertex for vertex in polygon.vertices]]
         for i in range(1, len(vertices) - 1)], dtype=np.int64
    )
    input_geometry_hash = geometry_sha256(source_vertices, source_triangles)
    unwrap(obj, args.angle_limit, args.island_margin)
    metrics = uv_metrics(
        obj,
        args.resolution,
        max_overlap_texels=args.max_overlap_texels,
        overlap_timeout_seconds=args.overlap_timeout_seconds,
        max_candidate_pairs=args.max_candidate_pairs,
        min_utilisation=args.min_utilisation,
    )
    metrics["input"] = args.input
    metrics["output"] = args.output
    metrics["angle_limit_radians"] = args.angle_limit
    metrics["island_margin"] = args.island_margin
    metrics["provenance"] = provenance_record(
        receipt_sha256=receipt_hash, anchor_ids=anchor_ids,
        input_geometry_sha256=input_geometry_hash,
        output_geometry_sha256=input_geometry_hash,
        geometry_unchanged=True,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
    )
    save_json(args.report, metrics)

    print(
        "UV_EXACT "
        f"gate={metrics['gate_passed']} "
        f"utilisation={metrics['atlas_utilisation'] * 100.0:.2f}% "
        f"overlap_pairs={metrics['positive_overlap_pair_count']} "
        f"overlap_texels={metrics['positive_overlap_total_texels_equivalent']:.6f} "
        f"degenerate={metrics['degenerate_uv_triangles']}",
        flush=True,
    )
    if not metrics["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
