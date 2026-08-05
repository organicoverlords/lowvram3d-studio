"""Bake a Z-up orientation into a delivered GLB, preserving geometry, UVs and material exactly.

The raster route inherits its orientation from the marching-cubes proxy, which lands the subject
with its long axis on X and an identity object rotation -- so the asset is not merely displayed
sideways, its vertex data is sideways, and every consumer (viewer, engine import, thumbnailer) has
to compensate. Rotating the object and applying the transform fixes it at the source.

This is a rigid rotation only: no vertex is added, removed, merged or scaled, the UV layer is
untouched, and the packed base-colour texture is carried through unchanged. Triangle count before
and after is asserted equal, so the stage cannot quietly become a geometry edit.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json


def world_bounds(objects: list) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([1e30, 1e30, 1e30])
    hi = -lo
    for obj in objects:
        for corner in obj.bound_box:
            world = np.array(obj.matrix_world @ Vector(corner))
            lo = np.minimum(lo, world)
            hi = np.maximum(hi, world)
    return lo, hi


def rotation_to_z_up(up_axis: int) -> Matrix:
    """Rigid rotation mapping the given world axis onto +Z."""
    if up_axis == 0:
        return Matrix.Rotation(math.radians(-90.0), 4, "Y")   # +X -> +Z
    if up_axis == 1:
        return Matrix.Rotation(math.radians(90.0), 4, "X")    # +Y -> +Z
    return Matrix.Identity(4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh imported")

    triangles_before = sum(
        max(len(polygon.vertices) - 2, 0) for obj in objects for polygon in obj.data.polygons
    )
    vertices_before = sum(len(obj.data.vertices) for obj in objects)
    uv_layers_before = {obj.name: [layer.name for layer in obj.data.uv_layers] for obj in objects}

    lo, hi = world_bounds(objects)
    dimensions_before = hi - lo
    up_axis = int(np.argmax(dimensions_before))
    rotation = rotation_to_z_up(up_axis)

    for obj in objects:
        obj.matrix_world = rotation @ obj.matrix_world

    # Bake the rotation into the mesh data so the file itself is upright, rather than relying on a
    # node transform that some importers drop or flatten.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    triangles_after = sum(
        max(len(polygon.vertices) - 2, 0) for obj in objects for polygon in obj.data.polygons
    )
    vertices_after = sum(len(obj.data.vertices) for obj in objects)
    uv_layers_after = {obj.name: [layer.name for layer in obj.data.uv_layers] for obj in objects}
    if triangles_after != triangles_before or vertices_after != vertices_before:
        raise RuntimeError(
            f"Reorientation changed geometry: triangles {triangles_before}->{triangles_after}, "
            f"vertices {vertices_before}->{vertices_after}"
        )
    if uv_layers_after != uv_layers_before:
        raise RuntimeError(f"Reorientation changed UV layers: {uv_layers_before} -> {uv_layers_after}")

    lo_after, hi_after = world_bounds(objects)
    dimensions_after = hi_after - lo_after

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output, selected_only=False)

    if args.report:
        save_json(args.report, {
            "success": True,
            "input": str(args.input),
            "output": str(args.output),
            "detected_up_axis": "XYZ"[up_axis],
            "rotation_applied": up_axis != 2,
            "dimensions_before": [round(float(v), 6) for v in dimensions_before],
            "dimensions_after": [round(float(v), 6) for v in dimensions_after],
            "up_axis_after": "XYZ"["XYZ".index("XYZ"[int(np.argmax(dimensions_after))])],
            "triangles": triangles_after,
            "vertices": vertices_after,
            "geometry_preserved": True,
        })

    print(
        f"REORIENT_UPRIGHT up={'XYZ'[up_axis]}->Z triangles={triangles_after} "
        f"dims={np.round(dimensions_after, 4).tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
