"""Dissolve degenerate sliver triangles from a LOD before unwrapping.

The UV gate requires zero degenerate UV triangles, but the degeneracy originates in 3D: collapse
decimation leaves a handful of near-zero-area slivers whose UV area necessarily falls below the
detector's epsilon no matter which unwrapper runs. On LOD0 these are 65 triangles carrying
0.0004% of the surface area, so dissolving them is below any visible threshold while being the
only way the gate can be satisfied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--merge-distance", type=float, default=1e-4)
    parser.add_argument("--degenerate-threshold", type=float, default=1e-4)
    args = parser.parse_args(argv_after_double_dash())

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

    before_faces = len(obj.data.polygons)
    before_vertices = len(obj.data.vertices)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=args.merge_distance)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=args.degenerate_threshold)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode="OBJECT")

    after_faces = len(obj.data.polygons)
    after_vertices = len(obj.data.vertices)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=args.output, export_format="GLB", use_selection=True, export_yup=True
    )

    save_json(
        args.report,
        {
            "input": args.input,
            "output": args.output,
            "merge_distance": args.merge_distance,
            "degenerate_threshold": args.degenerate_threshold,
            "faces_before": before_faces,
            "faces_after": after_faces,
            "faces_removed": before_faces - after_faces,
            "vertices_before": before_vertices,
            "vertices_after": after_vertices,
        },
    )
    print(
        f"DESLIVER faces {before_faces} -> {after_faces} "
        f"(removed {before_faces - after_faces}), verts {before_vertices} -> {after_vertices}",
        flush=True,
    )


if __name__ == "__main__":
    main()
