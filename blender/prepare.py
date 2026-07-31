from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import bmesh

from common import (
    argv_after_double_dash,
    export_glb,
    import_mesh,
    join_meshes,
    mesh_objects,
    mesh_stats,
    normalize_scene,
    reset_scene,
    save_json,
    select_only,
)


def remove_tiny_loose_parts(obj: bpy.types.Object, minimum_fraction: float) -> int:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    components: list[list[bmesh.types.BMVert]] = []
    unseen = set(bm.verts)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        group = [seed]
        while stack:
            vert = stack.pop()
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other in unseen:
                    unseen.remove(other)
                    group.append(other)
                    stack.append(other)
        components.append(group)
    largest = max((len(group) for group in components), default=0)
    delete_verts = [vert for group in components if len(group) < largest * minimum_fraction for vert in group]
    count = len(delete_verts)
    if delete_verts:
        bmesh.ops.delete(bm, geom=delete_verts, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()
    return count


def smart_uv(obj: bpy.types.Object) -> None:
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    select_only([obj])
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.02, area_weight=0.0, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def decimate(obj: bpy.types.Object, target_faces: int) -> None:
    current = len(obj.data.polygons)
    if current <= target_faces:
        return
    ratio = max(0.01, min(1.0, target_faces / current))
    modifier = obj.modifiers.new("GameMeshDecimate", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    select_only([obj])
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--target-faces", type=int, default=50000)
    parser.add_argument("--tiny-part-fraction", type=float, default=0.0005)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    imported = import_mesh(args.input)
    before = mesh_stats(imported)
    obj = join_meshes(imported)
    removed = remove_tiny_loose_parts(obj, args.tiny_part_fraction)
    normalize = normalize_scene([obj])
    select_only([obj])
    try:
        bpy.ops.object.shade_smooth_by_angle()
    except Exception:
        bpy.ops.object.shade_smooth()
    decimate(obj, args.target_faces)
    smart_uv(obj)
    after = mesh_stats(mesh_objects())
    export_glb(args.output)
    save_json(args.stats, {"before": before, "after": after, "removed_vertices": removed, "normalization": normalize})


if __name__ == "__main__":
    main()
