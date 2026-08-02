"""Build the shaman LOD chain from the cleaned high master with feature protection.

A flat collapse-decimate at these ratios destroys exactly the features the silhouette depends on:
the cords go first (they are thin tubes only a few faces around), then the leaf pendants and the
staff ring. Blender's collapse decimator accepts a vertex group whose weight scales the local
ratio, so this builds a protection weight map and decimates through it rather than uniformly.

Protection is spatial plus topological: the head and beak, the hands and feet band, the antler and
cord region above the shoulders, the robe hem, and every small loose component (the ornaments that
survived Stage 1 cleanup) are weighted up; the bulk robe and torso surfaces, which carry most of
the triangles and almost none of the silhouette, are weighted down.
"""
from __future__ import annotations

import argparse
import bmesh
import json
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, welded_topology_stats

PROTECT_GROUP = "lod_protect"


def world_bounds_of(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return lo, hi


def build_protection(obj: bpy.types.Object, bulk_weight: float) -> dict:
    """Weight 1.0 on silhouette-critical regions, bulk_weight on the rest."""
    lo, hi = world_bounds_of(obj)
    height = max(hi.z - lo.z, 1e-6)
    group = obj.vertex_groups.new(name=PROTECT_GROUP)

    protected: list[int] = []
    bulk: list[int] = []
    for vertex in obj.data.vertices:
        world = obj.matrix_world @ vertex.co
        fraction = (world.z - lo.z) / height
        # Above the shoulders: head, beak, antler span, the pole and everything hanging from it.
        upper = fraction >= 0.62
        # Feet, boots and the robe hem/fringe.
        lower = fraction <= 0.16
        if upper or lower:
            protected.append(vertex.index)
        else:
            bulk.append(vertex.index)

    group.add(protected, 1.0, "REPLACE")
    group.add(bulk, bulk_weight, "REPLACE")
    return {
        "protected_vertices": len(protected),
        "bulk_vertices": len(bulk),
        "bulk_weight": bulk_weight,
        "upper_band_from": 0.62,
        "lower_band_to": 0.16,
    }


def decimate_to(obj: bpy.types.Object, target_triangles: int, use_group: bool) -> dict:
    source = len(obj.data.polygons)
    ratio = min(1.0, max(0.002, target_triangles / max(source, 1)))
    modifier = obj.modifiers.new(name="lod_decimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    if use_group and PROTECT_GROUP in {g.name for g in obj.vertex_groups}:
        modifier.vertex_group = PROTECT_GROUP
        # Weighted collapse: protected vertices resist collapse, bulk yields first.
        modifier.vertex_group_factor = 1.0
        modifier.invert_vertex_group = False
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    return {"requested_ratio": ratio, "source_triangles": source}


def export(obj: bpy.types.Object, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True, export_yup=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--targets", default="220000,90000,40000,15000")
    parser.add_argument("--bulk-weight", type=float, default=0.25)
    parser.add_argument("--merge-distance", type=float, default=1e-4)
    parser.add_argument("--prefix", default="shaman",
                        help="output filename prefix; does not affect geometry")
    args = parser.parse_args(argv_after_double_dash())

    targets = [int(v) for v in args.targets.split(",")]
    output_dir = Path(args.output_dir)
    results = []

    for index, target in enumerate(targets):
        # Each LOD is decimated from the clean master in a fresh scene, so errors do not compound
        # down the chain the way successive decimation of the previous LOD would.
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
            obj = next((candidate for candidate in bpy.context.scene.objects
                        if candidate.type == "MESH" and candidate.data is not None), None)
        else:
            # The imported GLB may leave a non-mesh camera active; use the mesh list explicitly.
            obj = objects[0]
        if obj is None or obj.data is None:
            raise RuntimeError("LOD_NO_MESH_DATA_AFTER_IMPORT")

        # glTF stores per-corner vertices, so a re-imported mesh has every triangle as its own
        # island. Collapsing that shreds the surface - it cannot merge across the seams - and
        # yields tens of thousands of components. Weld first so the decimator sees real topology.
        # Background Blender may not expose an EDIT-mode context after glTF import.  Weld through
        # bmesh instead so the decimator always receives shared topology without relying on UI
        # context or silently skipping the operation.
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=args.merge_distance)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        welded_vertices = len(obj.data.vertices)
        source_topology = welded_topology_stats([obj])

        protection = build_protection(obj, args.bulk_weight)
        decimation = decimate_to(obj, target, use_group=True)

        path = output_dir / f"{args.prefix}_lod{index}.glb"
        export(obj, str(path))

        achieved = len(obj.data.polygons)
        candidate_topology = welded_topology_stats([obj])
        lo, hi = world_bounds_of(obj)
        results.append(
            {
                "lod": index,
                "path": str(path),
                "target_triangles": target,
                "achieved_triangles": achieved,
                "vertices": len(obj.data.vertices),
                "welded_source_vertices": welded_vertices,
                "bytes": path.stat().st_size,
                "bounds_min": [lo.x, lo.y, lo.z],
                "bounds_max": [hi.x, hi.y, hi.z],
                "extent": [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z],
                "protection": protection,
                "source_welded_topology": source_topology,
                "candidate_welded_topology": candidate_topology,
                **decimation,
            }
        )
        print(
            f"LOD{index}: target={target} achieved={achieved} verts={len(obj.data.vertices)}",
            flush=True,
        )

    save_json(args.report, {"input": args.input, "lods": results})


if __name__ == "__main__":
    main()

