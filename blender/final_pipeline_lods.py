"""Build bounded production LODs with generic view-aware feature protection."""
from __future__ import annotations

import argparse
import bmesh
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, welded_topology_stats

PROTECT_GROUP = "lod_protect"


def world_bounds_of(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return lo, hi


def _axis(value: str, default: int) -> int:
    if value is None or value == "":
        return default
    value = str(value).lower()
    if value in "xyz":
        return "xyz".index(value)
    parsed = int(value)
    if parsed not in (0, 1, 2):
        raise ValueError(f"invalid axis {value}")
    return parsed


def _world_positions(obj: bpy.types.Object) -> np.ndarray:
    raw = np.empty(len(obj.data.vertices) * 3, np.float64)
    obj.data.vertices.foreach_get("co", raw)
    local = raw.reshape(-1, 3)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _silhouette_vertices(points: np.ndarray, up_axis: int, right_axis: int,
                         front_axis: int, bins: int = 384) -> np.ndarray:
    """Return vertices near the projected boundary in six fixed orthographic views."""
    depth_axis = front_axis
    vertical = points[:, up_axis]
    vmin, vmax = float(vertical.min()), float(vertical.max())
    vspan = max(vmax - vmin, 1e-9)
    bin_ids = np.clip(((vertical - vmin) / vspan * (bins - 1)).astype(np.int32), 0, bins - 1)
    marked = np.zeros(len(points), dtype=bool)
    global_horizontal_span = max(float(points[:, right_axis].ptp()),
                                 float(points[:, depth_axis].ptp()), 1e-9)
    tolerance = global_horizontal_span * 0.012
    for right_cos, depth_sin in (
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.766044, 0.642788), (-0.766044, -0.642788),
    ):
        horizontal = points[:, right_axis] * right_cos + points[:, depth_axis] * depth_sin
        low = np.full(bins, np.inf, dtype=np.float64)
        high = np.full(bins, -np.inf, dtype=np.float64)
        np.minimum.at(low, bin_ids, horizontal)
        np.maximum.at(high, bin_ids, horizontal)
        marked |= np.abs(horizontal - low[bin_ids]) <= tolerance
        marked |= np.abs(horizontal - high[bin_ids]) <= tolerance
    return marked


def build_protection(obj: bpy.types.Object, bulk_weight: float, *, up_axis: int,
                     right_axis: int, front_axis: int) -> dict:
    """Weight silhouette, topology boundaries, sharp regions, and thin extrema."""
    points = _world_positions(obj)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    height = max(float(hi[up_axis] - lo[up_axis]), 1e-6)
    silhouette = _silhouette_vertices(points, up_axis, right_axis, front_axis)
    boundary = np.zeros(len(points), dtype=bool)
    sharp = np.zeros(len(points), dtype=bool)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for edge in bm.edges:
        if edge.is_boundary or len(edge.link_faces) != 2:
            for vert in edge.verts:
                boundary[vert.index] = True
        elif edge.link_faces:
            a, b = edge.link_faces[0].normal, edge.link_faces[1].normal
            if a.dot(b) < 0.90:
                for vert in edge.verts:
                    sharp[vert.index] = True
    bm.free()

    fraction = (points[:, up_axis] - lo[up_axis]) / height
    extrema = (fraction <= 0.08) | (fraction >= 0.92)
    protected_mask = silhouette | boundary | sharp | extrema
    group = obj.vertex_groups.new(name=PROTECT_GROUP)
    protected = np.flatnonzero(protected_mask).tolist()
    bulk = np.flatnonzero(~protected_mask).tolist()
    group.add(protected, 1.0, "REPLACE")
    group.add(bulk, bulk_weight, "REPLACE")
    return {
        "protected_vertices": len(protected),
        "bulk_vertices": len(bulk),
        "bulk_weight": bulk_weight,
        "silhouette_vertices": int(silhouette.sum()),
        "boundary_vertices": int(boundary.sum()),
        "sharp_vertices": int(sharp.sum()),
        "extreme_vertices": int(extrema.sum()),
        "views": ["front", "rear", "left", "right", "front_three_quarter", "rear_three_quarter"],
        "up_axis": "xyz"[up_axis],
        "right_axis": "xyz"[right_axis],
        "front_axis": "xyz"[front_axis],
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
    # Blender's collapse modifier can leave exported loop normals from the pre-collapse mesh.
    # Rebuild face winding/normals before QA so the visual gate measures the reduced geometry,
    # rather than stale or flipped shading data. This does not change positions, indices, or the
    # triangle budget.
    mesh = obj.data
    mesh.validate(verbose=False, clean_customdata=False)
    mesh.update()
    normal_bmesh = bmesh.new()
    normal_bmesh.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(normal_bmesh, faces=list(normal_bmesh.faces))
    normal_bmesh.normal_update()
    normal_bmesh.to_mesh(mesh)
    normal_bmesh.free()
    mesh.update()
    return {"requested_ratio": ratio, "source_triangles": source, "normals_recalculated": True}


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
    parser.add_argument("--up-axis", default="z")
    parser.add_argument("--right-axis", default="x")
    parser.add_argument("--front-axis", default="y")
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

        up_axis = _axis(args.up_axis, 2)
        right_axis = _axis(args.right_axis, 0)
        front_axis = _axis(args.front_axis, 1)
        if len({up_axis, right_axis, front_axis}) != 3:
            raise RuntimeError("LOD axes must be distinct")
        protection = build_protection(
            obj, args.bulk_weight, up_axis=up_axis, right_axis=right_axis,
            front_axis=front_axis,
        )
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

