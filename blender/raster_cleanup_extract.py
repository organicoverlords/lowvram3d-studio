"""Geometry repair gate and extraction for the fast raster texture route.

The previous implementation received a GLB whose UV/split-normal vertices looked disconnected and
made deletion decisions before reconstructing topology. That removed valid surface patches and
created thousands of open edges. This stage now:

1. welds coincident vertices before component analysis;
2. applies an explicit, testable cleanup policy;
3. fills only tiny, unambiguous boundary loops;
4. rejects any cleanup that increases open-boundary damage;
5. exports/extracts only after the geometry gate passes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from common import argv_after_double_dash, import_mesh, reset_scene, save_json
from lowvram3d.geometry_quality import ComponentMetrics, decide_component, topology_gate

VIEWS = {
    "front": (0.0, -3.0, 0.0),
    "right": (3.0, 0.0, 0.0),
    "back": (0.0, 3.0, 0.0),
    "left": (-3.0, 0.0, 0.0),
}
ORTHO_SCALE = 2.6
WELD_DISTANCE = 4e-4
ATTACH_DISTANCE_FRACTION = 0.006
MAX_HOLE_EDGES = 12
MAX_HOLE_DIAMETER_FRACTION = 0.015
MAX_HOLE_PERIMETER_FRACTION = 0.05


def component_faces(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMFace]]:
    bm.faces.ensure_lookup_table()
    seen: set[int] = set()
    components: list[list[bmesh.types.BMFace]] = []
    for face in bm.faces:
        if face.index in seen:
            continue
        stack = [face]
        seen.add(face.index)
        members: list[bmesh.types.BMFace] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for edge in current.edges:
                for neighbour in edge.link_faces:
                    if neighbour.index not in seen:
                        seen.add(neighbour.index)
                        stack.append(neighbour)
        components.append(members)
    return components


def unique_component_vertices(faces: list[bmesh.types.BMFace]) -> list[bmesh.types.BMVert]:
    return list({vertex for face in faces for vertex in face.verts})


def world_bounds(points: list[Vector]) -> tuple[Vector, Vector]:
    return (
        Vector(tuple(min(point[axis] for point in points) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in points) for axis in range(3))),
    )


def world_face_area(face: bmesh.types.BMFace, matrix) -> float:
    vertices = [matrix @ vertex.co for vertex in face.verts]
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    area = 0.0
    for index in range(1, len(vertices) - 1):
        area += ((vertices[index] - origin).cross(vertices[index + 1] - origin)).length * 0.5
    return area


def build_component_bvh(component: dict) -> BVHTree | None:
    vertices: list[Vector] = []
    polygons: list[tuple[int, int, int]] = []
    index_by_vertex: dict[int, int] = {}
    matrix = component["object"].matrix_world
    for face in component["faces"]:
        polygon: list[int] = []
        for vertex in face.verts:
            key = id(vertex)
            if key not in index_by_vertex:
                index_by_vertex[key] = len(vertices)
                vertices.append(matrix @ vertex.co)
            polygon.append(index_by_vertex[key])
        if len(polygon) == 3:
            polygons.append(tuple(polygon))
    return BVHTree.FromPolygons(vertices, polygons) if vertices and polygons else None


def topology_counts(bm: bmesh.types.BMesh) -> dict[str, int]:
    return {
        "faces": len(bm.faces),
        "components": len(component_faces(bm)),
        "boundary_edges": sum(1 for edge in bm.edges if len(edge.link_faces) == 1),
        "non_manifold_edges": sum(1 for edge in bm.edges if not edge.is_manifold),
    }


def boundary_groups(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMEdge]]:
    boundary = {edge for edge in bm.edges if len(edge.link_faces) == 1}
    groups: list[list[bmesh.types.BMEdge]] = []
    while boundary:
        first = boundary.pop()
        stack = [first]
        group = [first]
        while stack:
            edge = stack.pop()
            neighbours = {
                linked
                for vertex in edge.verts
                for linked in vertex.link_edges
                if linked in boundary
            }
            for neighbour in neighbours:
                boundary.remove(neighbour)
                stack.append(neighbour)
                group.append(neighbour)
        groups.append(group)
    return groups


def fill_small_holes(bm: bmesh.types.BMesh, matrix, scene_diag: float) -> list[dict]:
    filled: list[dict] = []
    for edges in boundary_groups(bm):
        vertices = list({vertex for edge in edges for vertex in edge.verts})
        degrees = {
            vertex: sum(1 for edge in edges if vertex in edge.verts)
            for vertex in vertices
        }
        if not vertices or any(degree != 2 for degree in degrees.values()):
            continue
        world = [matrix @ vertex.co for vertex in vertices]
        low, high = world_bounds(world)
        diameter_fraction = (high - low).length / max(scene_diag, 1e-9)
        perimeter = sum((matrix @ edge.verts[0].co - matrix @ edge.verts[1].co).length for edge in edges)
        perimeter_fraction = perimeter / max(scene_diag, 1e-9)
        if (
            len(edges) > MAX_HOLE_EDGES
            or diameter_fraction > MAX_HOLE_DIAMETER_FRACTION
            or perimeter_fraction > MAX_HOLE_PERIMETER_FRACTION
        ):
            continue
        result = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
        new_faces = list(result.get("faces", []))
        if new_faces:
            filled.append(
                {
                    "edge_count": len(edges),
                    "new_faces": len(new_faces),
                    "diameter_fraction": round(diameter_fraction, 6),
                    "perimeter_fraction": round(perimeter_fraction, 6),
                }
            )
    return filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--cleanup-mode",
        choices=("conservative", "single_subject_strict"),
        default="conservative",
    )
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh imported")

    scene_points = [obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]
    if not scene_points:
        raise RuntimeError("Imported mesh contains no vertices")
    scene_low, scene_high = world_bounds(scene_points)
    scene_diag = max((scene_high - scene_low).length, 1e-9)
    attach_distance = scene_diag * ATTACH_DISTANCE_FRACTION

    per_object: list[dict] = []
    all_components: list[dict] = []
    faces_before = 0
    boundary_before = 0
    total_area = 0.0

    for object_index, obj in enumerate(objects):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=WELD_DISTANCE)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        before = topology_counts(bm)
        faces_before += before["faces"]
        boundary_before += before["boundary_edges"]
        components = component_faces(bm)
        per_object.append({"object": obj, "bm": bm, "components": components, "before": before})
        for component_index, faces in enumerate(components):
            vertices = unique_component_vertices(faces)
            world = [obj.matrix_world @ vertex.co for vertex in vertices]
            low, high = world_bounds(world)
            area = sum(world_face_area(face, obj.matrix_world) for face in faces)
            total_area += area
            all_components.append(
                {
                    "object_index": object_index,
                    "component_index": component_index,
                    "object": obj,
                    "faces": faces,
                    "vertices": vertices,
                    "world_vertices": world,
                    "face_count": len(faces),
                    "area": area,
                    "extent": (high - low).length,
                }
            )

    if not all_components:
        raise RuntimeError("No connected components found after topology weld")

    main_component = max(all_components, key=lambda component: component["face_count"])
    main_bvh = build_component_bvh(main_component)
    decisions: list[dict] = []
    keep_keys: set[tuple[int, int]] = set()

    for component in all_components:
        key = (component["object_index"], component["component_index"])
        is_main = component is main_component
        sample = component["world_vertices"][:: max(1, len(component["world_vertices"]) // 64)]
        distances = []
        if main_bvh and not is_main:
            for point in sample:
                nearest = main_bvh.find_nearest(point)
                if nearest and nearest[3] is not None:
                    distances.append(float(nearest[3]))
        contact_ratio = (
            sum(distance <= attach_distance for distance in distances) / len(distances)
            if distances
            else (1.0 if is_main else 0.0)
        )
        nearest_fraction = min(distances) / scene_diag if distances else (0.0 if is_main else 1.0)
        metrics = ComponentMetrics(
            face_count=component["face_count"],
            face_fraction=component["face_count"] / max(faces_before, 1),
            area_fraction=component["area"] / max(total_area, 1e-12),
            extent_fraction=component["extent"] / scene_diag,
            contact_ratio=contact_ratio,
            nearest_distance_fraction=nearest_fraction,
            is_main=is_main,
        )
        decision = decide_component(metrics, args.cleanup_mode)
        if not decision.removable:
            keep_keys.add(key)
        decisions.append(
            {
                "object": component["object"].name,
                "component_id": component["component_index"],
                "faces": component["face_count"],
                "face_fraction": round(metrics.face_fraction, 6),
                "area_fraction": round(metrics.area_fraction, 6),
                "extent_fraction": round(metrics.extent_fraction, 6),
                "contact_ratio": round(metrics.contact_ratio, 6),
                "nearest_distance_fraction": round(metrics.nearest_distance_fraction, 6),
                "action": decision.action,
                "reason": decision.reason,
            }
        )

    object_reports: list[dict] = []
    hole_fills: list[dict] = []
    removed_faces = 0
    faces_after = 0
    boundary_after = 0

    for object_index, item in enumerate(per_object):
        obj = item["object"]
        bm = item["bm"]
        doomed = [
            face
            for component_index, faces in enumerate(item["components"])
            if (object_index, component_index) not in keep_keys
            for face in faces
        ]
        removed_faces += len(doomed)
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context="FACES")
            loose_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
            if loose_vertices:
                bmesh.ops.delete(bm, geom=loose_vertices, context="VERTS")

        tiny_fills = fill_small_holes(bm, obj.matrix_world, scene_diag)
        for fill in tiny_fills:
            hole_fills.append({"object": obj.name, **fill})
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        after = topology_counts(bm)
        faces_after += after["faces"]
        boundary_after += after["boundary_edges"]
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        object_reports.append({"object": obj.name, "before": item["before"], "after": after})

    for obj in list(objects):
        if len(obj.data.polygons) == 0:
            bpy.data.objects.remove(obj, do_unlink=True)
    objects = [obj for obj in objects if obj.name in bpy.data.objects and len(obj.data.polygons) > 0]

    gate_passed, gate_errors = topology_gate(
        faces_before=faces_before,
        faces_after=faces_after,
        boundary_before=boundary_before,
        boundary_after=boundary_after,
        mode=args.cleanup_mode,
    )
    report = {
        "success": gate_passed,
        "policy_version": 2,
        "cleanup_mode": args.cleanup_mode,
        "weld_distance": WELD_DISTANCE,
        "scene_diagonal": scene_diag,
        "faces_before": faces_before,
        "faces_after": faces_after,
        "removed_faces": removed_faces,
        "removed_faces_percent": round(removed_faces / max(faces_before, 1) * 100.0, 4),
        "components_before": len(all_components),
        "components_kept": len(keep_keys),
        "components_removed": len(all_components) - len(keep_keys),
        "boundary_edges_before": boundary_before,
        "boundary_edges_after": boundary_after,
        "small_holes_filled": len(hole_fills),
        "hole_fills": hole_fills,
        "gate_errors": gate_errors,
        "objects": object_reports,
        "components": decisions,
    }
    save_json(args.report, report)
    if not gate_passed:
        raise RuntimeError(f"Geometry cleanup gate failed: {gate_errors}")

    output_glb = Path(args.output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output_glb), export_format="GLB", use_selection=False)

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    uvs: list[list[list[float]]] = []
    normals: list[list[float]] = []
    base = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        matrix = obj.matrix_world
        normal_matrix = matrix.to_3x3().inverted_safe().transposed()
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            raise RuntimeError(f"{obj.name} has no UV layer for raster projection")
        vertices.extend([list(matrix @ vertex.co) for vertex in obj.data.vertices])
        for triangle in obj.data.loop_triangles:
            triangles.append([base + index for index in triangle.vertices])
            uvs.append([list(uv_layer.data[loop_index].uv) for loop_index in triangle.loops])
            normals.append(list((normal_matrix @ triangle.normal).normalized()))
        base += len(obj.data.vertices)

    vertices_array = np.asarray(vertices, np.float32)
    triangles_array = np.asarray(triangles, np.int32)
    uvs_array = np.asarray(uvs, np.float32)
    normals_array = np.asarray(normals, np.float32)
    centroids = vertices_array[triangles_array].mean(axis=1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    visibility: dict[str, np.ndarray] = {}
    for name, location in VIEWS.items():
        camera = Vector(location)
        visible = np.zeros(len(triangles_array), dtype=bool)
        for index in range(len(triangles_array)):
            origin = Vector(centroids[index].tolist()) + Vector(normals_array[index].tolist()) * 1e-3
            direction = camera - origin
            hit, *_ = scene.ray_cast(
                depsgraph,
                origin,
                direction.normalized(),
                distance=direction.length,
            )
            visible[index] = not hit
        visibility[name] = visible

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        verts=vertices_array,
        tris=triangles_array,
        uvs=uvs_array,
        normals=normals_array,
        view_names=np.array(list(VIEWS)),
        view_locs=np.asarray(list(VIEWS.values()), np.float32),
        ortho_scale=np.float32(ORTHO_SCALE),
        **{f"vis_{name}": visibility[name] for name in VIEWS},
    )
    print(
        "GEOMETRY_REPAIR_EXTRACT "
        f"mode={args.cleanup_mode} faces={faces_before}->{faces_after} "
        f"components={len(all_components)}->{len(keep_keys)} "
        f"boundary={boundary_before}->{boundary_after} holes_filled={len(hole_fills)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
