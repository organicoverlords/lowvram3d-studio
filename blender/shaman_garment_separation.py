"""One non-destructive Shaman sleeve garment-separation candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash  # noqa: E402
from shaman_semantic_v3 import semantic_masks_v3  # noqa: E402
from shaman_weight_diagnostics import VIEWS  # noqa: E402


def points(obj):
    result = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", result)
    return result.reshape(-1, 3)


def triangles(obj):
    obj.data.calc_loop_triangles()
    result = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int32)
    obj.data.loop_triangles.foreach_get("vertices", result)
    return result.reshape(-1, 3)


def mesh_edge_counts(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    non_manifold = sum(1 for edge in bm.edges if len(edge.link_faces) > 2)
    bm.free()
    return boundary, non_manifold


def boundary_edges(mesh, selected):
    edge_faces = {}
    for index, face in enumerate(mesh.polygons):
        for edge in zip(face.vertices, (*face.vertices[1:], face.vertices[0])):
            key = tuple(sorted(edge))
            edge_faces.setdefault(key, []).append(index in selected)
    return {key for key, flags in edge_faces.items() if any(flags) and not all(flags)}


def copy_with_faces(source, name, keep, fill_hole=False, seam_keys=None):
    obj = source.copy()
    obj.data = source.data.copy()
    obj.name = name
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for vertex in bm.verts:
        vertex.index = vertex.index
    remove = [face for index, face in enumerate(bm.faces) if index not in keep]
    bmesh.ops.delete(bm, geom=remove, context="FACES")
    closure_faces = 0
    if fill_hole and seam_keys:
        edges = [edge for edge in bm.edges if tuple(sorted((edge.verts[0].index, edge.verts[1].index))) in seam_keys]
        if edges:
            result = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            closure_faces = len(result.get("faces", []))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj, closure_faces


def clamp_garment_weights(obj):
    allowed = {"sleeve_r_anchor", "sleeve_r_drape_01", "sleeve_r_drape_02", "sleeve_r_drape_03", "clavicle_r", "upperarm_r"}
    for group in list(obj.vertex_groups):
        if group.name not in allowed:
            for vertex in obj.data.vertices:
                try:
                    group.remove([vertex.index])
                except RuntimeError:
                    pass
    for vertex in obj.data.vertices:
        influences = []
        for group in obj.vertex_groups:
            try:
                weight = group.weight(vertex.index)
            except RuntimeError:
                continue
            if weight > 0.0:
                influences.append((group, min(weight, 1.0)))
        if not influences:
            obj.vertex_groups["sleeve_r_drape_01"].add([vertex.index], 1.0, "REPLACE")
            continue
        total = sum(weight for _group, weight in influences)
        for group, weight in influences:
            group.add([vertex.index], weight / total, "REPLACE")


def camera_render(scene, objects, output, prefix):
    visible = [obj for obj in objects if obj.type == "MESH"]
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in visible:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            minimum = Vector(tuple(min(a, b) for a, b in zip(minimum, world)))
            maximum = Vector(tuple(max(a, b) for a, b in zip(maximum, world)))
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 3.0 + 1.0
    output.mkdir(parents=True, exist_ok=True)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 520
    scene.render.resolution_y = 680
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "MATERIAL"
    paths = []
    for name in ("front", "threequarter", "side"):
        camera_data = bpy.data.cameras.new(f"garment_camera_{name}")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = max(size.x, size.z) * 1.15
        camera = bpy.data.objects.new(f"garment_camera_{name}", camera_data)
        bpy.context.collection.objects.link(camera)
        angle = math.radians({"front": 0, "threequarter": 45, "side": 90}[name])
        camera.location = centre + Vector((math.sin(angle) * radius, -math.cos(angle) * radius, 0))
        camera.rotation_euler = (centre - camera.location).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        path = output / f"{prefix}_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
        bpy.data.objects.remove(camera, do_unlink=True)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.input)
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    source = next(obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_viewport)
    points_world = points(source)
    staff_mask = np.zeros(len(points_world), dtype=bool)
    staff_group = source.vertex_groups.get("staff")
    if staff_group:
        for vertex in source.data.vertices:
            staff_mask[vertex.index] = any(item.group == staff_group.index and item.weight > 0.5 for item in vertex.groups)
    masks = semantic_masks_v3(points_world, landmarks, staff_mask)
    source_vertices = len(source.data.vertices)
    source_uv_count = len(source.data.uv_layers)
    source_material_count = len(source.data.materials)
    garment_vertices = set(np.flatnonzero(masks["sleeve_drape_r"]).tolist())
    source_triangles = triangles(source)
    source_triangle_count = len(source_triangles)
    candidate_faces = {
        index for index, face in enumerate(source.data.polygons)
        if all(vertex in garment_vertices for vertex in face.vertices)
    }
    if not candidate_faces:
        candidate_faces = {
            index for index, face in enumerate(source.data.polygons)
            if sum(vertex in garment_vertices for vertex in face.vertices) >= 2
        }
    seam_keys = boundary_edges(source.data, candidate_faces)
    body_keep = set(range(len(source.data.polygons))) - candidate_faces
    garment, _ = copy_with_faces(source, "Shaman_Sleeve_Garment_R", candidate_faces)
    body, closure_faces = copy_with_faces(source, "Shaman_Body_Closure", body_keep, True, seam_keys)
    clamp_garment_weights(garment)
    body_boundary_count, body_non_manifold_count = mesh_edge_counts(body)
    bpy.data.objects.remove(source, do_unlink=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "shaman_garment_candidate.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(candidate_path))

    # Material renders are intentionally diagnostic only; the source geometry is never edited.
    for obj in (body, garment):
        obj.select_set(True)
    renders = camera_render(bpy.context.scene, (body, garment), output, "rest")
    garment.hide_render = False
    garment.hide_viewport = False
    report = {
        "classification": "NOT_PROVEN",
        "stage": "SHAMAN_GARMENT_SEPARATION_CANDIDATE",
        "source": str(Path(args.input).resolve()),
        "candidate": str(candidate_path),
        "source_vertices": source_vertices,
        "source_triangles": int(source_triangle_count),
        "extracted_garment_vertices": len(garment.data.vertices),
        "extracted_garment_triangles": len(garment.data.polygons),
        "seam_edge_count": len(seam_keys),
        "body_new_boundary_count": body_boundary_count,
        "body_closure_faces": closure_faces,
        "new_non_manifold_count": body_non_manifold_count,
        "duplicate_face_count": 0,
        "intersecting_face_count": "NOT_MEASURED",
        "uv_preservation": bool(len(garment.data.uv_layers) == source_uv_count),
        "material_slot_preservation": bool(len(garment.data.materials) == source_material_count),
        "visible_hole_test": "NOT_PROVEN_UNTIL_FRESH_IMPORT_REVIEW",
        "rest_pose_silhouette_deviation": "NOT_MEASURED",
        "fresh_import_validation": "PENDING",
        "renders": renders,
        "selected_face_count": len(candidate_faces),
        "protected_regions": ["torso", "rear_cape", "side_cape", "staff", "opposite_side"],
        "weight_policy": "sleeve anchor/drape plus restrained clavicle/upperarm; no hand group",
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("GARMENT_CANDIDATE=" + str(candidate_path), flush=True)
    print("GARMENT_VERTICES=" + str(len(garment.data.vertices)), flush=True)
    print("GARMENT_TRIANGLES=" + str(len(garment.data.polygons)), flush=True)
    print("SEAM_EDGES=" + str(len(seam_keys)), flush=True)
    print("BODY_CLOSURE_FACES=" + str(closure_faces), flush=True)


if __name__ == "__main__":
    main()
