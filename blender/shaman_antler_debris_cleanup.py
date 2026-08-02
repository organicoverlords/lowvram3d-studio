"""Stage 1: remove floating debris above the antlers on the textured shaman.

Fail-closed component classifier. A component is removed only when every debris
condition holds at once: it is not the protected core, it sits inside the ROI
above the antler base, it is small in both triangle count and bounding box, and
it is physically separated from the protected core by a clear gap. Anything that
fails one condition is kept and reported as ambiguous.

The source asset is opened read-only; results are written to a fresh run root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, reset_scene  # noqa: E402

# Thresholds fixed before the run, expressed as fractions of model height.
ROI_START_FRACTION = 0.84        # antler base; everything above is inspected
MAX_DEBRIS_TRIANGLES = 600
MAX_DEBRIS_EXTENT = 0.060        # bbox longest side / model height
MIN_SEPARATION = 0.004           # gap to protected core / model height
VIEWS = {"front": 0.0, "right": 90.0, "back": 180.0, "three_quarter": 40.0}


def components(mesh: bmesh.types.BMesh) -> list[list[int]]:
    mesh.verts.ensure_lookup_table()
    seen = np.zeros(len(mesh.verts), dtype=bool)
    groups = []
    for index in range(len(mesh.verts)):
        if seen[index]:
            continue
        stack = [mesh.verts[index]]
        seen[index] = True
        members = []
        while stack:
            vertex = stack.pop()
            members.append(vertex.index)
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other is not None and not seen[other.index]:
                    seen[other.index] = True
                    stack.append(other)
        groups.append(members)
    return groups


def render_views(obj_names, output_dir: Path, prefix: str, highlight=None) -> list[str]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.show_cavity = True
    shading.color_type = "VERTEX" if highlight else "SINGLE"
    shading.single_color = (0.62, 0.62, 0.64)

    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for name in obj_names:
        obj = bpy.data.objects[name]
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], world[axis])
                maximum[axis] = max(maximum[axis], world[axis])
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 3.0 + 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, azimuth in VIEWS.items():
        data = bpy.data.cameras.new(f"cam_{name}")
        data.type = "ORTHO"
        data.ortho_scale = max(size.x, size.z) * 1.12
        camera = bpy.data.objects.new(f"cam_{name}", data)
        scene.collection.objects.link(camera)
        angle = math.radians(azimuth)
        camera.location = centre + Vector((math.sin(angle) * radius, -math.cos(angle) * radius, 0.0))
        camera.rotation_euler = (centre - camera.location).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        destination = output_dir / f"{prefix}_{name}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        written.append(str(destination))
        bpy.data.objects.remove(camera, do_unlink=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one textured mesh, found {len(meshes)}")
    obj = meshes[0]

    before = {
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "materials": [m.name for m in obj.data.materials if m],
        "uv_layers": [layer.name for layer in obj.data.uv_layers],
    }

    render_dir = Path(args.render_dir)
    before_renders = render_views([obj.name], render_dir, "antler_debris_before")

    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    groups = components(mesh)
    coordinates = np.array([list(v.co) for v in mesh.verts], dtype=np.float64)

    z = coordinates[:, 2]
    height = float(z.max() - z.min())
    roi_z = float(z.min()) + height * ROI_START_FRACTION

    sizes = [len(group) for group in groups]
    core_index = int(np.argmax(sizes))
    core_vertices = groups[core_index]

    tree = KDTree(len(core_vertices))
    for position, index in enumerate(core_vertices):
        tree.insert(Vector(coordinates[index]), position)
    tree.balance()

    records = []
    debris_vertices: set[int] = set()
    for index, group in enumerate(groups):
        block = coordinates[group]
        extent = float((block.max(axis=0) - block.min(axis=0)).max())
        centroid = block.mean(axis=0)
        triangles = len({face.index for vertex in group for face in mesh.verts[vertex].link_faces})
        separation = min(tree.find(Vector(point))[2] for point in block) if index != core_index else 0.0

        conditions = {
            "is_protected_core": index == core_index,
            "inside_roi": bool(block[:, 2].min() >= roi_z),
            "small_triangle_count": bool(triangles <= MAX_DEBRIS_TRIANGLES),
            "small_extent": bool(extent <= height * MAX_DEBRIS_EXTENT),
            "separated_from_core": bool(separation >= height * MIN_SEPARATION),
        }
        is_debris = (
            not conditions["is_protected_core"]
            and conditions["inside_roi"]
            and conditions["small_triangle_count"]
            and conditions["small_extent"]
            and conditions["separated_from_core"]
        )
        ambiguous = (
            not conditions["is_protected_core"]
            and not is_debris
            and conditions["inside_roi"]
        )
        record = {
            "component": index,
            "vertex_count": len(group),
            "triangle_count": triangles,
            "extent": extent,
            "extent_over_height": extent / max(height, 1e-9),
            "centroid": [float(v) for v in centroid],
            "z_min": float(block[:, 2].min()),
            "separation_from_core": float(separation),
            "separation_over_height": float(separation / max(height, 1e-9)),
            "conditions": conditions,
            "classification": (
                "protected_core" if conditions["is_protected_core"]
                else "debris_removed" if is_debris
                else "ambiguous_kept" if ambiguous
                else "outside_roi_kept"
            ),
        }
        records.append(record)
        if is_debris:
            debris_vertices.update(group)

    # Highlight pass: colour debris red on the pre-cleanup mesh.
    colour = obj.data.color_attributes.get("debris_overlay")
    if colour is None:
        colour = obj.data.color_attributes.new(
            name="debris_overlay", type="BYTE_COLOR", domain="POINT"
        )
    colours = np.tile(
        np.array([0.62, 0.62, 0.64, 1.0], dtype=np.float32), (len(obj.data.vertices), 1)
    )
    if debris_vertices:
        colours[sorted(debris_vertices), :3] = (0.95, 0.10, 0.10)
    colour.data.foreach_set("color", colours.reshape(-1))
    obj.data.color_attributes.active_color = colour
    obj.data.update()
    overlay_renders = render_views([obj.name], render_dir, "antler_debris_removed_overlay")

    removed_triangles = sum(r["triangle_count"] for r in records if r["classification"] == "debris_removed")
    if debris_vertices:
        bmesh.ops.delete(
            mesh,
            geom=[mesh.verts[i] for i in sorted(debris_vertices)],
            context="VERTS",
        )
        mesh.to_mesh(obj.data)
        obj.data.update()
    mesh.free()

    after = {
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "materials": [m.name for m in obj.data.materials if m],
        "uv_layers": [layer.name for layer in obj.data.uv_layers],
    }
    after_renders = render_views([obj.name], render_dir, "antler_debris_after")

    debris_components = [r for r in records if r["classification"] == "debris_removed"]
    ambiguous_components = [r for r in records if r["classification"] == "ambiguous_kept"]

    failures = []
    if after["materials"] != before["materials"]:
        failures.append("MATERIAL_SLOTS_CHANGED")
    if after["uv_layers"] != before["uv_layers"]:
        failures.append("UV_LAYERS_CHANGED")
    removed_ratio = (before["polygons"] - after["polygons"]) / max(before["polygons"], 1)
    if removed_ratio > 0.01:
        failures.append("REMOVED_MORE_THAN_ONE_PERCENT_OF_GEOMETRY")

    report = {
        "stage": "ANTLER_DEBRIS_CLEANUP",
        "passed": not failures and bool(debris_components),
        "failures": failures,
        "source": args.input,
        "source_modified": False,
        "source_is_rejected_visual_baseline": True,
        "thresholds": {
            "roi_start_fraction": ROI_START_FRACTION,
            "max_debris_triangles": MAX_DEBRIS_TRIANGLES,
            "max_debris_extent_over_height": MAX_DEBRIS_EXTENT,
            "min_separation_over_height": MIN_SEPARATION,
        },
        "model_height": height,
        "roi_z": roi_z,
        "component_count": len(groups),
        "protected_core_component": core_index,
        "protected_core_vertices": len(core_vertices),
        "debris_component_count": len(debris_components),
        "ambiguous_kept_count": len(ambiguous_components),
        "removed_vertices": before["vertices"] - after["vertices"],
        "removed_triangles": removed_triangles,
        "removed_polygon_ratio": removed_ratio,
        "before": before,
        "after": after,
        "renders": {
            "before": before_renders,
            "after": after_renders,
            "overlay": overlay_renders,
        },
        "components": sorted(
            [r for r in records if r["classification"] != "protected_core"],
            key=lambda r: -r["vertex_count"],
        )[:60],
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"COMPONENTS={len(groups)}", flush=True)
    print(f"DEBRIS_COMPONENTS={len(debris_components)}", flush=True)
    print(f"AMBIGUOUS_KEPT={len(ambiguous_components)}", flush=True)
    print(f"REMOVED_VERTICES={report['removed_vertices']}", flush=True)
    print(f"REMOVED_TRIANGLES={removed_triangles}", flush=True)
    print(f"REMOVED_RATIO={removed_ratio:.6f}", flush=True)

    if failures:
        print("CLEANUP_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.output_glb, export_format="GLB", use_selection=False)
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)
    print("CLEANUP_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
