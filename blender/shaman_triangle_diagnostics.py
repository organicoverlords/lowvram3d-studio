"""V3 triangle-boundary diagnostics and semantic region overlays.

Measures deformed triangles, not just displaced vertices. A rigid garment panel
moves every one of its vertices together, so a vertex-only pass reports it as
healthy; only the triangles spanning the panel boundary reveal it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402
from shaman_semantic_v3 import (  # noqa: E402
    CLASS_ORDER,
    cross_region_triangles,
    region_summary,
    semantic_masks_v3,
    triangle_metrics,
    vertex_classes,
)
from shaman_weight_diagnostics import VIEWS, mesh_points  # noqa: E402

EPSILON = 1e-4
ISOLATED = (
    ("clavicle", "clavicle_r", (0.0, 0.0, -25.0)),
    ("upperarm", "upperarm_r", (-12.0, 0.0, -45.0)),
    ("elbow", "lowerarm_r", (-55.0, 0.0, 0.0)),
    ("wrist", "hand_r", (0.0, 0.0, -45.0)),
    ("sleeve_bones", "sleeve_r_drape_01", (-30.0, 0.0, -15.0)),
)

PALETTE = {
    "staff": (1.00, 1.00, 1.00),
    "torso_core": (0.55, 0.30, 0.85),
    "rear_cape": (0.95, 0.35, 0.10),
    "cape_r": (0.95, 0.10, 0.55),
    "cape_l": (0.60, 0.10, 0.40),
    "arm_core_r": (0.10, 0.95, 0.25),
    "hand_core_r": (1.00, 0.95, 0.15),
    "sleeve_anchor_r": (0.20, 0.75, 0.95),
    "sleeve_drape_upper_r": (0.10, 0.35, 0.90),
    "sleeve_drape_lower_r": (0.05, 0.15, 0.55),
    "arm_core_l": (0.30, 0.60, 0.30),
    "hand_core_l": (0.60, 0.60, 0.20),
    "sleeve_anchor_l": (0.25, 0.45, 0.55),
    "sleeve_drape_upper_l": (0.15, 0.25, 0.45),
    "sleeve_drape_lower_l": (0.08, 0.12, 0.30),
    "hanging_accessories": (0.20, 0.85, 0.85),
}


def evaluated_points(obj) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    buffer = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", buffer)
    points = buffer.reshape(-1, 3).copy()
    evaluated.to_mesh_clear()
    return points


def triangle_array(obj) -> np.ndarray:
    obj.data.calc_loop_triangles()
    count = len(obj.data.loop_triangles)
    buffer = np.empty(count * 3, dtype=np.int32)
    obj.data.loop_triangles.foreach_get("vertices", buffer)
    return buffer.reshape(-1, 3)


def paint(obj, colours: np.ndarray, name: str) -> None:
    layer = obj.data.color_attributes.get(name)
    if layer is None:
        layer = obj.data.color_attributes.new(name=name, type="BYTE_COLOR", domain="POINT")
    layer.data.foreach_set("color", colours.reshape(-1))
    obj.data.color_attributes.active_color = layer
    obj.data.update()


def render_views(obj, output: Path, prefix: str, views) -> list[str]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 460
    scene.render.resolution_y = 580
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "VERTEX"
    shading.show_cavity = False

    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for corner in obj.bound_box:
        world = obj.matrix_world @ Vector(corner)
        for axis in range(3):
            minimum[axis] = min(minimum[axis], world[axis])
            maximum[axis] = max(maximum[axis], world[axis])
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 3.0 + 1.0

    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name in views:
        data = bpy.data.cameras.new(f"c_{name}")
        data.type = "ORTHO"
        data.ortho_scale = max(size.x, size.z) * 1.12
        camera = bpy.data.objects.new(f"c_{name}", data)
        scene.collection.objects.link(camera)
        angle = math.radians(VIEWS[name])
        camera.location = centre + Vector((math.sin(angle) * radius, -math.cos(angle) * radius, 0.0))
        camera.rotation_euler = (centre - camera.location).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        destination = output / f"{prefix}_{name}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        written.append(str(destination))
        bpy.data.objects.remove(camera, do_unlink=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.input)
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and not o.hide_viewport)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    if armature.animation_data:
        armature.animation_data.action = None

    points = mesh_points(obj)
    count = points.shape[0]
    staff_group = obj.vertex_groups.get("staff_deform")
    staff_mask = np.zeros(count, dtype=bool)
    if staff_group is not None:
        index = staff_group.index
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                if element.group == index and element.weight > 0.5:
                    staff_mask[vertex.index] = True
                    break

    masks = semantic_masks_v3(points, landmarks, staff_mask)
    classes = vertex_classes(masks, count)
    triangles = triangle_array(obj)

    output = Path(args.output_dir)
    colours = np.tile(np.array([0.30, 0.30, 0.34, 1.0], dtype=np.float32), (count, 1))
    for index, name in enumerate(CLASS_ORDER):
        if name in PALETTE:
            colours[classes == index, :3] = PALETTE[name]
    paint(obj, colours, "semantic_v3")
    region_renders = render_views(obj, output, f"semantic_regions_{args.label}",
                                  ["front", "back", "three_quarter", "side"])

    cross = cross_region_triangles(classes, triangles)
    mixed_mask = cross.pop("mixed_mask")
    mixed_vertices = np.zeros(count, dtype=bool)
    mixed_vertices[triangles[mixed_mask].reshape(-1)] = True
    overlay = np.tile(np.array([0.28, 0.28, 0.32, 1.0], dtype=np.float32), (count, 1))
    overlay[mixed_vertices, :3] = (1.0, 0.15, 0.05)
    paint(obj, overlay, "cross_region_v3")
    cross_renders = render_views(obj, output, f"cross_region_triangles_{args.label}",
                                 ["front", "back", "three_quarter", "side"])

    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()
    rest = evaluated_points(obj)

    drape_r = masks["sleeve_drape_r"]
    tests = []
    for label, bone_name, degrees in ISOLATED:
        if bone_name not in armature.pose.bones:
            tests.append({"test": label, "bone": bone_name, "skipped": "BONE_ABSENT"})
            continue
        for bone in armature.pose.bones:
            bone.matrix_basis.identity()
        pose_bone = armature.pose.bones[bone_name]
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = Euler(
            [math.radians(v) for v in degrees], "XYZ"
        ).to_quaternion()
        bpy.context.view_layer.update()
        posed = evaluated_points(obj)

        displacement = np.linalg.norm(posed - rest, axis=1)
        moved = displacement > EPSILON
        regions = {}
        for name in CLASS_ORDER:
            mask = masks.get(name)
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                continue
            regions[name] = {
                "region_vertices": int(mask.sum()),
                "displaced": int((mask & moved).sum()),
                "max_displacement": float(displacement[mask].max()) if mask.any() else 0.0,
            }
        tests.append({
            "test": label,
            "bone": bone_name,
            "rotation_degrees": list(degrees),
            "displaced_vertices": int(moved.sum()),
            "max_displacement": float(displacement.max()),
            "finite": bool(np.isfinite(posed).all()),
            "sleeve_drape_r_displaced": int((drape_r & moved).sum()),
            "regions": regions,
            "triangles": triangle_metrics(rest, posed, triangles),
        })

    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()

    def by_name(name):
        return next((t for t in tests if t["test"] == name and "skipped" not in t), None)

    arm_tests = [t for t in tests if t["test"] in {"clavicle", "upperarm", "elbow", "wrist"}]
    wrist = by_name("wrist")
    elbow = by_name("elbow")
    gates = {
        "torso_core_displaced_by_arm": max(t["regions"]["torso_core"]["displaced"] for t in arm_tests),
        "rear_cape_displaced_by_arm": max(t["regions"]["rear_cape"]["displaced"] for t in arm_tests),
        "side_cape_displaced_by_arm": max(
            t["regions"]["cape_r"]["displaced"] + t["regions"]["cape_l"]["displaced"]
            for t in arm_tests
        ),
        "staff_displaced_by_arm": max(t["regions"]["staff"]["displaced"] for t in arm_tests),
        "drape_displaced_by_wrist": wrist["sleeve_drape_r_displaced"] if wrist else -1,
        "drape_displaced_by_elbow": elbow["sleeve_drape_r_displaced"] if elbow else -1,
        "drape_total": int(drape_r.sum()),
        "flipped_normals": max(t["triangles"]["flipped_normals"] for t in arm_tests),
        "inverted_triangles": max(t["triangles"]["inverted_triangles"] for t in arm_tests),
        "degenerate_introduced": max(t["triangles"]["degenerate_introduced"] for t in arm_tests),
        "extreme_stretch_triangles": max(t["triangles"]["extreme_stretch_triangles"] for t in arm_tests),
        "max_edge_stretch": max(t["triangles"]["max_edge_stretch"] for t in arm_tests),
        "all_finite": all(t["finite"] for t in arm_tests),
    }

    report = {
        "stage": "TRIANGLE_AND_SEMANTIC_V3_DIAGNOSTICS",
        "label": args.label,
        "input": args.input,
        "vertex_count": count,
        "triangle_count": int(triangles.shape[0]),
        "region_sizes": region_summary(masks, count),
        "cross_region_triangles": cross,
        "isolated_tests": tests,
        "gates": gates,
        "renders": {"semantic_regions": region_renders, "cross_region": cross_renders},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"REGIONS={json.dumps(report['region_sizes'])}", flush=True)
    print(f"MIXED_TRIANGLES={cross['mixed_triangles']} ratio={cross['mixed_ratio']:.5f}", flush=True)
    for test in tests:
        if "skipped" in test:
            print(f"{args.label} {test['test']:13s} SKIPPED {test['skipped']}", flush=True)
            continue
        tri = test["triangles"]
        print(
            f"{args.label} {test['test']:13s} moved={test['displaced_vertices']:6d} "
            f"drape={test['sleeve_drape_r_displaced']:5d} "
            f"stretch={tri['max_edge_stretch']:.3f} flips={tri['flipped_normals']} "
            f"degen={tri['degenerate_introduced']}",
            flush=True,
        )
    print(f"{args.label}_GATES={json.dumps(gates)}", flush=True)


if __name__ == "__main__":
    main()
