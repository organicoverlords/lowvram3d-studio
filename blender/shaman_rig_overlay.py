"""Milestone A: joint overlay render and rest-pose deformation validation.

Renders the skeleton over the mesh so observed and inferred joints can be judged
visually, and proves the rest pose is numerically unchanged by the armature
modifier (a bind that shifts the mesh at rest is broken before any clip exists).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402

VIEWS = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0, "three_quarter": 40.0}
REST_TOLERANCE = 1e-5


def evaluated_points(obj) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    buffer = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", buffer)
    points = buffer.reshape(-1, 3).copy()
    evaluated.to_mesh_clear()
    return points


def base_points(obj) -> np.ndarray:
    buffer = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", buffer)
    return buffer.reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.input)
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    mesh_obj = next(obj for obj in bpy.data.objects if obj.type == "MESH")
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")

    # Rest-pose deformation gate: with no action, the evaluated mesh must equal
    # the bind mesh. Any drift means the bind matrices are wrong.
    if armature.animation_data:
        armature.animation_data.action = None
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()

    rest = evaluated_points(mesh_obj)
    bind = base_points(mesh_obj)
    delta = np.linalg.norm(rest - bind, axis=1)
    max_delta = float(delta.max())
    finite = bool(np.isfinite(rest).all())

    failures = []
    if max_delta > REST_TOLERANCE:
        failures.append("REST_POSE_DEFORMATION_DRIFT")
    if not finite:
        failures.append("NONFINITE_REST_POSE_VERTICES")

    # Overlay render: armature drawn in front of the mesh.
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 800
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.55, 0.55, 0.58)
    shading.show_xray = True
    shading.xray_alpha = 0.28
    armature.show_in_front = True
    armature.data.display_type = "OCTAHEDRAL"

    # Blender does not render armatures in F12 output - they are viewport-only.
    # Joint markers must therefore be real geometry, sized by uncertainty radius
    # and coloured by confidence, or the overlay proves nothing.
    marker_collection = bpy.data.collections.new("joint_markers")
    scene.collection.children.link(marker_collection)
    marker_material = bpy.data.materials.new("joint_marker")
    marker_material.use_nodes = False
    for entry in landmarks["joints"]:
        position = Vector(entry["position"])
        confidence = float(entry["confidence"])
        radius = max(float(entry["uncertainty_radius"]) * 0.5, 0.012)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=position, segments=12, ring_count=8)
        marker = bpy.context.active_object
        marker.name = f"joint_{entry['name']}"
        for collection in list(marker.users_collection):
            collection.objects.unlink(marker)
        marker_collection.objects.link(marker)
        colour = marker.data.color_attributes.new(
            name="marker_colour", type="BYTE_COLOR", domain="POINT"
        )
        # Green = observed (>=0.6), red = inferred fallback.
        rgba = (0.15, 0.85, 0.25, 1.0) if confidence >= 0.6 else (0.95, 0.20, 0.15, 1.0)
        values = np.tile(np.array(rgba, dtype=np.float32), (len(marker.data.vertices), 1))
        colour.data.foreach_set("color", values.reshape(-1))
        marker.data.update()
    shading.color_type = "VERTEX"
    mesh_colour = mesh_obj.data.color_attributes.get("region_colour")
    if mesh_colour is None:
        layer = mesh_obj.data.color_attributes.new(
            name="overlay_grey", type="BYTE_COLOR", domain="POINT"
        )
        greys = np.tile(
            np.array([0.55, 0.55, 0.58, 1.0], dtype=np.float32),
            (len(mesh_obj.data.vertices), 1),
        )
        layer.data.foreach_set("color", greys.reshape(-1))
        mesh_obj.data.update()
        mesh_obj.data.color_attributes.active_color = layer

    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for corner in mesh_obj.bound_box:
        world = mesh_obj.matrix_world @ Vector(corner)
        for axis in range(3):
            minimum[axis] = min(minimum[axis], world[axis])
            maximum[axis] = max(maximum[axis], world[axis])
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 3.0 + 1.0
    ortho = max(size.x, size.z) * 1.12

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    renders = []
    for name, azimuth in VIEWS.items():
        data = bpy.data.cameras.new(f"cam_{name}")
        data.type = "ORTHO"
        data.ortho_scale = ortho
        camera = bpy.data.objects.new(f"cam_{name}", data)
        scene.collection.objects.link(camera)
        angle = math.radians(azimuth)
        camera.location = centre + Vector((math.sin(angle) * radius, -math.cos(angle) * radius, 0.0))
        camera.rotation_euler = (centre - camera.location).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        destination = output / f"overlay_{name}.png"
        scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        renders.append(str(destination))
        bpy.data.objects.remove(camera, do_unlink=True)

    observed = [j for j in landmarks["joints"] if j["confidence"] >= 0.6]
    inferred = [j for j in landmarks["joints"] if j["confidence"] < 0.6]

    report = {
        "stage": "MILESTONE_A_SOURCE_POSE_SKELETON",
        "passed": not failures,
        "failures": failures,
        "rest_pose": {
            "max_vertex_delta": max_delta,
            "tolerance": REST_TOLERANCE,
            "finite": finite,
            "vertex_count": int(rest.shape[0]),
        },
        "joint_overlay_renders": renders,
        "observed_joint_count": len(observed),
        "inferred_joint_count": len(inferred),
        "observed_joints": [j["name"] for j in observed],
        "inferred_joints": [
            {"name": j["name"], "confidence": j["confidence"], "source": j["source"]}
            for j in inferred
        ],
        "bone_count": len(armature.data.bones),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"REST_MAX_DELTA={max_delta:.3e}", flush=True)
    print(f"OVERLAY_RENDERS={len(renders)}", flush=True)
    print(f"OBSERVED_JOINTS={len(observed)} INFERRED_JOINTS={len(inferred)}", flush=True)
    if failures:
        print("MILESTONE_A_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)
    print("MILESTONE_A_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
