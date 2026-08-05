"""Semantic weight diagnostics: heat maps and isolated-rotation displacement tests.

Runs against any rigged .blend so the same measurements describe the state
before and after a weight repair. Every displacement figure is attributed to a
semantic region, because a global "max displacement" number cannot distinguish
a sleeve following the arm from a cape slab following the arm.
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

EPSILON = 1e-4
HEAT_BONES = (
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "chest", "spine_01", "spine_02",
    "cloth_r_01", "cloth_r_02", "cloth_r_03",
    "cloth_b_01", "cloth_b_02", "cloth_b_03",
)
ISOLATED = (
    ("clavicle", "clavicle_r", (0.0, 0.0, -25.0)),
    ("upperarm", "upperarm_r", (-12.0, 0.0, -45.0)),
    ("elbow", "lowerarm_r", (-55.0, 0.0, 0.0)),
    ("wrist", "hand_r", (0.0, 0.0, -45.0)),
)
VIEWS = {"front": 0.0, "back": 180.0, "three_quarter": 40.0, "side": 90.0}


def mesh_points(obj) -> np.ndarray:
    count = len(obj.data.vertices)
    buffer = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", buffer)
    return buffer.reshape(-1, 3)


def evaluated_points(obj) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    buffer = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", buffer)
    points = buffer.reshape(-1, 3).copy()
    evaluated.to_mesh_clear()
    return points


def semantic_masks(points: np.ndarray, landmarks: dict, staff_mask: np.ndarray) -> dict:
    """Hard semantic classification used for every gate in this stage."""

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    symmetry = landmarks["symmetry_plane_x"]
    torso_y = landmarks["torso_centreline_y"]
    hip_z = landmarks["hip_z"]
    neck_z = landmarks["neck_z"]
    chest_z = landmarks["chest_z"]
    shoulder_z = landmarks["shoulder_z"]
    core = landmarks.get("torso_core") or {}
    core_low = core.get("low", symmetry - 0.35)
    core_high = core.get("high", symmetry + 0.35)

    joints = {j["name"]: np.array(j["position"]) for j in landmarks["joints"]}

    def chain_distance(side: str) -> np.ndarray:
        segments = [
            (joints[f"upperarm_{side}"], joints[f"lowerarm_{side}"]),
            (joints[f"lowerarm_{side}"], joints[f"hand_{side}"]),
        ]
        best = np.full(points.shape[0], np.inf)
        for a, b in segments:
            ab = b - a
            length_sq = float(ab @ ab)
            if length_sq < 1e-12:
                distance = np.linalg.norm(points - a, axis=1)
            else:
                t = np.clip(((points - a) @ ab) / length_sq, 0.0, 1.0)
                distance = np.linalg.norm(points - (a[None, :] + t[:, None] * ab[None, :]), axis=1)
            best = np.minimum(best, distance)
        return best

    sleeve_radius = float(landmarks["body_height_ground_to_skull"]) * 0.058
    masks = {}

    # Order matters. torso_core and rear_cape are defined purely geometrically
    # so they cannot be reshaped by whatever the sleeve happens to cover; the
    # sleeve is then defined by exclusion from them. Defining the sleeve first
    # and subtracting it from the cape would let a bad sleeve mask silently
    # shrink the very regions the gates are supposed to protect.
    masks["torso_core"] = (
        (x >= core_low) & (x <= core_high) & (z >= hip_z) & (z <= neck_z) & ~staff_mask
    )
    masks["rear_cape"] = (
        (y > torso_y + 0.02) & (z >= hip_z - 0.35) & (z <= neck_z) & ~staff_mask
    )

    for side, sign in (("l", -1.0), ("r", 1.0)):
        outside_core = (x - (core_high if side == "r" else core_low)) * sign > 0.0
        near_chain = chain_distance(side) <= sleeve_radius
        band = (z >= joints[f"hand_{side}"][2] - 0.06) & (z <= shoulder_z + 0.05)
        masks[f"sleeve_{side}"] = (
            near_chain & outside_core & band
            & ~staff_mask & ~masks["torso_core"] & ~masks["rear_cape"]
        )
        hand = joints[f"hand_{side}"]
        masks[f"hand_{side}_region"] = (
            (np.linalg.norm(points - hand, axis=1) <= sleeve_radius * 1.5)
            & ~staff_mask & ~masks["torso_core"] & ~masks["rear_cape"]
        )

    # side_cape means "lateral cloth that is not arm". The hand sphere is part of
    # the arm classification, so it is excluded here too - otherwise a handful of
    # genuine hand vertices would be counted as cape and the gate would measure
    # something other than what it claims.
    masks["side_cape"] = (
        (((x < core_low) | (x > core_high)) & (z >= hip_z - 0.35) & (z <= shoulder_z))
        & ~masks["sleeve_l"] & ~masks["sleeve_r"]
        & ~masks["hand_l_region"] & ~masks["hand_r_region"]
        & ~masks["rear_cape"] & ~staff_mask
    )
    masks["hanging_accessories"] = (z > chest_z) & (np.abs(x - symmetry) > 0.42) & ~staff_mask
    masks["staff"] = staff_mask
    masks["opposite_side"] = (x - symmetry < 0.0) & ~staff_mask
    masks["sleeve_radius"] = sleeve_radius
    return masks


def bake_heat(obj, group_name: str) -> None:
    layer = obj.data.color_attributes.get("weight_heat")
    if layer is None:
        layer = obj.data.color_attributes.new(
            name="weight_heat", type="BYTE_COLOR", domain="POINT"
        )
    group = obj.vertex_groups.get(group_name)
    weights = np.zeros(len(obj.data.vertices), dtype=np.float64)
    if group is not None:
        index = group.index
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                if element.group == index:
                    weights[vertex.index] = element.weight
                    break
    colours = np.zeros((len(obj.data.vertices), 4), dtype=np.float32)
    colours[:, 3] = 1.0
    colours[:, 0] = weights                       # red rises with weight
    colours[:, 2] = np.clip(1.0 - weights * 3.0, 0.0, 1.0)
    colours[:, 1] = np.clip(0.28 - weights * 0.28, 0.0, 1.0)
    layer.data.foreach_set("color", colours.reshape(-1))
    obj.data.color_attributes.active_color = layer
    obj.data.update()


def setup_render(vertex_colour: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 460
    scene.render.resolution_y = 580
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "FLAT" if vertex_colour else "STUDIO"
    shading.color_type = "VERTEX" if vertex_colour else "SINGLE"
    shading.single_color = (0.62, 0.62, 0.64)
    shading.show_cavity = not vertex_colour


def render(obj, output: Path, prefix: str, views) -> list[str]:
    scene = bpy.context.scene
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
    parser.add_argument("--render-heat", action="store_true")
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.input)
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    obj = next(o for o in bpy.data.objects if o.type == "MESH" and not o.hide_viewport)
    armature = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    if armature.animation_data:
        armature.animation_data.action = None

    points = mesh_points(obj)
    staff_group = obj.vertex_groups.get("staff_deform")
    staff_mask = np.zeros(points.shape[0], dtype=bool)
    if staff_group is not None:
        index = staff_group.index
        for vertex in obj.data.vertices:
            for element in vertex.groups:
                if element.group == index and element.weight > 0.5:
                    staff_mask[vertex.index] = True
                    break

    masks = semantic_masks(points, landmarks, staff_mask)
    region_names = [k for k, v in masks.items() if isinstance(v, np.ndarray)]

    output = Path(args.output_dir)
    heat_renders = {}
    if args.render_heat:
        setup_render(vertex_colour=True)
        for bone in HEAT_BONES:
            if bone in obj.vertex_groups:
                bake_heat(obj, bone)
                heat_renders[bone] = render(
                    obj, output / "heat", f"heat_{args.label}_{bone}",
                    ["front", "back", "three_quarter"],
                )

    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()
    rest = evaluated_points(obj)

    tests = []
    for label, bone_name, degrees in ISOLATED:
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

        per_region = {}
        for name in region_names:
            mask = masks[name]
            selected = mask & moved
            per_region[name] = {
                "region_vertices": int(mask.sum()),
                "displaced_vertices": int(selected.sum()),
                "max_displacement": float(displacement[mask].max()) if mask.any() else 0.0,
                "mean_displacement": float(displacement[selected].mean()) if selected.any() else 0.0,
            }

        tests.append({
            "test": label,
            "bone": bone_name,
            "rotation_degrees": list(degrees),
            "displaced_vertices": int(moved.sum()),
            "max_displacement": float(displacement.max()),
            "finite": bool(np.isfinite(posed).all()),
            "regions": per_region,
        })

    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()

    def region_value(test, region, key):
        return test["regions"].get(region, {}).get(key, 0)

    gates = {
        "staff_displaced_by_free_arm": max(
            region_value(t, "staff", "displaced_vertices") for t in tests
        ),
        "torso_core_displaced_by_lowerarm_or_hand": max(
            region_value(t, "torso_core", "displaced_vertices")
            for t in tests if t["test"] in {"elbow", "wrist"}
        ),
        "rear_cape_displaced_by_lowerarm_or_hand": max(
            region_value(t, "rear_cape", "displaced_vertices")
            for t in tests if t["test"] in {"elbow", "wrist"}
        ),
        "side_cape_displaced_by_upperarm": region_value(
            next(t for t in tests if t["test"] == "upperarm"), "side_cape", "displaced_vertices"
        ),
        "opposite_side_displaced": max(
            region_value(t, "opposite_side", "displaced_vertices") for t in tests
        ),
        "all_finite": all(t["finite"] for t in tests),
    }
    failures = []
    if gates["staff_displaced_by_free_arm"] > 0:
        failures.append("STAFF_DISPLACED_BY_FREE_ARM")
    if gates["torso_core_displaced_by_lowerarm_or_hand"] > 0:
        failures.append("TORSO_CORE_DISPLACED_BY_LOWERARM_OR_HAND")
    if gates["rear_cape_displaced_by_lowerarm_or_hand"] > 0:
        failures.append("REAR_CAPE_DISPLACED_BY_LOWERARM_OR_HAND")
    if gates["opposite_side_displaced"] > 0:
        failures.append("OPPOSITE_SIDE_DISPLACED")
    if not gates["all_finite"]:
        failures.append("NONFINITE_TRANSFORMS")

    report = {
        "stage": "SEMANTIC_WEIGHT_DIAGNOSTICS",
        "label": args.label,
        "input": args.input,
        "passed": not failures,
        "failures": failures,
        "epsilon": EPSILON,
        "sleeve_radius": masks["sleeve_radius"],
        "region_sizes": {n: int(masks[n].sum()) for n in region_names},
        "gates": gates,
        "isolated_tests": tests,
        "heat_renders": heat_renders,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    for test in tests:
        regions = test["regions"]
        print(
            f"{args.label} {test['test']:9s} moved={test['displaced_vertices']:7d} "
            f"torso={regions['torso_core']['displaced_vertices']:6d} "
            f"rear_cape={regions['rear_cape']['displaced_vertices']:6d} "
            f"side_cape={regions['side_cape']['displaced_vertices']:6d} "
            f"sleeve_r={regions['sleeve_r']['displaced_vertices']:5d} "
            f"staff={regions['staff']['displaced_vertices']:4d} "
            f"opp={regions['opposite_side']['displaced_vertices']:6d}",
            flush=True,
        )
    print(f"{args.label}_FAILURES=" + (",".join(failures) if failures else "none"), flush=True)


if __name__ == "__main__":
    main()
