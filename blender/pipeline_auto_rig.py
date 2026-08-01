"""Conservative humanoid auto-rig for Pipeline V2.

Runs only after the separate rig-readiness gate proves limb clearance.  It creates a compact game
skeleton, applies Blender automatic weights, validates coverage and exports a fresh GLB.  Unsupported
profiles fail closed rather than receiving a human skeleton by accident.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json, world_bounds

SUPPORTED = {"humanoid", "humanoid_complex_accessories"}


def bone(edit_bones, name: str, head: Vector, tail: Vector, parent=None, connected=False):
    item = edit_bones.new(name)
    item.head = head
    item.tail = tail
    item.parent = parent
    item.use_connect = bool(parent and connected)
    return item


def build_armature(minimum: Vector, maximum: Vector):
    centre = (minimum + maximum) * 0.5
    width = max(maximum.x - minimum.x, 1e-4)
    depth = max(maximum.y - minimum.y, 1e-4)
    height = max(maximum.z - minimum.z, 1e-4)

    data = bpy.data.armatures.new("AssetSkeleton")
    armature = bpy.data.objects.new("AssetSkeleton", data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = data.edit_bones

    x, y, z = centre.x, centre.y, minimum.z
    root = bone(eb, "root", Vector((x, y, z)), Vector((x, y, z + height * 0.06)))
    pelvis = bone(eb, "pelvis", Vector((x, y, z + height * 0.39)),
                  Vector((x, y, z + height * 0.49)), root)
    spine = bone(eb, "spine_01", pelvis.tail, Vector((x, y, z + height * 0.60)), pelvis, True)
    chest = bone(eb, "spine_02", spine.tail, Vector((x, y, z + height * 0.70)), spine, True)
    neck = bone(eb, "neck", chest.tail, Vector((x, y, z + height * 0.77)), chest, True)
    head = bone(eb, "head", neck.tail, Vector((x, y, z + height * 0.90)), neck, True)

    shoulder_z = z + height * 0.69
    elbow_z = z + height * 0.54
    wrist_z = z + height * 0.42
    for side, sign in (("L", 1.0), ("R", -1.0)):
        clavicle = bone(eb, f"clavicle.{side}", Vector((x, y, shoulder_z)),
                        Vector((x + sign * width * 0.16, y, shoulder_z)), chest)
        upper = bone(eb, f"upper_arm.{side}", clavicle.tail,
                     Vector((x + sign * width * 0.30, y, elbow_z)), clavicle, True)
        forearm = bone(eb, f"forearm.{side}", upper.tail,
                       Vector((x + sign * width * 0.36, y, wrist_z)), upper, True)
        bone(eb, f"hand.{side}", forearm.tail,
             Vector((x + sign * width * 0.39, y - depth * 0.01, z + height * 0.37)), forearm, True)

    hip_z = z + height * 0.40
    knee_z = z + height * 0.22
    ankle_z = z + height * 0.055
    for side, sign in (("L", 1.0), ("R", -1.0)):
        thigh = bone(eb, f"thigh.{side}", Vector((x + sign * width * 0.08, y, hip_z)),
                     Vector((x + sign * width * 0.09, y, knee_z)), pelvis)
        shin = bone(eb, f"shin.{side}", thigh.tail,
                    Vector((x + sign * width * 0.09, y, ankle_z)), thigh, True)
        bone(eb, f"foot.{side}", shin.tail,
             Vector((x + sign * width * 0.09, y - depth * 0.18, z + height * 0.025)), shin, True)

    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--minimum-weight-coverage", type=float, default=0.995)
    args = parser.parse_args(argv_after_double_dash())

    if args.profile not in SUPPORTED:
        save_json(args.report, {"passed": False, "profile": args.profile,
                                "failures": ["unsupported rig profile"]})
        raise SystemExit(2)

    reset_scene()
    meshes = import_mesh(args.input)
    if not meshes:
        raise RuntimeError(f"no mesh imported from {args.input}")
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        meshes = [bpy.context.view_layer.objects.active]
    mesh = meshes[0]
    minimum, maximum = world_bounds(meshes)
    armature = build_armature(minimum, maximum)

    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    failures = []
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc:
        failures.append(f"automatic weights failed: {exc}")

    weighted = 0
    if not failures:
        for vertex in mesh.data.vertices:
            if any(group.weight > 1e-6 for group in vertex.groups):
                weighted += 1
    coverage = weighted / max(len(mesh.data.vertices), 1)
    if coverage < args.minimum_weight_coverage:
        failures.append(
            f"weight coverage {coverage:.6f} below {args.minimum_weight_coverage:.6f}"
        )

    # Minimal idle action proves animation survives export without pretending to be a finished
    # motion set.  Secondary ornament motion remains a later profile-specific concern.
    if not failures:
        action = bpy.data.actions.new("idle_validation")
        armature.animation_data_create()
        armature.animation_data.action = action
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        root_pose = armature.pose.bones.get("root")
        for frame, angle in ((1, 0.0), (30, 0.012), (60, 0.0)):
            root_pose.rotation_mode = "XYZ"
            root_pose.rotation_euler[2] = angle
            root_pose.keyframe_insert("rotation_euler", frame=frame)
        bpy.ops.object.mode_set(mode="OBJECT")

    report = {
        "input": args.input,
        "profile": args.profile,
        "passed": not failures,
        "failures": failures,
        "vertices": len(mesh.data.vertices),
        "weighted_vertices": weighted,
        "weight_coverage": coverage,
        "bone_count": len(armature.data.bones),
        "bones": [item.name for item in armature.data.bones],
        "animations": ["idle_validation"] if not failures else [],
    }
    save_json(args.report, report)
    if failures:
        raise SystemExit(2)

    Path(args.output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output_glb, selected_only=False)
    print(f"AUTO_RIG bones={report['bone_count']} coverage={coverage:.6f}", flush=True)


if __name__ == "__main__":
    main()
