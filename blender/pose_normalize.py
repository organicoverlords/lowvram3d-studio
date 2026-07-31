"""Normalize an existing rigged biped to deterministic A/T pose while preserving source pose.

Run through Blender with ``--python blender/pose_normalize.py -- ...``. The script never overwrites
its input and exports the source-pose copy before applying a new rest pose.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json


BIPED_BONES = (
    "upper_arm.L", "forearm.L", "hand.L",
    "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L",
    "thigh.R", "shin.R", "foot.R",
)


def armature_in_scene() -> bpy.types.Object:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected exactly one armature, found {len(armatures)}")
    return armatures[0]


def require_bones(armature: bpy.types.Object) -> None:
    missing = [name for name in BIPED_BONES if armature.pose.bones.get(name) is None]
    if missing:
        raise RuntimeError(f"missing required biped bones: {missing}")


def align_bone(armature: bpy.types.Object, name: str, target: Vector) -> None:
    bone = armature.pose.bones[name]
    current = (bone.tail - bone.head).normalized()
    desired = target.normalized()
    delta = current.rotation_difference(desired)
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = delta @ bone.rotation_quaternion


def apply_pose(armature: bpy.types.Object, mode: str, arm_degrees: float) -> dict:
    require_bones(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    if mode == "a_pose":
        radians = math.radians(arm_degrees)
        lateral, down = math.sin(radians), math.cos(radians)
        align_bone(armature, "upper_arm.L", Vector((-lateral, 0.0, -down)))
        align_bone(armature, "upper_arm.R", Vector((lateral, 0.0, -down)))
    elif mode == "t_pose":
        align_bone(armature, "upper_arm.L", Vector((-1.0, 0.0, 0.0)))
        align_bone(armature, "upper_arm.R", Vector((1.0, 0.0, 0.0)))
    elif mode != "neutral_biped":
        raise RuntimeError(f"unsupported biped pose mode: {mode}")

    # Small symmetric elbow bend avoids a perfectly locked deformation singularity.
    for name, bend in (("forearm.L", -8.0), ("forearm.R", 8.0)):
        bone = armature.pose.bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.y += math.radians(bend)

    bpy.context.view_layer.update()
    before = {
        name: [float(value) for value in (armature.pose.bones[name].tail - armature.pose.bones[name].head).normalized()]
        for name in ("upper_arm.L", "upper_arm.R")
    }
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"upper_arm_directions": before, "arm_degrees": arm_degrees}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", choices=("a_pose", "t_pose", "neutral_biped"), default="a_pose")
    parser.add_argument("--arm-degrees", type=float, default=40.0)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    import_mesh(args.input)
    source = Path(args.source_output)
    source.parent.mkdir(parents=True, exist_ok=True)
    export_glb(str(source), selected_only=False)

    armature = armature_in_scene()
    metrics = apply_pose(armature, args.mode, args.arm_degrees)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_glb(str(output), selected_only=False)
    save_json(args.report, {
        "success": True,
        "source_input": str(args.input),
        "source_pose_output": str(source),
        "bind_pose_output": str(output),
        "source_pose_preserved": True,
        "bind_pose": args.mode,
        "retarget_pose": "ue5_mannequin_a_pose" if args.mode == "a_pose" else args.mode,
        "alternate_pose": "t_pose" if args.mode == "a_pose" else None,
        "measurements": metrics,
        "deformation_proven": False,
        "manual_visual_validation_required": True,
    })


if __name__ == "__main__":
    main()
