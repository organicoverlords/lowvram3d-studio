from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json, select_only, world_bounds


def add_bone(armature, name, head, tail, parent=None):
    head_v, tail_v = Vector(head), Vector(tail)
    if (tail_v - head_v).length < 1e-5:
        tail_v.z += 0.01
    bone = armature.edit_bones.new(name)
    bone.head, bone.tail = head_v, tail_v
    if parent:
        bone.parent = armature.edit_bones.get(parent)
        bone.use_connect = False
    return bone


def load_pose_report(path: str) -> dict:
    target = Path(path) if path else None
    if not target or not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def landmark_mapper(report: dict, minimum: Vector, maximum: Vector):
    landmarks = report.get("pose", {}).get("landmarks", [])
    worlds = report.get("pose", {}).get("world_landmarks", [])
    center = (minimum + maximum) * 0.5
    size = maximum - minimum

    def point(index: int, fallback: Vector) -> Vector:
        if index >= len(landmarks):
            return fallback.copy()
        item = landmarks[index]
        if float(item.get("visibility", 0.0)) < 0.25:
            return fallback.copy()
        x = minimum.x + float(item.get("x", 0.5)) * size.x
        z = maximum.z - float(item.get("y", 0.5)) * size.z
        y = center.y
        if index < len(worlds):
            y -= float(worlds[index].get("z", 0.0)) * max(size.y, size.x) * 0.35
        return Vector((x, y, z))

    return point


def humanoid_bones(minimum: Vector, maximum: Vector, report: dict) -> list[tuple]:
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    z0, z1 = minimum.z, maximum.z
    pelvis_f = Vector((center.x, center.y, z0 + size.z * 0.48))
    chest_f = Vector((center.x, center.y, z0 + size.z * 0.72))
    neck_f = Vector((center.x, center.y, z0 + size.z * 0.84))
    head_f = Vector((center.x, center.y, z0 + size.z * 0.96))
    point = landmark_mapper(report, minimum, maximum)

    l_sh = point(11, chest_f + Vector((-size.x * 0.16, 0, 0)))
    r_sh = point(12, chest_f + Vector((size.x * 0.16, 0, 0)))
    l_el = point(13, l_sh + Vector((-size.x * 0.20, 0, -size.z * 0.10)))
    r_el = point(14, r_sh + Vector((size.x * 0.20, 0, -size.z * 0.10)))
    l_wr = point(15, l_el + Vector((-size.x * 0.18, 0, -size.z * 0.10)))
    r_wr = point(16, r_el + Vector((size.x * 0.18, 0, -size.z * 0.10)))
    l_hand = point(19, l_wr + Vector((-size.x * 0.06, 0, -size.z * 0.02)))
    r_hand = point(20, r_wr + Vector((size.x * 0.06, 0, -size.z * 0.02)))
    l_hip = point(23, pelvis_f + Vector((-size.x * 0.09, 0, 0)))
    r_hip = point(24, pelvis_f + Vector((size.x * 0.09, 0, 0)))
    pelvis = (l_hip + r_hip) * 0.5
    l_knee = point(25, Vector((l_hip.x, center.y, z0 + size.z * 0.25)))
    r_knee = point(26, Vector((r_hip.x, center.y, z0 + size.z * 0.25)))
    l_ankle = point(27, Vector((l_hip.x, center.y, z0 + size.z * 0.04)))
    r_ankle = point(28, Vector((r_hip.x, center.y, z0 + size.z * 0.04)))
    l_toe = point(31, l_ankle + Vector((0, -size.y * 0.12, 0)))
    r_toe = point(32, r_ankle + Vector((0, -size.y * 0.12, 0)))
    shoulders = (l_sh + r_sh) * 0.5
    chest = pelvis.lerp(shoulders, 0.72)
    neck = shoulders.lerp(point(0, head_f), 0.42)
    head = point(0, head_f)
    head_top = Vector((head.x, head.y, min(z1, head.z + size.z * 0.10)))

    return [
        ("root", Vector((pelvis.x, pelvis.y, z0)), pelvis, None),
        ("pelvis", pelvis, pelvis.lerp(chest, 0.32), "root"),
        ("spine", pelvis.lerp(chest, 0.32), chest, "pelvis"),
        ("chest", chest, shoulders, "spine"),
        ("neck", shoulders, neck, "chest"),
        ("head", neck, head_top, "neck"),
        ("clavicle.L", shoulders, l_sh, "chest"),
        ("upper_arm.L", l_sh, l_el, "clavicle.L"),
        ("forearm.L", l_el, l_wr, "upper_arm.L"),
        ("hand.L", l_wr, l_hand, "forearm.L"),
        ("clavicle.R", shoulders, r_sh, "chest"),
        ("upper_arm.R", r_sh, r_el, "clavicle.R"),
        ("forearm.R", r_el, r_wr, "upper_arm.R"),
        ("hand.R", r_wr, r_hand, "forearm.R"),
        ("thigh.L", l_hip, l_knee, "pelvis"),
        ("shin.L", l_knee, l_ankle, "thigh.L"),
        ("foot.L", l_ankle, l_toe, "shin.L"),
        ("thigh.R", r_hip, r_knee, "pelvis"),
        ("shin.R", r_knee, r_ankle, "thigh.R"),
        ("foot.R", r_ankle, r_toe, "shin.R"),
    ]


def creature_bones(minimum: Vector, maximum: Vector) -> list[tuple]:
    center = (minimum + maximum) * 0.5
    sx, sy, sz = maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z
    return [
        ("root", (center.x, minimum.y, center.z), (center.x, center.y, center.z), None),
        ("spine", (center.x, center.y, center.z), (center.x, center.y + sy * 0.25, center.z + sz * 0.08), "root"),
        ("neck", (center.x, center.y + sy * 0.25, center.z + sz * 0.08), (center.x, center.y + sy * 0.4, center.z + sz * 0.2), "spine"),
        ("head", (center.x, center.y + sy * 0.4, center.z + sz * 0.2), (center.x, maximum.y, center.z + sz * 0.25), "neck"),
        ("wing.L", (center.x, center.y, center.z + sz * 0.1), (minimum.x, center.y, center.z + sz * 0.05), "spine"),
        ("wing.R", (center.x, center.y, center.z + sz * 0.1), (maximum.x, center.y, center.z + sz * 0.05), "spine"),
        ("leg.L", (center.x - sx * 0.15, center.y, center.z), (center.x - sx * 0.15, center.y, minimum.z), "root"),
        ("leg.R", (center.x + sx * 0.15, center.y, center.z), (center.x + sx * 0.15, center.y, minimum.z), "root"),
        ("tail", (center.x, center.y - sy * 0.2, center.z), (center.x, minimum.y, center.z), "root"),
    ]


def make_armature(kind: str, objects: list[bpy.types.Object], pose_report: dict) -> tuple[bpy.types.Object, dict[str, str], bool]:
    minimum, maximum = world_bounds(objects)
    data = bpy.data.armatures.new("GameRig")
    armature = bpy.data.objects.new("GameRig", data)
    bpy.context.collection.objects.link(armature)
    select_only([armature])
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    object_bones: dict[str, str] = {}
    pose_guided = False
    if kind == "mechanical":
        center = (minimum + maximum) * 0.5
        add_bone(data, "root", (center.x, center.y, minimum.z), (center.x, center.y, center.z), None)
        scale = max((maximum - minimum).length * 0.02, 0.02)
        for index, obj in enumerate(objects):
            obj_min, obj_max = world_bounds([obj])
            pivot = (obj_min + obj_max) * 0.5
            bone_name = f"part_{index:03d}_{obj.name}"[:60]
            add_bone(data, bone_name, pivot, pivot + Vector((0, 0, scale)), "root")
            object_bones[obj.name] = bone_name
    else:
        bones = humanoid_bones(minimum, maximum, pose_report) if kind == "humanoid" else creature_bones(minimum, maximum)
        pose_guided = kind == "humanoid" and bool(pose_report.get("pose", {}).get("detected"))
        for name, head, tail, parent in bones:
            add_bone(data, name, head, tail, parent)
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature, object_bones, pose_guided


def bind_organic(objects, armature) -> str:
    select_only(objects + [armature])
    bpy.context.view_layer.objects.active = armature
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        return "automatic_weights"
    except RuntimeError:
        for obj in objects:
            modifier = obj.modifiers.new("Armature", "ARMATURE")
            modifier.object = armature
            obj.parent = armature
        return "armature_modifier_fallback"


def bind_rigid(objects, armature, object_bones: dict[str, str]) -> str:
    for obj in objects:
        obj.parent = armature
        obj.parent_type = "BONE"
        obj.parent_bone = object_bones.get(obj.name, "root")
    return "rigid_parts"


def new_action(armature, name: str, frames: int):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, frames
    return action


def key_rotation(armature, bone_name: str, frame: int, xyz: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        return
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = xyz
    bone.keyframe_insert("rotation_euler", frame=frame)


def key_location(armature, bone_name: str, frame: int, xyz: tuple[float, float, float]) -> None:
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        return
    bone.location = xyz
    bone.keyframe_insert("location", frame=frame)


def add_idle(armature) -> str:
    new_action(armature, "idle", 48)
    for frame, amount in ((1, 0.0), (13, 0.025), (25, 0.0), (37, -0.02), (48, 0.0)):
        key_rotation(armature, "spine", frame, (amount, 0.0, amount * 0.4))
        key_rotation(armature, "head", frame, (0.0, amount * 0.6, 0.0))
    return "idle"


def add_dance(armature) -> str:
    new_action(armature, "dance_loop", 96)
    frames = (1, 13, 25, 37, 49, 61, 73, 85, 96)
    for index, frame in enumerate(frames):
        phase = (index / 8.0) * math.tau
        sway = math.sin(phase)
        bounce = max(0.0, math.sin(phase * 2.0))
        key_location(armature, "root", frame, (sway * 0.035, 0.0, bounce * 0.035))
        key_rotation(armature, "pelvis", frame, (0.08 * bounce, 0.0, -0.22 * sway))
        key_rotation(armature, "spine", frame, (-0.08 * bounce, 0.08 * sway, 0.18 * sway))
        key_rotation(armature, "chest", frame, (0.0, -0.08 * sway, -0.22 * sway))
        key_rotation(armature, "head", frame, (0.05 * bounce, 0.0, 0.08 * sway))
        key_rotation(armature, "upper_arm.L", frame, (0.45 + 0.45 * sway, -0.10, -0.55 - 0.30 * sway))
        key_rotation(armature, "forearm.L", frame, (-0.50 + 0.20 * sway, 0.0, -0.15))
        key_rotation(armature, "upper_arm.R", frame, (0.45 - 0.45 * sway, 0.10, 0.55 - 0.30 * sway))
        key_rotation(armature, "forearm.R", frame, (-0.50 - 0.20 * sway, 0.0, 0.15))
        key_rotation(armature, "thigh.L", frame, (-0.18 * sway, 0.0, 0.08 * sway))
        key_rotation(armature, "thigh.R", frame, (0.18 * sway, 0.0, 0.08 * sway))
        key_rotation(armature, "shin.L", frame, (0.20 * max(0.0, sway), 0.0, 0.0))
        key_rotation(armature, "shin.R", frame, (0.20 * max(0.0, -sway), 0.0, 0.0))
    for fcurve in bpy.context.object.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "BEZIER"
        fcurve.modifiers.new(type="CYCLES")
    return "dance_loop"


def add_mechanical_actions(armature) -> list[str]:
    new_action(armature, "mechanical_cycle", 48)
    bones = [bone.name for bone in armature.data.bones if bone.name != "root"]
    rotating = [name for name in bones if any(token in name.lower() for token in ("wheel", "propeller", "rotor", "turbine"))] or bones[:4]
    for name in rotating:
        key_rotation(armature, name, 1, (0, 0, 0))
        key_rotation(armature, name, 48, (0, math.tau, 0))
    return ["mechanical_cycle"]


def add_actions(armature, kind: str, animation_preset: str) -> list[str]:
    if kind == "humanoid":
        actions = [add_idle(armature)]
        if animation_preset in {"dance", "all", "auto"}:
            actions.append(add_dance(armature))
        return actions
    if kind == "creature":
        new_action(armature, "creature_idle", 48)
        for frame, amount in ((1, 0.0), (24, 0.15), (48, 0.0)):
            key_rotation(armature, "head", frame, (0.0, amount, 0.0))
            key_rotation(armature, "tail", frame, (0.0, -amount, 0.0))
        return ["creature_idle"]
    return add_mechanical_actions(armature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--kind", choices=("auto", "humanoid", "creature", "mechanical", "static"), default="auto")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--animation-preset", choices=("none", "idle", "dance", "all", "auto"), default="auto")
    parser.add_argument("--pose-report", default="")
    args = parser.parse_args(argv_after_double_dash())
    reset_scene()
    objects = import_mesh(args.input)
    prompt, kind = args.prompt.lower(), args.kind
    if kind == "auto":
        if any(word in prompt for word in ("vehicle", "machine", "robot", "tank", "ship", "door", "wheel")):
            kind = "mechanical"
        elif any(word in prompt for word in ("human", "person", "character", "soldier", "humanoid", "avatar")):
            kind = "humanoid"
        else:
            kind = "creature"
    if kind == "static":
        export_glb(args.output)
        save_json(args.report, {"kind": kind, "binding": "none", "actions": [], "success": True})
        return
    pose_report = load_pose_report(args.pose_report)
    armature, object_bones, pose_guided = make_armature(kind, objects, pose_report)
    binding = bind_rigid(objects, armature, object_bones) if kind == "mechanical" else bind_organic(objects, armature)
    actions = add_actions(armature, kind, args.animation_preset)
    export_glb(args.output)
    save_json(args.report, {
        "success": True,
        "kind": kind,
        "binding": binding,
        "bones": [bone.name for bone in armature.data.bones],
        "actions": actions,
        "animation_preset": args.animation_preset,
        "pose_guided_proportions": pose_guided,
        "deformation_proven": False,
        "warning": "Pose-guided template fitting and automatic weights require visual deformation inspection before shipping.",
    })


if __name__ == "__main__":
    main()
