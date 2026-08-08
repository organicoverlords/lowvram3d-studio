"""CPU-only, fail-closed five-pose proof for an existing rigged Blender asset.

The input is opened/imported read-only and is never saved.  This is deliberately a
preview worker, not a rigging or retargeting stage: unsupported anatomy remains in
its source pose and is recorded in ``five_pose_proof.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, world_bounds

POSE_LABELS = (
    "source_rest",
    "a_or_preserved",
    "t_or_upper_body",
    "left_arm_test",
    "right_arm_test",
)
BIPED_ARM_BONES = ("upper_arm.L", "forearm.L", "hand.L", "upper_arm.R", "forearm.R", "hand.R")


def receipt_path(out_dir: Path) -> Path:
    return out_dir / "five_pose_proof.json"


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def load_input(path: Path) -> list[bpy.types.Object]:
    if path.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if path.suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return import_mesh(str(path))
    raise ValueError("--input must be an existing .blend, .glb, or .gltf")


def nonzero_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector, Vector]:
    minimum, maximum = world_bounds(meshes)
    extent = maximum - minimum
    if not all(math.isfinite(float(v)) for v in (*minimum, *maximum)) or min(extent) <= 1e-8:
        raise RuntimeError("mesh bounds are empty, non-finite, or zero-sized")
    return minimum, maximum, extent


def armature_in_scene() -> bpy.types.Object | None:
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    return armatures[0] if len(armatures) == 1 else None


def configure_cpu_workbench(width: int = 320, height: int = 320) -> None:
    scene = bpy.context.scene
    available = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if "BLENDER_WORKBENCH" not in available:
        raise RuntimeError("this Blender build does not expose BLENDER_WORKBENCH")
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.studio_light = "paint.sl"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = False
    scene.display.shading.show_xray = False
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (0.04, 0.04, 0.04)
    scene.render.image_settings.color_mode = "RGBA"


def make_camera(minimum: Vector, maximum: Vector) -> bpy.types.Object:
    centre = (minimum + maximum) * 0.5
    extent = maximum - minimum
    data = bpy.data.cameras.new("FivePoseProofCamera")
    data.type = "ORTHO"
    data.ortho_scale = max(float(extent.z) * 1.12, float(extent.x) * 1.35, 0.1)
    camera = bpy.data.objects.new("FivePoseProofCamera", data)
    bpy.context.collection.objects.link(camera)
    camera.location = centre + Vector((0.0, -max(float(extent.y) * 3.0, 3.0), float(extent.z) * 0.02))
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    return camera


def capture_pose(armature: bpy.types.Object) -> dict[str, tuple]:
    return {bone.name: (bone.location.copy(), bone.rotation_mode, bone.rotation_quaternion.copy(),
                        bone.rotation_euler.copy(), bone.scale.copy()) for bone in armature.pose.bones}


def restore_pose(armature: bpy.types.Object, snapshot: dict[str, tuple]) -> None:
    for bone in armature.pose.bones:
        location, mode, quaternion, euler, scale = snapshot[bone.name]
        bone.location = location
        bone.rotation_mode = mode
        bone.rotation_quaternion = quaternion
        bone.rotation_euler = euler
        bone.scale = scale
    bpy.context.view_layer.update()


def clear_pose(armature: bpy.types.Object) -> None:
    for bone in armature.pose.bones:
        bone.location = (0.0, 0.0, 0.0)
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def apply_safe_pose(armature: bpy.types.Object | None, label: str,
                    source_pose: dict[str, tuple] | None = None) -> tuple[str, str]:
    if armature is not None and source_pose is not None:
        restore_pose(armature, source_pose)
    if armature is None:
        return "preserved", "no unique armature"
    names = {bone.name for bone in armature.pose.bones}
    full = all(name in names for name in BIPED_ARM_BONES)
    upper = {side: f"upper_arm.{side}" in names for side in ("L", "R")}
    if label == "source_rest":
        return "source", "source pose/frame preserved"
    if label in {"a_or_preserved", "t_or_upper_body"} and not full:
        return "preserved", "unsupported biped bones; source pose preserved"
    if label == "left_arm_test" and not upper["L"]:
        return "preserved", "left upper arm unsupported; source pose preserved"
    if label == "right_arm_test" and not upper["R"]:
        return "preserved", "right upper arm unsupported; source pose preserved"

    clear_pose(armature)
    radians = math.radians(35.0)
    if label == "a_or_preserved":
        for side, sign in (("L", -1.0), ("R", 1.0)):
            armature.pose.bones[f"upper_arm.{side}"].rotation_euler.y = sign * radians
        return "applied", "deterministic A-style upper-arm test"
    if label == "t_or_upper_body":
        for side, sign in (("L", -1.0), ("R", 1.0)):
            armature.pose.bones[f"upper_arm.{side}"].rotation_euler.y = sign * (math.pi / 2.0)
        return "applied", "deterministic T-style upper-arm test"
    side = "L" if label == "left_arm_test" else "R"
    armature.pose.bones[f"upper_arm.{side}"].rotation_euler.x = math.radians(-25.0)
    return "applied", f"deterministic {side} upper-arm rotation"


def make_contact_sheet(paths: list[Path], target: Path) -> None:
    images = [bpy.data.images.load(str(path), check_existing=False) for path in paths]
    if not images:
        return
    width, height = images[0].size[:]
    sheet = bpy.data.images.new("five_pose_contact_sheet", width=width * len(images), height=height, alpha=True)
    pixels = [0.0] * (width * len(images) * height * 4)
    for index, image in enumerate(images):
        source = list(image.pixels[:])
        for row in range(height):
            src_start = row * width * 4
            dst_start = (row * width * len(images) + index * width) * 4
            pixels[dst_start:dst_start + width * 4] = source[src_start:src_start + width * 4]
    sheet.pixels.foreach_set(pixels)
    sheet.filepath_raw = str(target)
    sheet.file_format = "PNG"
    sheet.save()
    for image in images:
        bpy.data.images.remove(image)
    bpy.data.images.remove(sheet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv_after_double_dash())
    input_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    receipt = {"success": False, "input": str(input_path), "out_dir": str(out_dir), "cpu_only": True,
               "poses": [], "outputs": {}, "failures": []}
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"input does not exist: {input_path}")
        meshes = load_input(input_path)
        if not meshes:
            raise RuntimeError("no mesh objects found; refusing to fabricate a rig")
        minimum, maximum, extent = nonzero_bounds(meshes)
        armature = armature_in_scene()
        if armature is None:
            raise RuntimeError("no unique armature found; input is not a verifiable rigged asset")
        configure_cpu_workbench()
        make_camera(minimum, maximum)
        source_pose = capture_pose(armature)
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_paths = []
        for label in POSE_LABELS:
            status, detail = apply_safe_pose(armature, label, source_pose)
            bpy.context.scene.render.filepath = str(out_dir / f"{label}.png")
            bpy.ops.render.render(write_still=True)
            preview = out_dir / f"{label}.png"
            preview_paths.append(preview)
            receipt["poses"].append({"label": label, "status": status, "detail": detail,
                                     "file": str(preview), "frame": int(bpy.context.scene.frame_current)})
        contact = out_dir / "five_pose_contact_sheet.png"
        make_contact_sheet(preview_paths, contact)
        receipt["outputs"] = {"previews": [str(path) for path in preview_paths], "contact_sheet": str(contact),
                              "receipt": str(receipt_path(out_dir))}
        receipt["metrics"] = {"mesh_objects": len(meshes), "vertices": sum(len(obj.data.vertices) for obj in meshes),
                              "armature": armature.name, "bones": len(armature.data.bones),
                              "bounds_min": list(minimum), "bounds_max": list(maximum), "extent": list(extent),
                              "render_engine": "BLENDER_WORKBENCH", "render_device": "CPU-safe workbench"}
        receipt["success"] = True
    except Exception as exc:
        receipt["failures"].append(f"{type(exc).__name__}: {exc}")
    write_receipt(receipt_path(out_dir), receipt)
    return 0 if receipt["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
