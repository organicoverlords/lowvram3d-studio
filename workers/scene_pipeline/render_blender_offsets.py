"""Render exact-contract left/right offset views for one candidate."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")


def make_unlit_preview_materials() -> None:
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        nodes = material.node_tree.nodes; links = material.node_tree.links
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base = principled.inputs.get("Base Color")
        emission = principled.inputs.get("Emission Color") or principled.inputs.get("Emission")
        strength = principled.inputs.get("Emission Strength")
        if base is not None and emission is not None:
            for link in list(emission.links):
                links.remove(link)
            if base.is_linked:
                links.new(base.links[0].from_socket, emission)
            else:
                emission.default_value = base.default_value
            if strength is not None:
                strength.default_value = 1.0


def camera_pose(camera: bpy.types.Object, transform: list[list[float]], offset_x: float, fov_x: float) -> None:
    matrix = Matrix(transform); linear = matrix.to_3x3()
    origin = matrix @ Vector((offset_x, 0, 0, 1)); forward = (linear @ Vector((0, 0, 1))).normalized(); up = (linear @ Vector((0, -1, 0))).normalized(); right = forward.cross(up).normalized(); up = right.cross(forward).normalized()
    camera.location = origin.xyz; camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_4x4().to_euler()
    camera.data.type = "PERSP"; camera.data.sensor_fit = "HORIZONTAL"; camera.data.sensor_width = 32.0; camera.data.lens = 32.0 / (2.0 * math.tan(math.radians(fov_x) / 2.0)); camera.data.clip_start = 0.01; camera.data.clip_end = 10000.0


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("usage: blender -P render_blender_offsets.py -- GLB PREFIX CAMERA_JSON OFFSET")
    glb = Path(sys.argv[-4]); prefix = sys.argv[-3]; receipt_path = Path(sys.argv[-2]); offset = float(sys.argv[-1]); work = glb.parent
    fov_x = float(json.loads(receipt_path.read_text(encoding="utf-8"))["camera"]["fov_x_deg"]) if "camera" in json.loads(receipt_path.read_text(encoding="utf-8")) else 66.5083847
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(glb)); objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objects: raise RuntimeError("BLENDER_OFFSET_IMPORT_NO_MESH")
    make_unlit_preview_materials()
    for material in bpy.data.materials:
        if hasattr(material, "use_backface_culling"): material.use_backface_culling = True
    transform = json.loads((ROOT / "blender_transform_audit.json").read_text(encoding="utf-8"))["axis_fixture"]["M_glb_to_blender"]
    data = bpy.data.cameras.new("MoGeOffsetCamera"); camera = bpy.data.objects.new("MoGeOffsetCamera", data); bpy.context.collection.objects.link(camera)
    camera_pose(camera, transform, offset, fov_x)
    scene = bpy.context.scene; scene.camera = camera; scene.render.engine = "BLENDER_EEVEE"; scene.render.resolution_x = 512; scene.render.resolution_y = 384; scene.render.resolution_percentage = 100; scene.render.image_settings.file_format = "PNG"; scene.render.film_transparent = False; scene.view_settings.view_transform = "Standard"; scene.view_settings.look = "None"; scene.view_settings.exposure = 0.0; scene.view_settings.gamma = 1.0
    world = bpy.data.worlds.new("OffsetWorld"); scene.world = world; world.use_nodes = True; world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1); world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    name = "left" if offset < 0 else "right"; out = work / f"{prefix}_{name}.png"; scene.render.filepath = str(out); bpy.ops.render.render(write_still=True); print(out)


if __name__ == "__main__":
    main()
