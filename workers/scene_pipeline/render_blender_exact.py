"""Render the selected GLB from the proven transformed MoGe source camera."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"
FOV_X = 66.50838470458984
FOV_Y = 52.37591552734375


def make_unlit_preview_materials() -> None:
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        links = material.node_tree.links
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


def configure_camera(camera: bpy.types.Object, transform: list[list[float]]) -> dict:
    matrix = Matrix(transform)
    linear = matrix.to_3x3()
    origin = matrix @ Vector((0, 0, 0, 1))
    forward = (linear @ Vector((0, 0, 1))).normalized()
    up = (linear @ Vector((0, -1, 0))).normalized()
    right = forward.cross(up).normalized()
    up = right.cross(forward).normalized()
    camera.location = origin.xyz
    camera.rotation_euler = Matrix((right, up, -forward)).transposed().to_4x4().to_euler()
    data = camera.data
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 32.0
    data.lens = data.sensor_width / (2.0 * math.tan(math.radians(FOV_X) / 2.0))
    data.shift_x = 0.0
    data.shift_y = 0.0
    data.clip_start = 0.01
    data.clip_end = 10000.0
    return {"origin": list(origin.xyz), "forward": list(forward), "up": list(up), "right": list(right), "fov_x_deg": FOV_X, "fov_y_deg": FOV_Y, "lens_mm": data.lens}


def render(name: str, camera: bpy.types.Object, objects: list[bpy.types.Object]) -> str:
    path = ROOT / ("blender_exact_" + name + ".png")
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(ROOT / "balanced_010.glb"))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("BLENDER_EXACT_IMPORT_NO_MESH")
    make_unlit_preview_materials()
    transform = json.loads((ROOT / "blender_transform_audit.json").read_text(encoding="utf-8"))["axis_fixture"]["M_glb_to_blender"]
    camera_data = bpy.data.cameras.new("MoGeExactSourceCamera")
    camera = bpy.data.objects.new("MoGeExactSourceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_receipt = configure_camera(camera, transform)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 384
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = bpy.data.worlds.new("ExactSourceWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    for material in bpy.data.materials:
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = False
    renders = {"source_cull_off": render("source_cull_off", camera, mesh_objects)}
    for obj in mesh_objects:
        obj.display_type = "WIRE"
    renders["wireframe"] = render("wireframe", camera, mesh_objects)
    for obj in mesh_objects:
        obj.display_type = "TEXTURED"
    for material in bpy.data.materials:
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = True
    renders["source_cull_on"] = render("source_cull_on", camera, mesh_objects)
    for material in bpy.data.materials:
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = False
    receipt = {
        "schema": "blender_exact_source_camera_v1",
        "classification": "BLENDER_EXACT_SOURCE_CAMERA_RENDER_READY",
        "glb": str(ROOT / "balanced_010.glb"),
        "transform_source": str(ROOT / "blender_transform_audit.json"),
        "camera": camera_receipt,
        "resolution": [512, 384],
        "principal_point_shift": [0.0, 0.0],
        "backface_culling_initial": False,
        "colour_management": "Standard_None_Exposure0_Gamma1",
        "renders": renders,
        "source_plane_present": False,
    }
    (ROOT / "blender_exact_source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (PROOF / "blender_exact_source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
