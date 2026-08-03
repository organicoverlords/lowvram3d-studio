"""CPU visual proof for the selected source-visible GLB."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


GLB = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\balanced_010.glb")
ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")


def look_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render(name: str, location: Vector, target: Vector, camera: bpy.types.Object) -> str:
    camera.location = location
    look_at(camera, target)
    bpy.context.scene.render.filepath = str(ROOT / ("blender_" + name + ".png"))
    bpy.ops.render.render(write_still=True)
    return bpy.context.scene.render.filepath


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


def main() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(GLB))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("BLENDER_GLTF_IMPORT_NO_MESH")
    make_unlit_preview_materials()
    points = [obj.matrix_world @ Vector(corner) for obj in mesh_objects for corner in obj.bound_box]
    lower = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    upper = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    target = (lower + upper) * 0.5

    camera_data = bpy.data.cameras.new("SourceCamera")
    camera = bpy.data.objects.new("SourceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.lens = 50.0
    camera_data.angle = math.radians(66.5083847)
    camera_data.type = "PERSP"
    bpy.context.scene.camera = camera

    world = bpy.data.worlds.new("SceneWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.035, 0.045, 0.06, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(25), math.radians(-20), math.radians(25))
    sun_data.energy = 2.0

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"

    views = {
        "source": (Vector((0, 0, 100)), target),
        "left": (Vector((-320, -181, -50)), target),
        "right": (Vector((320, -181, -50)), target),
        "forward": (Vector((0, -181, 100)), target),
        "elevated": (Vector((0, -450, 100)), target),
        "rear": (Vector((0, -181, -400)), target),
        "debug_negative_far": (Vector((0, 0, -1000)), target),
        "debug_positive_near": (Vector((0, 0, 100)), target),
        "debug_source_plus_z": (Vector((0, 0, 0)), Vector((2, -181, 300))),
        "debug_source_minus_z": (Vector((0, 0, 0)), Vector((2, -181, -300))),
    }
    manifest = []
    for name, (location, view_target) in views.items():
        manifest.append({"name": name, "path": render(name, location, view_target, camera)})
    (ROOT / "blender_render_manifest.json").write_text(
        __import__("json").dumps({
            "schema": "blender_cpu_visual_proof_v1",
            "classification": "BLENDER_CPU_RENDER_PROVEN" if all(Path(item["path"]).is_file() for item in manifest) else "BLENDER_CPU_RENDER_REJECTED",
            "glb": str(GLB),
            "target": list(target),
            "world_bounds_min": list(lower),
            "world_bounds_max": list(upper),
            "mesh_objects": [{"name": obj.name, "location": list(obj.location), "dimensions": list(obj.dimensions)} for obj in mesh_objects],
            "camera_fov_deg": 66.5083847,
            "visual_camera_mode": "EXTERNALIZED_FOR_CPU_VISUAL_QA_NOT_UNREAL_SOURCE_CAMERA",
            "views": manifest,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
