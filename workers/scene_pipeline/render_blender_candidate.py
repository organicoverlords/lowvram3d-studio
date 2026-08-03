"""Render one existing Castlegrounds GLB from the audited exact source camera."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"
FOV_X = 66.50838470458984


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


def configure_camera(camera: bpy.types.Object, transform: list[list[float]], fov_x: float) -> None:
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
    data.lens = data.sensor_width / (2.0 * math.tan(math.radians(fov_x) / 2.0))
    data.clip_start = 0.01
    data.clip_end = 10000.0


def render(path: Path, *, cull: bool) -> str:
    for material in bpy.data.materials:
        if hasattr(material, "use_backface_culling"):
            material.use_backface_culling = cull
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: blender -P render_blender_candidate.py -- GLB_NAME OUTPUT_PREFIX")
    glb_arg = sys.argv[-3] if len(sys.argv) >= 4 else sys.argv[-2]
    prefix = sys.argv[-2] if len(sys.argv) >= 4 else sys.argv[-1]
    camera_json = Path(sys.argv[-1]) if len(sys.argv) >= 4 else None
    glb = Path(glb_arg)
    if not glb.is_absolute():
        glb = ROOT / glb
    work_root = glb.parent
    fov_x = float(json.loads(camera_json.read_text(encoding="utf-8"))["camera"]["fov_x_deg"]) if camera_json else FOV_X
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("BLENDER_EXACT_IMPORT_NO_MESH")
    make_unlit_preview_materials()
    transform = json.loads((ROOT / "blender_transform_audit.json").read_text(encoding="utf-8"))["axis_fixture"]["M_glb_to_blender"]
    camera_data = bpy.data.cameras.new("MoGeExactSourceCamera")
    camera = bpy.data.objects.new("MoGeExactSourceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    configure_camera(camera, transform, fov_x)
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
    renders = {
        "cull_off": render(work_root / f"{prefix}_cull_off.png", cull=False),
        "cull_on": render(work_root / f"{prefix}_cull_on.png", cull=True),
    }
    receipt = {"schema": "blender_exact_candidate_render_v1", "glb": str(glb), "prefix": prefix, "renders": renders, "camera_contract": "M_GLB_TO_BLENDER_AXIS_FIXTURE_EXACT_SOURCE"}
    (work_root / f"{prefix}_render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (PROOF / f"{prefix}_render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
