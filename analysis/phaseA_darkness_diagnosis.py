"""PHASE A: separate texture darkness from lighting/material darkness.

Renders the same mesh from six cardinal views under three diagnostic materials:
  BASECOLOR_EMISSION - base colour straight to Emission, strength 1, lighting-independent
  NEUTRAL_CLAY       - uniform mid-grey diffuse under fixed neutral lighting
  CURRENT_MATERIAL   - the production material/lighting as shipped

Background is rendered transparent so model pixels can be isolated by alpha; all luminance
statistics are computed over model pixels only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, ATLAS, OUTDIR = Path(argv[0]), Path(argv[1]), Path(argv[2])

VIEWS = {
    "front": (0.0, -3.0, 0.0),
    "back": (0.0, 3.0, 0.0),
    "left": (-3.0, 0.0, 0.0),
    "right": (3.0, 0.0, 0.0),
    "top": (0.0, 0.0, 3.0),
    "underside": (0.0, 0.0, -3.0),
}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("no meshes imported")

original_materials = {o.name: [m for m in o.data.materials] for o in meshes}

scene = bpy.context.scene
engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True          # background alpha=0 -> model pixels isolatable
scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.0, 0.0, 0.0, 1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.6
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# Fixed neutral lighting, identical for every camera and every lit mode.
key = bpy.data.lights.new("Key", "SUN")
key.energy = 3.0
key_object = bpy.data.objects.new("Key", key)
scene.collection.objects.link(key_object)
key_object.rotation_euler = (math.radians(50), 0.0, math.radians(30))
fill = bpy.data.lights.new("Fill", "SUN")
fill.energy = 1.5
fill_object = bpy.data.objects.new("Fill", fill)
scene.collection.objects.link(fill_object)
fill_object.rotation_euler = (math.radians(60), 0.0, math.radians(210))


def emission_material() -> bpy.types.Material:
    material = bpy.data.materials.new("BASECOLOR_EMISSION")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    uvmap = tree.nodes.new("ShaderNodeUVMap")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(ATLAS))
    texture.interpolation = "Closest"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(uvmap.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def clay_material() -> bpy.types.Material:
    material = bpy.data.materials.new("NEUTRAL_CLAY")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    principled.inputs["Roughness"].default_value = 0.6
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.2
    return material


def apply(material: bpy.types.Material | None) -> None:
    for obj in meshes:
        obj.data.materials.clear()
        if material is None:
            for original in original_materials[obj.name]:
                obj.data.materials.append(original)
        else:
            obj.data.materials.append(material)


def render_mode(mode: str, material: bpy.types.Material | None, lit: bool) -> None:
    apply(material)
    key_object.hide_render = not lit
    fill_object.hide_render = not lit
    for name, location in VIEWS.items():
        cam.location = Vector(location)
        direction = Vector((0.0, 0.0, 0.0)) - cam.location
        up = "Y" if name not in {"top", "underside"} else "Z"
        cam.rotation_euler = direction.to_track_quat("-Z", up).to_euler()
        scene.render.filepath = str(OUTDIR / f"{mode}_{name}.png")
        bpy.ops.render.render(write_still=True)


OUTDIR.mkdir(parents=True, exist_ok=True)
render_mode("BASECOLOR_EMISSION", emission_material(), lit=False)
render_mode("NEUTRAL_CLAY", clay_material(), lit=True)
render_mode("CURRENT_MATERIAL", None, lit=True)
print("PHASE_A_RENDERS_DONE " + json.dumps(sorted(VIEWS)), flush=True)
