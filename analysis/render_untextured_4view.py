"""Render a plain-shaded 4-view contact check of cleaned, untextured geometry."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, OUTDIR = Path(argv[0]), Path(argv[1])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

mat = bpy.data.materials.new("Clay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.45, 1.0)
for o in meshes:
    o.data.materials.clear()
    o.data.materials.append(mat)

scene = bpy.context.scene
engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 480
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.4, 0.4, 0.42, 1)
scene.view_settings.view_transform = "Standard"

key = bpy.data.lights.new("Key", "SUN")
key.energy = 3.0
key_o = bpy.data.objects.new("Key", key)
scene.collection.objects.link(key_o)
key_o.rotation_euler = (math.radians(50), 0, math.radians(30))

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.6
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

OUTDIR.mkdir(parents=True, exist_ok=True)
for name, loc in (
    ("front", (0.0, -3.0, 0.0)),
    ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.6)),
    ("side", (3.0, 0.0, 0.0)),
    ("back", (0.0, 3.0, 0.0)),
):
    cam.location = Vector(loc)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(OUTDIR / f"clean_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {scene.render.filepath}", flush=True)
