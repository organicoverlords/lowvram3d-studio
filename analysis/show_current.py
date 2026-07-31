"""Large, well-lit render of the current textured candidate for visual review."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, OUTDIR = Path(argv[0]), Path(argv[1])
SIZE = 768

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

scene = bpy.context.scene
engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = SIZE
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.30, 0.31, 0.34, 1)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

# Fixed three-point rig so every camera sees identical lighting.
for name, energy, rotation in (
    ("Key", 4.0, (math.radians(55), 0.0, math.radians(35))),
    ("Fill", 2.0, (math.radians(65), 0.0, math.radians(200))),
    ("Rim", 3.0, (math.radians(115), 0.0, math.radians(120))),
):
    light = bpy.data.lights.new(name, "SUN")
    light.energy = energy
    obj = bpy.data.objects.new(name, light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = rotation

cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.35
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

OUTDIR.mkdir(parents=True, exist_ok=True)
for name, loc in (
    ("front", (0.0, -3.0, 0.25)),
    ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.7)),
    ("side", (3.0, 0.0, 0.25)),
    ("back", (0.0, 3.0, 0.25)),
):
    cam.location = Vector(loc)
    direction = Vector((0.0, 0.0, -0.05)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(OUTDIR / f"current_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {name}", flush=True)
