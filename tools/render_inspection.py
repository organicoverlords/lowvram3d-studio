"""Render the all-assets lineup to a PNG.

Run inside Blender:
    blender.exe --background ALL_ASSETS_inspection.blend --python render_inspection.py

**Cycles on CPU, deliberately.** A Hunyuan3D-Paint run is on the GPU while this
renders, and a Blender render during a paint run has killed two paint runs
today. Forcing the CPU device costs minutes and cannot touch the CUDA context.

The camera is orthographic. A perspective camera framing a 95 m row has to stand
far enough back that the near assets dwarf the far ones, which is the opposite of
what a size comparison is for. Orthographic gives every asset the same scale on
screen, so the heights in the picture are the heights in the table.
"""

import sys
from pathlib import Path

import bpy
from mathutils import Vector

OUT = Path(r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803"
           r"\evidence\deliverables\ALL_ASSETS_lineup.png")

MARGIN = 1.06


def scene_bounds():
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            p = obj.matrix_world @ Vector(corner)
            lo = Vector((min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)))
            hi = Vector((max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)))
    return lo, hi


def main():
    scene = bpy.context.scene
    lo, hi = scene_bounds()
    span_x = hi.x - lo.x
    span_z = hi.z - lo.z
    print(f"row spans {span_x:.1f} m wide, {span_z:.1f} m tall")

    # A ground plane, so the assets read as standing on something and their
    # bases are visibly level rather than merely claimed to be.
    bpy.ops.mesh.primitive_plane_add(size=max(span_x, 200.0) * 2,
                                     location=((lo.x + hi.x) / 2, 0, 0))
    floor = bpy.context.active_object
    floor_mat = bpy.data.materials.new("floor")
    floor_mat.use_nodes = True
    floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"] \
        .default_value = (0.22, 0.22, 0.24, 1.0)
    floor.data.materials.append(floor_mat)

    camera_data = bpy.data.cameras.new("lineup")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span_x * MARGIN
    camera = bpy.data.objects.new("lineup", camera_data)
    camera.location = ((lo.x + hi.x) / 2, -max(span_x, 100.0),
                       lo.z + span_z / 2)
    camera.rotation_euler = (1.5708, 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for light in [o for o in scene.objects if o.type == "LIGHT"]:
        light.data.energy = 4.0

    fill_data = bpy.data.lights.new("fill", type="SUN")
    fill_data.energy = 1.6
    fill = bpy.data.objects.new("fill", fill_data)
    fill.rotation_euler = (1.1, 0.0, -1.4)
    scene.collection.objects.link(fill)

    if scene.world and scene.world.node_tree:
        scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"          # see module docstring
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 3400
    scene.render.resolution_y = max(600, int(3400 * (span_z * 1.35) / span_x))
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(OUT)

    print(f"rendering {scene.render.resolution_x}x{scene.render.resolution_y} "
          f"on {scene.cycles.device}")
    bpy.ops.render.render(write_still=True)
    print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")


main()
