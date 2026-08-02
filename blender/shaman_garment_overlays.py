"""Render non-destructive diagnostic overlays for a Shaman garment candidate."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args(argv_after_double_dash())
    bpy.ops.wm.open_mainfile(filepath=args.input)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    for obj in meshes:
        material = bpy.data.materials.new(f"overlay_{obj.name}")
        material.diffuse_color = (0.16, 0.22, 0.30, 1.0)
        if args.mode == "garment" and "Garment" in obj.name:
            material.diffuse_color = (0.95, 0.15, 0.08, 1.0)
        elif args.mode == "seam":
            material.diffuse_color = (0.12, 0.30, 0.55, 1.0) if "Body" in obj.name else (1.0, 0.65, 0.05, 1.0)
        elif args.mode == "closure" and "Body" in obj.name:
            material.diffuse_color = (0.15, 0.85, 0.25, 1.0)
        obj.data.materials.clear()
        obj.data.materials.append(material)
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            minimum = Vector(tuple(min(a, b) for a, b in zip(minimum, point)))
            maximum = Vector(tuple(max(a, b) for a, b in zip(maximum, point)))
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 3.0 + 1.0
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 520
    scene.render.resolution_y = 680
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "MATERIAL"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, angle in (("front", 0), ("threequarter", 45), ("side", 90)):
        data = bpy.data.cameras.new(f"overlay_{name}")
        data.type = "ORTHO"
        data.ortho_scale = max(size.x, size.z) * 1.15
        camera = bpy.data.objects.new(f"overlay_{name}", data)
        bpy.context.collection.objects.link(camera)
        radians = math.radians(angle)
        camera.location = centre + Vector((math.sin(radians) * radius, -math.cos(radians) * radius, 0))
        camera.rotation_euler = (centre - camera.location).normalized().to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera
        scene.render.filepath = str(output / f"{args.prefix}_{name}.png")
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(camera, do_unlink=True)


if __name__ == "__main__":
    main()
