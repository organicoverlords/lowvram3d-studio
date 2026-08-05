"""Render the bounded RearHeadSafe final view set from a fresh GLB import."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds
from shaman_texture_review import add_lights, analyse, configure_unlit, place_camera, setup_world


VIEWS = [
    ("front", 0.0, 0.0, 1.0),
    ("three_quarter_left", -35.0, 8.0, 1.0),
    ("three_quarter_right", 35.0, 8.0, 1.0),
    ("rear", 180.0, 0.0, 1.0),
    ("rear_left", 145.0, 5.0, 1.0),
    ("rear_right", 215.0, 5.0, 1.0),
    ("rear_close", 180.0, 2.0, 0.34),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--unlit", action="store_true")
    parser.add_argument("--front-direction", choices=("+z", "-z"), required=True)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.glb)
    if not objects:
        raise RuntimeError("no mesh imported")
    if args.unlit:
        configure_unlit(objects)
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    head = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.86))
    setup_world()
    add_lights(radius)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE" if args.unlit else "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    camera_data = bpy.data.cameras.new("rear_semantic_review")
    camera = bpy.data.objects.new("rear_semantic_review", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    front_sign = -1.0 if args.front_direction == "+z" else 1.0
    for name, yaw, pitch, zoom in VIEWS:
        place_camera(camera, centre, radius, yaw, pitch, zoom, head if name == "rear_close" else centre, front_sign)
        path = output / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        results[name] = {**analyse(path), "path": str(path)}
        print(f"REAR_SAFE_REVIEW {name}: {results[name]}", flush=True)
    save_json(args.report, {"schema": "rear_semantic_final_review_v1", "glb": args.glb, "unlit": args.unlit, "resolution": args.resolution, "front_direction": args.front_direction, "views": results})


if __name__ == "__main__":
    main()
