"""Fresh-import CPU proof for the bounded material-recovery fallback."""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import argv_after_double_dash, import_mesh, mesh_stats, reset_scene, save_json, world_bounds
from shaman_texture_review import add_lights, analyse, place_camera, setup_world


VIEWS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("rear", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("top", 0.0, 55.0),
    ("bottom", 0.0, -55.0),
    ("front_three_quarter", 35.0, 8.0),
    ("rear_three_quarter", 215.0, 8.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--front-direction", choices=("+z", "-z"), required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args(argv_after_double_dash())

    glb = Path(args.glb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError("FRESH_IMPORT_NO_MESH")

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    setup_world()
    add_lights(radius)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    camera_data = bpy.data.cameras.new("cpu_fallback_qa_camera")
    camera = bpy.data.objects.new("cpu_fallback_qa_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    front_sign = -1.0 if args.front_direction == "+z" else 1.0
    views = {}
    for name, yaw, pitch in VIEWS:
        place_camera(camera, centre, radius, yaw, pitch, 1.0, centre, front_sign)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        views[name] = {"path": str(path), **analyse(path)}
        print(f"CPU_FALLBACK_VIEW {name} {path}", flush=True)

    mesh = mesh_stats(objects)
    report = {
        "schema": "cpu_fallback_eight_view_qa_v1",
        "classification": "PROVEN_FRESH_IMPORT_AND_EIGHT_VIEW_RENDER",
        "glb": str(glb),
        "glb_bytes": glb.stat().st_size,
        "fresh_import": True,
        "mesh": mesh,
        "no_armature": mesh["armatures"] == 0,
        "no_actions": mesh["actions"] == 0,
        "front_direction_gltf": args.front_direction,
        "render_engine": scene.render.engine,
        "samples": args.samples,
        "views": views,
    }
    save_json(str(args.report), report)
    print(f"CPU_FALLBACK_QA_DONE views={len(views)} report={args.report}", flush=True)


if __name__ == "__main__":
    main()
