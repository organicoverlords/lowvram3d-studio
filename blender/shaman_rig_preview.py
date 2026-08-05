"""Deterministic multi-view preview renders for the shaman rig/motion stages.

Workbench renders only: this is geometry and deformation proof, not a lighting or
texture proof. Cameras are orthographic and derived from the model bounds so the
same framing is reused across milestones and stays comparable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, reset_scene  # noqa: E402


# name -> azimuth degrees around +Z, measured from -Y (front)
VIEWS = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": 270.0,
    "three_quarter": 45.0,
}


def load(path: str) -> None:
    reset_scene()
    suffix = Path(path).suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise RuntimeError(f"unsupported preview input: {path}")


def scene_bounds() -> tuple[Vector, Vector]:
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], world[axis])
                maximum[axis] = max(maximum[axis], world[axis])
    if not found:
        raise RuntimeError("preview input contains no mesh")
    return minimum, maximum


def configure(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.62, 0.62, 0.64)
    shading.show_cavity = True
    scene.display.render_aa = f"{samples}" if str(samples) in {"5", "8", "11", "16", "32"} else "8"


def place_camera(name: str, azimuth: float, centre: Vector, radius: float, ortho: float):
    data = bpy.data.cameras.new(f"cam_{name}")
    data.type = "ORTHO"
    data.ortho_scale = ortho
    camera = bpy.data.objects.new(f"cam_{name}", data)
    bpy.context.scene.collection.objects.link(camera)

    angle = math.radians(azimuth)
    offset = Vector((math.sin(angle) * radius, -math.cos(angle) * radius, 0.0))
    camera.location = centre + offset
    direction = (centre - camera.location).normalized()
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="view")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--action", default=None)
    parser.add_argument("--views", default="front,right,back,left,three_quarter")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv_after_double_dash())

    load(args.input)
    configure(args.width, args.height, args.samples)

    if args.action:
        armature = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
        if armature is None:
            raise RuntimeError("--action requires an armature in the scene")
        if armature.animation_data is None:
            armature.animation_data_create()
        if args.action not in bpy.data.actions:
            raise RuntimeError(f"action {args.action!r} not found")
        armature.animation_data.action = bpy.data.actions[args.action]

    if args.frame is not None:
        bpy.context.scene.frame_set(args.frame)
        bpy.context.view_layer.update()

    minimum, maximum = scene_bounds()
    centre = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size.x, size.y, size.z) * 3.0 + 1.0
    ortho = max(size.x, size.z) * 1.15

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in [item.strip() for item in args.views.split(",") if item.strip()]:
        if name not in VIEWS:
            raise RuntimeError(f"unknown view {name!r}")
        camera = place_camera(name, VIEWS[name], centre, radius, ortho)
        bpy.context.scene.camera = camera
        destination = output_dir / f"{args.prefix}_{name}.png"
        bpy.context.scene.render.filepath = str(destination)
        bpy.ops.render.render(write_still=True)
        written.append(str(destination))
        print(f"PREVIEW_RENDERED={destination}", flush=True)
        bpy.data.objects.remove(camera, do_unlink=True)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(
                {
                    "input": args.input,
                    "frame": args.frame,
                    "renders": written,
                    "bounds_min": [float(v) for v in minimum],
                    "bounds_max": [float(v) for v in maximum],
                    "ortho_scale": float(ortho),
                    "engine": "BLENDER_WORKBENCH",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"PREVIEW_COUNT={len(written)}", flush=True)


if __name__ == "__main__":
    main()
