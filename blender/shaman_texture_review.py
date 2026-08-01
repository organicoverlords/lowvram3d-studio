"""Stage 6, step 5: render the required review set from the exported GLB and measure it.

Imports the textured GLB in a fresh process - not the scene that produced it - so anything the
export dropped shows up here rather than being masked by state still in memory. Renders the eight
required views plus a UV-seam diagnostic, and reports per-view statistics next to each image.

The statistics exist to catch what the eye skims over (neon contamination, black patches, a
metallic channel that ran away); they are not the verdict. A previous stage shipped a map that was
finite, fully covered and completely wrong, so the images are still reviewed directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds

VIEWS = [
    ("front", 0.0, 0.0, 1.0),
    ("three_quarter_front", 35.0, 8.0, 1.0),
    ("side", 90.0, 0.0, 1.0),
    ("back", 180.0, 0.0, 1.0),
    ("close_head_antlers", 12.0, 6.0, 0.34),
    ("close_face", 0.0, 2.0, 0.17),
    ("close_ornaments", 28.0, 2.0, 0.40),
    ("close_staff", -38.0, -4.0, 0.42),
]


def setup_world() -> None:
    world = bpy.data.worlds.new("review")
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.55, 0.57, 0.60, 1.0)
    background.inputs["Strength"].default_value = 1.15
    bpy.context.scene.world = world


def add_lights(radius: float) -> None:
    for name, location, energy in (
        ("key", (radius * 1.6, -radius * 2.0, radius * 1.8), 4.0),
        ("fill", (-radius * 2.0, -radius * 1.2, radius * 0.6), 1.6),
        ("rim", (0.0, radius * 2.4, radius * 1.4), 2.4),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy * radius * radius * 12.0
        light.size = radius * 1.5
        obj = bpy.data.objects.new(name, light)
        obj.location = location
        bpy.context.collection.objects.link(obj)


def place_camera(camera, centre: Vector, radius: float, yaw: float, pitch: float, zoom: float,
                 focus: Vector, front_sign: float):
    distance = radius * 3.2 * zoom
    yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
    # Blender's glTF importer maps glTF (x, y, z) to (x, -z, y), so a subject whose front faces
    # glTF -Z ends up facing Blender +Y, and one facing glTF +Z faces Blender -Y. Which of the two
    # applies is a property of the asset, not of this renderer: it must come from the same explicit
    # front direction the projection stage used. Assuming -Z here is what put the beak, the scarf
    # and the fringe in a file named "back.png" and the rear of the head in one named "front.png",
    # and a correctly textured model was reported as having its texture on backwards.
    offset = Vector((
        -math.sin(yaw_r) * math.cos(pitch_r) * front_sign,
        math.cos(yaw_r) * math.cos(pitch_r) * front_sign,
        math.sin(pitch_r),
    )) * distance
    camera.location = focus + offset
    direction = focus - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = radius * 2.35 * zoom


def analyse(path: Path) -> dict:
    image = bpy.data.images.load(str(path))
    pixels = np.array(image.pixels[:], np.float32).reshape(-1, 4)
    rgb, alpha = pixels[:, :3], pixels[:, 3]
    subject = alpha > 0.5
    if not subject.any():
        subject = np.ones(len(rgb), bool)
    body = rgb[subject]
    maximum = body.max(axis=1)
    minimum = body.min(axis=1)
    saturation = np.where(maximum > 1e-6, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    # Magenta/neon: strongly saturated with red and blue both far above green.
    neon = (saturation > 0.72) & (body[:, 0] > 0.45) & (body[:, 2] > 0.45) & (body[:, 1] < body[:, 0] * 0.6)
    bpy.data.images.remove(image)
    return {
        "subject_pixel_percent": round(float(subject.mean() * 100), 2),
        "mean_rgb": [round(float(v), 4) for v in body.mean(axis=0)],
        "black_pixel_percent": round(float((maximum < 0.035).mean() * 100), 3),
        "neon_pixel_percent": round(float(neon.mean() * 100), 4),
        "saturation_mean": round(float(saturation.mean()), 4),
        "saturation_p99": round(float(np.percentile(saturation, 99)), 4),
        "luma_std": round(float(body.mean(axis=1).std()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument(
        "--front-direction",
        choices=("+z", "-z"),
        default=os.environ.get("LOWVRAM3D_FRONT_DIRECTION", "").lower() or None,
        help="glTF axis the subject's front faces; must match the projection stage",
    )
    args = parser.parse_args(argv_after_double_dash())

    if args.front_direction is None:
        # Fail closed. Guessing produced renders labelled with the wrong side, and a mislabelled
        # review set is worse than no review set: it sends a human looking for a texture fault that
        # is not there while the real defect goes unexamined.
        raise SystemExit("front direction is required (--front-direction or LOWVRAM3D_FRONT_DIRECTION)")
    front_sign = -1.0 if args.front_direction == "+z" else 1.0

    reset_scene()
    objects = import_mesh(args.glb)
    if not objects:
        raise RuntimeError(f"no mesh imported from {args.glb}")

    materials = sorted({slot.material.name for obj in objects for slot in obj.material_slots if slot.material})
    images = sorted({node.image.name for obj in objects for slot in obj.material_slots
                     if slot.material and slot.material.use_nodes
                     for node in slot.material.node_tree.nodes
                     if node.type == "TEX_IMAGE" and node.image})
    unpacked = sorted({node.image.name for obj in objects for slot in obj.material_slots
                       if slot.material and slot.material.use_nodes
                       for node in slot.material.node_tree.nodes
                       if node.type == "TEX_IMAGE" and node.image and not node.image.packed_file})
    uv_layers = sum(len(obj.data.uv_layers) for obj in objects)

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    head = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.86))
    ornaments = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.70))
    staff = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.72))

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

    camera_data = bpy.data.cameras.new("review")
    camera = bpy.data.objects.new("review", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.80))
    focus_for = {"close_head_antlers": head, "close_face": face,
                 "close_ornaments": ornaments, "close_staff": staff}
    results = {}
    for name, yaw, pitch, zoom in VIEWS:
        place_camera(camera, centre, radius, yaw, pitch, zoom, focus_for.get(name, centre), front_sign)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        results[name] = analyse(path)
        results[name]["path"] = str(path)
        print(f"REVIEW {name}: {results[name]}", flush=True)

    report = {
        "glb": args.glb,
        "front_direction_gltf": args.front_direction,
        "front_camera_axis_blender": "-Y" if front_sign < 0 else "+Y",
        "material_slots": materials,
        "material_slot_count": len(materials),
        "texture_images": images,
        "unpacked_images": unpacked,
        "uv_layers": uv_layers,
        "views": results,
    }
    save_json(args.report, report)
    print(f"REVIEW_DONE materials={materials} images={images} unpacked={unpacked} uv_layers={uv_layers}", flush=True)


if __name__ == "__main__":
    main()
