"""CPU Blender helper for bounded orientation and colour post-processing."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from common import export_glb, import_mesh, reset_scene, world_bounds
from shaman_texture_review import add_lights, place_camera, setup_world


def prepare(objects, resolution: int, samples: int):
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    setup_world()
    add_lights(radius)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    camera_data = bpy.data.cameras.new("postprocess_camera")
    camera = bpy.data.objects.new("postprocess_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera, centre, radius


def render_yaws(glb: Path, output_dir: Path, front_direction: str, resolution: int, samples: int):
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError(f"no mesh imported from {glb}")
    scene, camera, centre, radius = prepare(objects, resolution, samples)
    front_sign = -1.0 if front_direction == "+z" else 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    for yaw in (0, 90, 180, 270):
        place_camera(camera, centre, radius, yaw, 0.0, 1.0, centre, front_sign)
        path = output_dir / f"yaw_{yaw}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    place_camera(camera, centre, radius, 35.0, 8.0, 1.0, centre, front_sign)
    scene.render.filepath = str(output_dir / "three_quarter.png")
    bpy.ops.render.render(write_still=True)


def rotate_export(glb: Path, output_glb: Path, yaw_deg: float):
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError(f"no mesh imported from {glb}")
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    root = bpy.data.objects.new("orientation_root_fix", None)
    bpy.context.collection.objects.link(root)
    root.location = centre
    for obj in objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world_matrix
    root.rotation_euler.z = math.radians(float(yaw_deg))
    export_glb(str(output_glb), selected_only=False)


def replace_basecolor_export(glb: Path, basecolor: Path, output_glb: Path):
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError(f"no mesh imported from {glb}")
    replacement = bpy.data.images.load(str(basecolor), check_existing=False)
    replacement.pack()
    replaced = 0
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                socket = node.inputs.get("Base Color")
                if not socket or not socket.is_linked:
                    continue
                source = socket.links[0].from_node
                if source.type == "TEX_IMAGE":
                    source.image = replacement
                    replaced += 1
    if replaced == 0:
        raise RuntimeError("BASECOLOR_NODE_NOT_FOUND")
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    export_glb(str(output_glb), selected_only=False)


if __name__ == "__main__":
    # Blender appends arguments after ``--`` to sys.argv; keeping the parser local avoids any
    # interaction with Blender's own options.
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("render_yaws", "rotate_export", "replace_basecolor_export"), required=True)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-glb", default="")
    parser.add_argument("--basecolor", default="")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--front-direction", choices=("+z", "-z"), default="-z")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args(argv)
    glb = Path(args.glb)
    if args.mode == "render_yaws":
        render_yaws(glb, Path(args.output_dir), args.front_direction, args.resolution, args.samples)
    elif args.mode == "rotate_export":
        rotate_export(glb, Path(args.output_glb), args.yaw_deg)
    else:
        replace_basecolor_export(glb, Path(args.basecolor), Path(args.output_glb))
