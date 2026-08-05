"""Fresh-import eight-view and close-up QA of a finished textured GLB.

The producing run's own word is not evidence: this reimports the GLB in a clean Blender
process, checks that a base-colour image actually resolved, and renders the eight review
angles plus close-ups of the features the asset is judged on - face, equipment, tail and
the repaired bar region.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, mesh_stats, reset_scene, world_bounds

VIEWS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("rear", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("top", 0.0, 89.0),
    ("bottom", 0.0, -89.0),
    ("front_three_quarter", 35.0, 12.0),
    ("rear_three_quarter", 215.0, 12.0),
)
#: Close-ups as (name, yaw, pitch, height fraction from the model's base, framing width).
CLOSEUPS = (
    ("face", 0.0, 5.0, 0.86, 0.42),
    ("rear_head", 180.0, 5.0, 0.86, 0.42),
    ("equipment", 200.0, 10.0, 0.60, 0.60),
    ("tail", 300.0, -5.0, 0.30, 0.55),
    ("repaired_bar_region", 0.0, -30.0, 0.12, 0.70),
)


def texture_report(objects) -> dict:
    images, materials = set(), set()
    active_images = {}
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            materials.add(material.name)
            if not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    images.add((node.image.name, tuple(node.image.size),
                                bool(node.image.has_data)))
            active = active_base_color_image(material)
            active_images[material.name] = {
                "name": active.name if active is not None else None,
                "size": list(active.size) if active is not None else None,
                "has_data": bool(active is not None and active.has_data),
            }
    return {
        "materials": sorted(materials),
        "images": [{"name": n, "size": list(s), "has_data": d} for n, s, d in sorted(images)],
        "active_base_color_images": active_images,
        "base_colour_image_resolved": any(v["has_data"] and v["size"][0] > 0
                                            for v in active_images.values() if v["size"]),
    }


def active_base_color_node(material):
    """Resolve the TEX_IMAGE feeding Principled Base Color, never the first image node."""
    if material is None or not material.use_nodes:
        return None
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None or "Base Color" not in principled.inputs:
        return None
    queue = deque(link.from_node for link in principled.inputs["Base Color"].links)
    visited = set()
    while queue:
        node = queue.popleft()
        if node.name in visited:
            continue
        visited.add(node.name)
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node
        for socket in node.inputs:
            queue.extend(link.from_node for link in socket.links)
    return None


def active_base_color_image(material):
    node = active_base_color_node(material)
    return node.image if node is not None else None


def make_unlit(objects) -> int:
    """Rewire every material to emit its own base-colour texture.

    A recessed feature - a face inside a ghillie hood - can be invisible under lights and
    still be present in the atlas. Emission removes shading from the question entirely.
    """
    converted = 0
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            source = active_base_color_node(material)
            output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
            if output is None:
                continue
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Strength"].default_value = 1.0
            if source is not None:
                links.new(source.outputs["Color"], emission.inputs["Color"])
            else:
                principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
                if principled is not None and "Base Color" in principled.inputs:
                    emission.inputs["Color"].default_value = principled.inputs["Base Color"].default_value
            links.new(emission.outputs["Emission"], output.inputs["Surface"])
            converted += 1
    return converted


def setup(resolution: int, samples: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("qa_world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.52, 1.0)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.2
    return scene


def add_lights(centre: Vector, radius: float):
    for name, offset, energy in (("key", (1.0, -1.4, 1.2), 6.0),
                                 ("fill", (-1.3, -0.9, 0.4), 2.5),
                                 ("rim", (0.2, 1.5, 0.9), 3.5)):
        data = bpy.data.lights.new(f"qa_{name}", "AREA")
        data.energy = energy * (radius ** 2) * 40.0
        data.size = radius * 1.6
        light = bpy.data.objects.new(f"qa_{name}", data)
        light.location = centre + Vector(offset) * radius * 3.0
        direction = (centre - light.location).normalized()
        light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(light)


def orthographic_camera():
    data = bpy.data.cameras.new("qa_camera")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("qa_camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def aim(camera, target: Vector, standoff: float, yaw_deg: float, pitch_deg: float, width: float):
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    direction = Vector((math.sin(yaw) * math.cos(pitch),
                        -math.cos(yaw) * math.cos(pitch),
                        math.sin(pitch)))
    camera.location = target + direction * standoff * 4.0
    camera.data.ortho_scale = width
    camera.rotation_euler = (target - camera.location).normalized().to_track_quat(
        "-Z", "Y").to_euler()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--unlit", action="store_true",
                        help="render base colour as emission, with no lights")
    args = parser.parse_args(argv_after_double_dash())

    glb = Path(args.glb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError("TEXTURED_QA_FRESH_IMPORT_NO_MESH")
    textures = texture_report(objects)
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    height = maximum.z - minimum.z

    scene = setup(args.resolution, args.samples)
    unlit_materials = make_unlit(objects) if args.unlit else 0
    if not args.unlit:
        add_lights(centre, radius)
    else:
        scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    camera = orthographic_camera()

    views = {}
    for name, yaw, pitch in VIEWS:
        aim(camera, centre, radius, yaw, pitch, radius * 2.15)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        views[name] = str(path)
        print(f"TEXTURED_QA_VIEW {name} {path}", flush=True)

    closeups = {}
    for name, yaw, pitch, level, width in CLOSEUPS:
        target = Vector((centre.x, centre.y, minimum.z + height * level))
        aim(camera, target, radius, yaw, pitch, radius * width)
        path = output_dir / f"closeup_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        closeups[name] = str(path)
        print(f"TEXTURED_QA_CLOSEUP {name} {path}", flush=True)

    report = {
        "schema": "textured_asset_qa_v1",
        "glb": str(glb),
        "glb_bytes": glb.stat().st_size,
        "fresh_import": True,
        "mesh": mesh_stats(objects),
        "textures": textures,
        "render_engine": scene.render.engine,
        "samples": args.samples,
        "unlit": bool(args.unlit),
        "unlit_materials_converted": unlit_materials,
        "resolution": args.resolution,
        "views": views,
        "closeups": closeups,
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TEXTURED_QA_DONE views={len(views)} closeups={len(closeups)}", flush=True)


if __name__ == "__main__":
    main()
