"""Render one rear view with all materials neutralized.

This is a diagnostic only. It imports the final GLB in a fresh Blender process,
replaces every material with the same flat emission colour, and uses the exact
review camera placement for the rear view. Any facial colour pattern that
disappears here was texture/material-derived; only silhouette, occlusion and
actual geometry remain in the unlit image.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds
from shaman_texture_review import add_lights, place_camera, setup_world


def neutralize(objects, colour=(0.5, 0.5, 0.5, 1.0)) -> None:
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            material.use_nodes = True
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
            if output is None:
                output = nodes.new("ShaderNodeOutputMaterial")
            emission = nodes.new("ShaderNodeEmission")
            emission.name = "QA_Flat_Neutral_Emission"
            emission.inputs["Color"].default_value = colour
            emission.inputs["Strength"].default_value = 1.0
            surface = output.inputs.get("Surface")
            if surface is not None:
                for link in list(surface.links):
                    links.remove(link)
                links.new(emission.outputs["Emission"], surface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.glb)
    if not objects:
        raise RuntimeError(f"no mesh imported from {args.glb}")
    neutralize(objects)

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    camera_data = bpy.data.cameras.new("rear_flat_diagnostic")
    camera = bpy.data.objects.new("rear_flat_diagnostic", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    # Match shaman_texture_review.py: the accepted asset's front is glTF -Z.
    place_camera(camera, centre, radius, 180.0, 0.0, 1.0, centre, 1.0)

    renders = {}
    for label, engine in (("unlit", "BLENDER_EEVEE"), ("lit", "CYCLES")):
        if engine == "BLENDER_EEVEE":
            scene.render.engine = engine
        else:
            scene.render.engine = engine
            scene.cycles.samples = args.samples
        path = output_dir / f"rear_flat_{label}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        renders[label] = str(path)

    report = {
        "diagnostic": "rear_flat_texture",
        "glb": str(args.glb),
        "neutral_material": {"type": "emission", "colour": [0.5, 0.5, 0.5, 1.0]},
        "camera": {"yaw_degrees": 180.0, "pitch_degrees": 0.0, "front_direction_gltf": "-z"},
        "renders": renders,
        "interpretation": "If a facial colour pattern disappears in the neutral unlit render, it was texture/material-derived; geometry silhouettes and occlusion remain.",
    }
    save_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
