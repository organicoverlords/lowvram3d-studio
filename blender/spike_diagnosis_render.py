"""Render source and candidate with the exact production front camera and one flat material."""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds
from cpu_fallback_eight_view_qa import place_camera
from shaman_texture_review import add_lights, setup_world


def flat_material():
    material = bpy.data.materials.new("SpikeDiagnosisFlatOpaque")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.62, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def render_one(glb: Path, output_dir: Path, resolution: int, samples: int) -> dict:
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError(f"FRESH_IMPORT_NO_MESH:{glb}")
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    material = flat_material()
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    setup_world()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    camera_data = bpy.data.cameras.new("spike_diagnosis_camera")
    camera = bpy.data.objects.new("spike_diagnosis_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = {}
    for name, yaw, pitch in (("front", 0.0, 0.0), ("front_three_quarter", 35.0, 8.0)):
        # +Z is the established glTF front, hence the -1 sign used by the production QA runner.
        place_camera(camera, centre, radius, yaw, pitch, 1.0, centre, -1.0)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered[name] = str(path)
    return {
        "glb": str(glb),
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "centre": list(centre),
        "radius": radius,
        "front_direction_gltf": "+z",
        "rendered": rendered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv_after_double_dash())
    root = Path(args.output_dir)
    report = {
        "schema": "spike_diagnosis_flat_render_v1",
        "same_camera": True,
        "flat_opaque_material": True,
        "source": render_one(Path(args.source), root / "source", args.resolution, args.samples),
        "candidate": render_one(Path(args.candidate), root / "candidate", args.resolution, args.samples),
    }
    save_json(args.report, report)
    print(f"SPIKE_FLAT_RENDER_DONE report={args.report}", flush=True)


if __name__ == "__main__":
    main()
