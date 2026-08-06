"""Render a deterministic nine-view GLB diagnostic set in Blender."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--engine", choices=("auto", "BLENDER_EEVEE", "CYCLES"), default="auto")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("GLB_DIAGNOSTIC_NO_MESH_BOUNDS")
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return minimum, maximum


def add_area_light(name: str, location: Vector, target: Vector, energy: float, size: float, color: tuple[float, float, float]) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = arguments()
    args.input = args.input.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    minimum, maximum = world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    largest = max(extent.x, extent.y, extent.z, 1e-3)
    radius = largest * 3.2

    scene = bpy.context.scene
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    requested_engine = args.engine
    if requested_engine == "auto":
        requested_engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.engine = requested_engine
    if requested_engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        scene.cycles.use_denoising = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("DiagnosticWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.06, 0.07, 1.0)
    background.inputs["Strength"].default_value = 0.75
    scene.world = world

    add_area_light("Key", center + Vector((-radius, -radius, radius)), center, 1100.0, largest * 2.0, (1.0, 0.94, 0.86))
    add_area_light("Fill", center + Vector((radius, -radius * 0.6, radius * 0.35)), center, 650.0, largest * 2.5, (0.80, 0.90, 1.0))
    add_area_light("Rim", center + Vector((0.0, radius, radius * 0.6)), center, 900.0, largest * 1.8, (1.0, 1.0, 1.0))

    camera_data = bpy.data.cameras.new("DiagnosticCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = largest * 1.22
    camera = bpy.data.objects.new("DiagnosticCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    views = [
        ("01_front", Vector((0.0, -1.0, 0.04))),
        ("02_front_left", Vector((-1.0, -1.0, 0.06))),
        ("03_left", Vector((-1.0, 0.0, 0.04))),
        ("04_rear_left", Vector((-1.0, 1.0, 0.06))),
        ("05_rear", Vector((0.0, 1.0, 0.04))),
        ("06_rear_right", Vector((1.0, 1.0, 0.06))),
        ("07_right", Vector((1.0, 0.0, 0.04))),
        ("08_front_right", Vector((1.0, -1.0, 0.06))),
        ("09_bottom", Vector((0.0, -0.08, -1.0))),
    ]
    rendered = []
    for name, direction in views:
        direction.normalize()
        camera.location = center + direction * radius
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        output = args.out_dir / f"{name}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        rendered.append(output.name)

    manifest = {
        "schema": "glb_diagnostic_render_set_v1",
        "label": args.label,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "resolution": args.resolution,
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "view_order": rendered,
        "camera": "orthographic; front is -Y; deterministic nine-view set",
        "classification": "VISUAL_DIAGNOSTIC_ONLY",
    }
    (args.out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
