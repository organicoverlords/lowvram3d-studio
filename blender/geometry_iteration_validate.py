"""Fresh-import validation and neutral proof renders for geometry iterations."""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy
from mathutils import Vector

from common import (
    argv_after_double_dash,
    configure_render,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    look_at,
    reset_scene,
    save_json,
    select_only,
    welded_topology_stats,
    world_bounds,
)


def require_output(path: Path, label: str, minimum_bytes: int = 128) -> None:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise RuntimeError(f"Missing or too-small {label}: {path}")


def add_neutral_material(objects: list[bpy.types.Object]) -> None:
    material = bpy.data.materials.new("GeometryProofClay")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.32, 0.38, 0.42, 1.0)
        principled.inputs["Roughness"].default_value = 0.72
        principled.inputs["Metallic"].default_value = 0.0
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()


def add_lighting(center: Vector, scale: float) -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("GeometryProofWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.025, 0.025, 0.025, 1.0)
        background.inputs["Strength"].default_value = 0.35

    lights = (
        ("Key", (center.x - scale * 1.4, center.y - scale * 1.6, center.z + scale * 1.7), 1250.0, scale * 1.4),
        ("Fill", (center.x + scale * 1.5, center.y - scale * 0.8, center.z + scale * 0.5), 700.0, scale * 1.2),
        ("Rim", (center.x, center.y + scale * 1.5, center.z + scale * 1.4), 1000.0, scale * 1.0),
    )
    for name, location, energy, size in lights:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = max(size, 0.1)
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        look_at(obj, center)


def render_view(
    name: str,
    center: Vector,
    extent: Vector,
    direction: Vector,
    output: Path,
) -> None:
    distance = max(extent.x, extent.y, extent.z, 1.0) * 2.8
    camera_data = bpy.data.cameras.new(f"Camera_{name}")
    camera_data.type = "ORTHO"
    horizontal = extent.x if abs(direction.y) >= abs(direction.x) else extent.y
    camera_data.ortho_scale = max(extent.z, horizontal, 1e-3) * 1.22
    camera = bpy.data.objects.new(f"Camera_{name}", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + direction.normalized() * distance
    camera.location.z += extent.z * 0.035
    look_at(camera, center)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    require_output(output, f"{name} preview", 1024)
    bpy.data.objects.remove(camera, do_unlink=True)
    bpy.data.cameras.remove(camera_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--roundtrip-output", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--preview-dir", required=True)
    args = parser.parse_args(argv_after_double_dash())

    source = Path(args.input)
    roundtrip = Path(args.roundtrip_output)
    validation_path = Path(args.validation)
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    source_objects = import_mesh(str(source))
    if not source_objects:
        raise RuntimeError("Generated GLB contains no mesh objects")
    source_stats = extended_mesh_stats(source_objects)
    source_topology = welded_topology_stats(source_objects)
    if not source_stats.get("finite_bounds"):
        raise RuntimeError(f"Generated GLB has non-finite bounds: {source_stats}")
    if int(source_stats.get("triangles", 0)) < 10000:
        raise RuntimeError(
            f"Generated GLB is below the geometry floor: {source_stats.get('triangles')} triangles"
        )

    select_only(source_objects)
    export_glb(str(roundtrip), selected_only=True)
    require_output(roundtrip, "roundtrip GLB", 4096)

    # A second import is the actual validation boundary. Blender can exit zero after a Python
    # exception, so the caller also declares every output and verifies it independently.
    reset_scene()
    objects = import_mesh(str(roundtrip))
    if not objects:
        raise RuntimeError("Fresh Blender import of roundtrip GLB contains no meshes")
    fresh_stats = extended_mesh_stats(objects)
    fresh_topology = welded_topology_stats(objects)
    minimum, maximum = world_bounds(objects)
    extent = maximum - minimum
    center = (minimum + maximum) * 0.5
    source_triangles = int(source_stats.get("triangles", 0))
    fresh_triangles = int(fresh_stats.get("triangles", 0))
    triangle_delta = fresh_triangles - source_triangles
    triangle_ratio = fresh_triangles / max(source_triangles, 1)

    nonzero_extent = min(abs(extent.x), abs(extent.y), abs(extent.z)) > 1e-6
    success = (
        bool(objects)
        and bool(fresh_stats.get("finite_bounds"))
        and fresh_triangles >= 10000
        and nonzero_extent
        and abs(triangle_ratio - 1.0) <= 0.01
    )

    add_neutral_material(objects)
    add_lighting(center, max(extent.x, extent.y, extent.z, 1.0))
    configure_render(768, 768, transparent=True)
    views = {
        "front": Vector((0.0, -1.0, 0.0)),
        "three_quarter": Vector((0.72, -0.72, 0.0)),
        "side": Vector((1.0, 0.0, 0.0)),
        "back": Vector((0.0, 1.0, 0.0)),
    }
    preview_paths = {}
    for name, direction in views.items():
        output = preview_dir / f"{name}.png"
        render_view(name, center, extent, direction, output)
        preview_paths[name] = str(output)

    extent_values = {
        "x": float(extent.x),
        "y": float(extent.y),
        "z": float(extent.z),
    }
    longest_axis = max(extent_values, key=extent_values.get)
    report = {
        "success": success,
        "source": str(source),
        "roundtrip": str(roundtrip),
        "source_stats": source_stats,
        "fresh_import_stats": fresh_stats,
        "source_welded_topology": source_topology,
        "fresh_welded_topology": fresh_topology,
        "triangle_delta": triangle_delta,
        "triangle_ratio": triangle_ratio,
        "bounds_min": [float(value) for value in minimum],
        "bounds_max": [float(value) for value in maximum],
        "extent": extent_values,
        "longest_axis": longest_axis,
        "likely_sideways": longest_axis != "z",
        "previews": preview_paths,
        "scope": "geometry-only validation; no target FBX, textures, rigging or pose changes",
    }
    save_json(str(validation_path), report)
    require_output(validation_path, "validation report", 256)
    if not success:
        raise RuntimeError(f"Fresh geometry validation failed: {report}")


if __name__ == "__main__":
    main()
