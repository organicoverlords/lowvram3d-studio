"""Fresh-import flat-material proof that the bar repair changed only the bar.

Renders the canonical input and the repaired candidate through the same eight cameras
with one flat opaque material, so any silhouette difference is geometry and not shading,
and adds a wireframe close-up framed on the repaired region.
"""
from __future__ import annotations

import argparse
import json
import math
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
    ("front_three_quarter", 35.0, 8.0),
    ("rear_three_quarter", 215.0, 8.0),
)


def flat_material():
    material = bpy.data.materials.new("bar_repair_flat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.72, 0.72, 0.74, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def configure_scene(resolution: int, samples: int):
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
    return scene


def orthographic_camera(name: str):
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO"
    camera = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def aim(camera, centre: Vector, radius: float, yaw_deg: float, pitch_deg: float,
        span: float = 2.2, standoff: float | None = None):
    """``standoff`` decouples how far back the camera sits from how much it frames.

    A close-up needs a small ortho scale but still has to stand off past the whole mesh,
    otherwise the camera lands inside the body and renders its interior.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    direction = Vector((math.sin(yaw) * math.cos(pitch),
                        -math.cos(yaw) * math.cos(pitch),
                        math.sin(pitch)))
    camera.location = centre + direction * ((standoff if standoff is not None else radius) * 4.0)
    camera.data.ortho_scale = radius * span
    forward = (centre - camera.location).normalized()
    # "Y" keeps world +Z upright for the horizontal rings; at the poles Blender falls back
    # to a stable roll of its own, which is fine because top/bottom have no natural up.
    camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()


def render_set(glb: Path, output_dir: Path, tag: str, resolution: int, samples: int) -> dict:
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError(f"FRESH_IMPORT_NO_MESH:{tag}")
    material = flat_material()
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    scene = configure_scene(resolution, samples)
    camera = orthographic_camera(f"{tag}_camera")
    output_dir.mkdir(parents=True, exist_ok=True)
    views = {}
    for name, yaw, pitch in VIEWS:
        aim(camera, centre, radius, yaw, pitch)
        path = output_dir / f"{tag}_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        views[name] = str(path)
        print(f"BAR_REPAIR_VIEW {tag} {name} {path}", flush=True)
    return {
        "tag": tag,
        "glb": str(glb),
        "mesh": mesh_stats(objects),
        "centre": list(centre),
        "radius": radius,
        "views": views,
    }


def wireframe_closeup(glb: Path, output_dir: Path, tag: str, region_min, region_max,
                      resolution: int) -> dict:
    """glTF is Y-up and Blender is Z-up, so (x, y, z) arrives as (x, -z, y)."""
    reset_scene()
    objects = import_mesh(str(glb))
    material = flat_material()
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    corners = [Vector((x, -z, y))
               for x in (region_min[0], region_max[0])
               for y in (region_min[1], region_max[1])
               for z in (region_min[2], region_max[2])]
    lows = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    highs = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    centre = (lows + highs) * 0.5
    radius = max((highs - lows).length * 0.5, 1e-4)
    mesh_min, mesh_max = world_bounds(objects)
    standoff = max((mesh_max - mesh_min).length, 1e-3)
    scene = configure_scene(resolution, 8)
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("bar_repair_world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    for obj in objects:
        modifier = obj.modifiers.new("bar_repair_wire", "WIREFRAME")
        modifier.thickness = radius * 0.006
        modifier.use_replace = True
    camera = orthographic_camera(f"{tag}_wire_camera")
    paths = {}
    for name, yaw, pitch in (("wire_top", 0.0, 89.0), ("wire_front", 0.0, 0.0),
                             ("wire_three_quarter", 35.0, 25.0)):
        aim(camera, centre, radius, yaw, pitch, span=6.0, standoff=standoff)
        path = output_dir / f"{tag}_{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths[name] = str(path)
        print(f"BAR_REPAIR_WIRE {tag} {name} {path}", flush=True)
    return {"region_min_blender": list(lows), "region_max_blender": list(highs), "views": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--region-min", nargs=3, type=float, required=True)
    parser.add_argument("--region-max", nargs=3, type=float, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args(argv_after_double_dash())

    output_dir = Path(args.output_dir)
    report = {
        "schema": "bar_repair_visual_proof_v1",
        "flat_opaque_material": True,
        "orthographic": True,
        "fresh_import": True,
        "source": render_set(Path(args.source), output_dir, "source", args.resolution, args.samples),
        "candidate": render_set(Path(args.candidate), output_dir, "candidate",
                                args.resolution, args.samples),
        "wireframe_source": wireframe_closeup(Path(args.source), output_dir, "source",
                                              args.region_min, args.region_max, args.resolution),
        "wireframe_candidate": wireframe_closeup(Path(args.candidate), output_dir, "candidate",
                                                 args.region_min, args.region_max, args.resolution),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BAR_REPAIR_VISUAL_PROOF_DONE report={args.report}", flush=True)


if __name__ == "__main__":
    main()
