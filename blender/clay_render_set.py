"""Orientation-normalised clay renders for comparing meshes from different sources.

Untextured on purpose. Comparing a textured candidate against an untextured one, or against a
reference whose materials came from a different tool, compares shading and albedo as much as form -
and the question here is only about form. A single neutral material removes that confound.

The up axis, its sign and the lateral axis are supplied by the caller rather than assumed. Exporters
disagree about which axis is up, and assuming Y once put a reference's flank where its head should
have been and rendered another upside down while every number still looked reasonable.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds

# (name, yaw degrees, pitch degrees, zoom) - yaw 0 looks at the front.
VIEWS = [
    ("front", 0.0, 0.0, 1.0),
    ("left_three_quarter", -40.0, 8.0, 1.0),
    ("right_three_quarter", 40.0, 8.0, 1.0),
    ("side", 90.0, 0.0, 1.0),
    ("close_head", 0.0, 4.0, 0.30),
]


def clay_material():
    material = bpy.data.materials.new("Clay")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.57, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.72
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.28
    return material


def world_vertices(objects) -> "object":
    import numpy as np

    clouds = []
    for obj in objects:
        raw = np.empty(len(obj.data.vertices) * 3, np.float32)
        obj.data.vertices.foreach_get("co", raw)
        matrix = np.array(obj.matrix_world)
        clouds.append(raw.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3])
    return np.vstack(clouds)


def measure_orientation(objects, tri_areas=None):
    """Measure up and lateral axes from the mesh as it sits in Blender, after import.

    Deliberately measured here rather than translated from glTF space. The importer already applies
    a Y-up to Z-up conversion, but whether a given file needed it depends on how that file was
    authored, so a fixed glTF-to-Blender axis map is right for some assets and wrong for others.
    Applying one to a reference that had already arrived upright rotated it onto its side, and the
    receipt still recorded the axis it had "detected" as correct. Measuring after import removes the
    translation, and with it the chance of translating the wrong way.
    """
    import numpy as np

    points = world_vertices(objects)
    extent = points.max(0) - points.min(0)
    scale = max(float(np.linalg.norm(extent)), 1e-12)

    # Occupancy-grid symmetry: cheap, needs no KD-tree, and is enough to pick the mirror axis.
    grid = 48
    normalised = (points - points.min(0)) / np.maximum(extent, 1e-12)
    index = np.clip((normalised * (grid - 1)).astype(np.int32), 0, grid - 1)
    occupancy = np.zeros((grid, grid, grid), bool)
    occupancy[index[:, 0], index[:, 1], index[:, 2]] = True

    scores = []
    for axis in range(3):
        mirrored = np.flip(occupancy, axis=axis)
        union = np.logical_or(occupancy, mirrored).sum()
        scores.append(float(np.logical_and(occupancy, mirrored).sum() / max(union, 1)))
    lateral = int(np.argmax(scores))
    remaining = [a for a in range(3) if a != lateral]
    up = remaining[int(np.argmax(extent[remaining]))]

    # The heavy end is down, measured by surface area rather than by vertex count. Tessellation is
    # not uniform on these assets: the antler rack, cords and pendants carry far more vertices than
    # the large smooth robe, so counting vertices declares the head the heavy end and turns the
    # subject upside down - which is what happened to the reference, while the axis it reported
    # stayed correct.
    lower_area = upper_area = 0.0
    midpoint = 0.5 * (points[:, up].min() + points[:, up].max())
    for obj in objects:
        mesh = obj.data
        mesh.calc_loop_triangles()
        matrix = np.array(obj.matrix_world)
        raw = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", raw)
        verts = raw.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        loops = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", loops)
        tris = verts[loops.reshape(-1, 3)]
        areas = 0.5 * np.linalg.norm(
            np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        centres = tris.mean(axis=1)[:, up]
        lower_area += float(areas[centres < midpoint].sum())
        upper_area += float(areas[centres >= midpoint].sum())
    sign = 1.0 if lower_area >= upper_area else -1.0
    return up, sign, lateral, scores, scale


def orient(objects, forced_up: int = -1, forced_sign: float = 0.0, forced_lateral: int = -1) -> dict:
    """Stand the subject up using axes measured in Blender space."""
    measured_up, measured_sign, measured_lateral, scores, _ = measure_orientation(objects)
    up = measured_up if forced_up < 0 else forced_up
    sign = measured_sign if forced_sign == 0.0 else forced_sign
    lateral = measured_lateral if forced_lateral < 0 else forced_lateral

    up_vector = Vector((0.0, 0.0, 0.0))
    up_vector[up] = sign
    rotation = up_vector.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4()

    lateral_vector = Vector((0.0, 0.0, 0.0))
    lateral_vector[lateral] = 1.0
    rotated = rotation @ lateral_vector
    flat = Vector((rotated.x, rotated.y, 0.0))
    if flat.length > 1e-6:
        rotation = Matrix.Rotation(-math.atan2(flat.y, flat.x), 4, "Z") @ rotation

    for obj in objects:
        obj.matrix_world = rotation @ obj.matrix_world
    return {"measured_up_axis": "xyz"[measured_up], "measured_up_sign": measured_sign,
            "measured_lateral_axis": "xyz"[measured_lateral],
            "render_up_axis": "xyz"[up], "render_up_sign": sign,
            "render_lateral_axis": "xyz"[lateral],
            "symmetry_scores": [round(s, 4) for s in scores]}


def setup_world() -> None:
    world = bpy.data.worlds.new("clay")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.42, 0.44, 0.47, 1.0)
    background.inputs["Strength"].default_value = 1.1
    bpy.context.scene.world = world


def add_lights(radius: float) -> None:
    for name, location, energy in (
        ("key", (radius * 1.8, -radius * 2.2, radius * 1.9), 4.2),
        ("fill", (-radius * 2.2, -radius * 1.4, radius * 0.7), 1.7),
        ("rim", (0.0, radius * 2.6, radius * 1.5), 2.6),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy * radius * radius * 12.0
        light.size = radius * 1.6
        obj = bpy.data.objects.new(name, light)
        obj.location = location
        bpy.context.collection.objects.link(obj)


def place_camera(camera, radius: float, yaw: float, pitch: float, zoom: float, focus: Vector):
    distance = radius * 3.2 * zoom
    yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
    offset = Vector((
        -math.sin(yaw_r) * math.cos(pitch_r),
        -math.cos(yaw_r) * math.cos(pitch_r),
        math.sin(pitch_r),
    )) * distance
    camera.location = focus + offset
    camera.rotation_euler = (focus - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = radius * 2.35 * zoom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    # Kept for callers that still pass them, but the axes are measured after import rather than
    # taken on trust: a caller's glTF-space answer has to be translated, and the translation is what
    # went wrong.
    parser.add_argument("--up-axis", type=int, default=-1, help="ignored; measured after import")
    parser.add_argument("--up-sign", type=float, default=1.0, help="ignored; measured after import")
    parser.add_argument("--lateral-axis", type=int, default=-1, help="ignored; measured after import")
    parser.add_argument("--force-up-axis", type=int, default=-1, choices=(-1, 0, 1, 2))
    parser.add_argument("--force-up-sign", type=float, default=0.0)
    parser.add_argument("--force-lateral-axis", type=int, default=-1, choices=(-1, 0, 1, 2))
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.glb)
    if not objects:
        raise SystemExit(f"no mesh imported from {args.glb}")
    orientation = orient(objects, args.force_up_axis, args.force_up_sign, args.force_lateral_axis)

    material = clay_material()
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    head = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.82))

    setup_world()
    add_lights(radius)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

    camera_data = bpy.data.cameras.new("clay")
    camera = bpy.data.objects.new("clay", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = {}
    for name, yaw, pitch, zoom in VIEWS:
        place_camera(camera, radius, yaw, pitch, zoom, head if name == "close_head" else centre)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered[name] = str(path)
        print(f"CLAY {args.label or Path(args.glb).stem} {name}", flush=True)

    save_json(args.report, {
        "glb": args.glb,
        "label": args.label,
        "renders": rendered,
        **orientation,
    })
    print(f"CLAY_DONE {args.label or Path(args.glb).stem} views={len(rendered)}", flush=True)


if __name__ == "__main__":
    main()
