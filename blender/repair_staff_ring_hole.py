"""Restore a real through-hole in the shaman staff ring and prove it visually.

The input mesh is never modified. The model is temporarily orientation-normalised only so the
staff ring can be found from its silhouette; original object transforms are restored before export.
A cylindrical Boolean cut is accepted only when the centre becomes transparent in a fresh render,
the surrounding annulus remains present, and a direct ray through the ring no longer hits the
modified mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from clay_render_set import orient
from common import (
    argv_after_double_dash,
    export_glb,
    import_mesh,
    preferred_render_engine,
    reset_scene,
    save_json,
    select_only,
    world_bounds,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_count(objects) -> int:
    total = 0
    for obj in objects:
        obj.data.calc_loop_triangles()
        total += len(obj.data.loop_triangles)
    return total


def emission_material(name: str, colour: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = colour
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def clay_material():
    material = bpy.data.materials.new("StaffHoleClay")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.62, 0.60, 0.57, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.72
    return material


def setup_world() -> None:
    world = bpy.data.worlds.new("StaffHoleWorld")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world


def setup_camera(objects, resolution: int):
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    extent = maximum - minimum
    ortho = max(extent.x, extent.z) * 1.08
    depth = max(extent.y, ortho * 0.1)

    camera_data = bpy.data.cameras.new("StaffHoleCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho
    camera = bpy.data.objects.new("StaffHoleCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = Vector((centre.x, minimum.y - depth * 3.0, centre.z))
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = preferred_render_engine()
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    return camera, minimum, maximum, centre, ortho


def render(path: Path, material) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].material_override = material
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    scene.view_layers[0].material_override = None


def alpha_mask(path: Path) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        return np.flipud(pixels[:, :, 3] > 0.5)
    finally:
        bpy.data.images.remove(image)


def directional_radius(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    left = np.zeros((height, width), dtype=np.int16)
    right = np.zeros_like(left)
    up = np.zeros_like(left)
    down = np.zeros_like(left)

    for row in range(height):
        run = 0
        for col in range(width):
            run = run + 1 if mask[row, col] else 0
            left[row, col] = run
        run = 0
        for col in range(width - 1, -1, -1):
            run = run + 1 if mask[row, col] else 0
            right[row, col] = run

    for col in range(width):
        run = 0
        for row in range(height):
            run = run + 1 if mask[row, col] else 0
            up[row, col] = run
        run = 0
        for row in range(height - 1, -1, -1):
            run = run + 1 if mask[row, col] else 0
            down[row, col] = run

    return np.minimum.reduce((left, right, up, down))


def detect_ring(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("silhouette render is empty")

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    radius = directional_radius(mask)

    allowed = np.zeros_like(mask, dtype=bool)
    top_end = ymin + max(1, int(height * 0.34))
    side = max(1, int(width * 0.29))
    allowed[ymin:top_end, xmin : xmin + side] = True
    allowed[ymin:top_end, xmax - side + 1 : xmax + 1] = True
    scores = np.where(allowed & mask, radius, 0)

    row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    outer_radius = int(scores[row, col])
    if outer_radius < 6:
        raise RuntimeError(f"staff ring was not detected reliably; radius={outer_radius}px")

    hole_radius = max(4.0, outer_radius * 0.43)
    return {
        "centre_px": [int(col), int(row)],
        "outer_radius_px": outer_radius,
        "hole_radius_px": float(hole_radius),
        "foreground_bounds_px": [xmin, ymin, xmax, ymax],
        "detected_side": "left" if col < (xmin + xmax) * 0.5 else "right",
    }


def pixel_to_world(col: float, row: float, centre: Vector, ortho: float, resolution: int):
    world_per_pixel = ortho / float(resolution)
    x = centre.x + (col + 0.5 - resolution * 0.5) * world_per_pixel
    z = centre.z + (resolution * 0.5 - row - 0.5) * world_per_pixel
    return x, z, world_per_pixel


def nearest_object(objects, x: float, z: float):
    winner = None
    winner_distance = float("inf")
    for obj in objects:
        raw = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", raw)
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        world = raw.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
        distance = np.square(world[:, 0] - x) + np.square(world[:, 2] - z)
        local_min = float(distance.min()) if len(distance) else float("inf")
        if local_min < winner_distance:
            winner_distance = local_min
            winner = obj
    if winner is None:
        raise RuntimeError("no mesh object could be associated with the staff ring")
    return winner, math.sqrt(winner_distance)


def add_cutter(x: float, y: float, z: float, radius: float, depth: float):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=radius,
        depth=depth,
        end_fill_type="NGON",
        location=(x, y, z),
        rotation=(math.pi * 0.5, 0.0, 0.0),
    )
    cutter = bpy.context.active_object
    cutter.name = "StaffRingHoleCutter"
    return cutter


def apply_boolean(target, cutter) -> None:
    select_only([target])
    bpy.context.view_layer.objects.active = target
    modifier = target.modifiers.new("RestoreStaffRingThroughHole", "BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.solver = "EXACT"
    modifier.object = cutter
    if hasattr(modifier, "use_hole_tolerant"):
        modifier.use_hole_tolerant = True
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def disk_ratio(mask: np.ndarray, centre_px, inner: float, outer: float | None = None) -> float:
    col, row = centre_px
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    distance = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)
    region = distance <= inner if outer is None else ((distance >= inner) & (distance <= outer))
    return float(mask[region].mean()) if np.any(region) else 1.0


def ray_hits(target, x: float, z: float, minimum: Vector, maximum: Vector) -> bool:
    start_world = Vector((x, minimum.y - max(maximum.y - minimum.y, 1.0), z))
    end_world = Vector((x, maximum.y + max(maximum.y - minimum.y, 1.0), z))
    inverse = target.matrix_world.inverted()
    start_local = inverse @ start_world
    end_local = inverse @ end_world
    direction = end_local - start_local
    distance = direction.length
    if distance <= 1e-9:
        return True
    hit, *_ = target.ray_cast(start_local, direction.normalized(), distance=distance)
    return bool(hit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args(argv_after_double_dash())

    source = Path(args.input)
    output = Path(args.output)
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256(source)

    reset_scene()
    objects = import_mesh(str(source))
    if not objects:
        raise RuntimeError("input GLB contains no mesh objects")
    original_matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    triangles_before = triangle_count(objects)
    orientation = orient(objects)

    setup_world()
    camera, minimum, maximum, centre, ortho = setup_camera(objects, args.resolution)
    silhouette = emission_material("StaffHoleSilhouette", (1.0, 1.0, 1.0, 1.0))
    clay = clay_material()

    before_mask_path = evidence / "before_mask.png"
    render(before_mask_path, silhouette)
    before_mask = alpha_mask(before_mask_path)
    detection = detect_ring(before_mask)

    col, row = detection["centre_px"]
    x, z, world_per_pixel = pixel_to_world(col, row, centre, ortho, args.resolution)
    outer_world = detection["outer_radius_px"] * world_per_pixel
    hole_world = detection["hole_radius_px"] * world_per_pixel
    detection["centre_world_oriented"] = [x, centre.y, z]
    detection["outer_radius_world"] = outer_world
    detection["hole_radius_world"] = hole_world

    camera.location = Vector((x, minimum.y - max(maximum.y - minimum.y, ortho) * 2.5, z))
    camera.rotation_euler = (Vector((x, centre.y, z)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = outer_world * 5.5
    before_clay = evidence / "before_staff_close.png"
    render(before_clay, clay)

    camera, minimum, maximum, centre, ortho = setup_camera(objects, args.resolution)
    target, nearest_distance = nearest_object(objects, x, z)
    cutter_depth = max((maximum.y - minimum.y) * 3.0, hole_world * 8.0)
    cutter = add_cutter(x, centre.y, z, hole_world, cutter_depth)
    apply_boolean(target, cutter)
    triangles_after = triangle_count(objects)

    after_mask_path = evidence / "after_mask.png"
    render(after_mask_path, silhouette)
    after_mask = alpha_mask(after_mask_path)
    centre_foreground_ratio = disk_ratio(after_mask, detection["centre_px"], detection["hole_radius_px"] * 0.62)
    annulus_foreground_ratio = disk_ratio(
        after_mask,
        detection["centre_px"],
        detection["hole_radius_px"] * 1.28,
        detection["hole_radius_px"] * 2.05,
    )
    ray_blocked = ray_hits(target, x, z, minimum, maximum)

    camera.location = Vector((x, minimum.y - max(maximum.y - minimum.y, ortho) * 2.5, z))
    camera.rotation_euler = (Vector((x, centre.y, z)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = outer_world * 5.5
    after_clay = evidence / "after_staff_close.png"
    render(after_clay, clay)

    passed = centre_foreground_ratio <= 0.08 and annulus_foreground_ratio >= 0.42 and not ray_blocked
    if not passed:
        raise RuntimeError(
            "staff hole validation failed: "
            f"centre={centre_foreground_ratio:.4f} annulus={annulus_foreground_ratio:.4f} "
            f"ray_blocked={ray_blocked}"
        )

    for obj in objects:
        obj.matrix_world = original_matrices[obj.name]
    select_only(objects)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_glb(str(output), selected_only=True)

    source_hash_after = sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("input GLB changed during repair")

    report = {
        "passed": True,
        "operation": "restore_staff_ring_through_hole",
        "input": str(source),
        "input_sha256": source_hash_after,
        "output": str(output),
        "output_sha256": sha256(output),
        "input_unchanged": True,
        "triangles_before": triangles_before,
        "triangles_after": triangles_after,
        "triangle_delta": triangles_after - triangles_before,
        "target_object": target.name,
        "nearest_target_vertex_distance_world": nearest_distance,
        "detection": detection,
        "validation": {
            "centre_foreground_ratio": centre_foreground_ratio,
            "annulus_foreground_ratio": annulus_foreground_ratio,
            "ray_blocked_after_cut": ray_blocked,
            "real_through_hole_proven": True,
        },
        "evidence": {
            "before_mask": str(before_mask_path),
            "after_mask": str(after_mask_path),
            "before_staff_close": str(before_clay),
            "after_staff_close": str(after_clay),
        },
        "orientation": orientation,
    }
    save_json(args.report, report)
    print(
        "STAFF_HOLE_REPAIR_PASS "
        f"target={target.name} triangles={triangles_before}->{triangles_after} "
        f"centre={centre_foreground_ratio:.4f} annulus={annulus_foreground_ratio:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
