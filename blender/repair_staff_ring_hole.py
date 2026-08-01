"""Restore a real through-hole in the shaman staff ring using a localized patch repair.

The input mesh is never modified. The model is temporarily orientation-normalised only so the
staff ring can be found from its silhouette; original object transforms are restored before export.

This script does NOT run a Boolean against the fused million-triangle body. The staff-ring region
is isolated into a small patch, the cylindrical Boolean is applied to that patch alone, and the
patch is merged straight back.

How the patch is merged depends on the source. A corner-split source (three vertices per triangle,
as exported by glTF) shares no vertex between patch and remainder, so there is no seam to stitch:
welding is skipped, and the corner splits that Blender's EXACT solver merges are restored instead.
Only a source with genuinely shared vertices is welded, at WELD_TOLERANCE, along the patch boundary.

Geometry outside the repair box is then proven unchanged: no vertex position may move, appear or
disappear, and no vertex may be collapsed away.

Modes:
    --preflight-only     detect the ring, size the patch, render the detected region, stop.
    --localized-repair   run preflight, then cut, merge, validate and export.

Every long stage is deadline-bounded with a heartbeat; a stage that overruns hard-kills the
process rather than stalling silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import hashlib
import json
import math
import os
import threading
import time

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
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

IMPORT_BUDGET_SECONDS = 180.0
PREFLIGHT_BUDGET_SECONDS = 30.0
BOOLEAN_BUDGET_SECONDS = 120.0
HEARTBEAT_SECONDS = 10.0
MAX_PATCH_FACE_FRACTION = 0.05
WELD_TOLERANCE = 1e-6
CLAY_WORLD_STRENGTH = 0.35

_STAGE_TIMINGS: dict[str, float] = {}


class Stage:
    """Run a stage under a hard deadline with a heartbeat, or kill the process."""

    def __init__(self, name: str, budget: float) -> None:
        self.name = name
        self.budget = budget
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0.0
        self.cpu_started = 0.0

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            elapsed = time.monotonic() - self.started
            if elapsed >= self.budget:
                cpu = time.process_time() - self.cpu_started
                print(
                    f"STAGE_TIMEOUT stage={self.name} elapsed={elapsed:.1f}s "
                    f"budget={self.budget:.0f}s pid={os.getpid()} cpu_delta={cpu:.1f}s",
                    flush=True,
                )
                print("STAFF_HOLE_REPAIR_FAIL reason=stage_timeout", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(3)
            if int(elapsed) and int(elapsed) % int(HEARTBEAT_SECONDS) == 0:
                cpu = time.process_time() - self.cpu_started
                print(
                    f"HEARTBEAT stage={self.name} elapsed={elapsed:.0f}s "
                    f"budget={self.budget:.0f}s pid={os.getpid()} cpu_delta={cpu:.1f}s",
                    flush=True,
                )
                self._stop.wait(1.0)

    def __enter__(self) -> "Stage":
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        print(f"STAGE_BEGIN {self.name} budget={self.budget:.0f}s pid={os.getpid()}", flush=True)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        elapsed = time.monotonic() - self.started
        _STAGE_TIMINGS[self.name] = round(elapsed, 3)
        cpu = time.process_time() - self.cpu_started
        print(
            f"STAGE_END {self.name} elapsed={elapsed:.2f}s cpu_delta={cpu:.2f}s",
            flush=True,
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


# --------------------------------------------------------------------------------------
# mesh array helpers
# --------------------------------------------------------------------------------------


def vertex_world(obj) -> np.ndarray:
    raw = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", raw)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    return raw.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def polygon_topology(obj):
    mesh = obj.data
    count = len(mesh.polygons)
    loop_total = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    loop_vert = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    poly_of_loop = np.repeat(np.arange(count, dtype=np.int32), loop_total)
    return count, loop_vert, poly_of_loop


def outside_box_snapshot(objects, centre, half):
    """Sorted coordinates of every vertex outside the repair box, across all objects."""
    chunks = []
    for obj in objects:
        world = vertex_world(obj)
        delta = np.abs(world - np.asarray(centre, dtype=np.float64))
        inside = np.all(delta <= np.asarray(half, dtype=np.float64), axis=1)
        chunks.append(world[~inside])
    if not chunks:
        return hashlib.sha256(b"").hexdigest(), np.empty((0, 3))
    stacked = np.round(np.vstack(chunks), 9)
    order = np.lexsort((stacked[:, 2], stacked[:, 1], stacked[:, 0]))
    stacked = np.ascontiguousarray(stacked[order])
    return hashlib.sha256(stacked.tobytes()).hexdigest(), stacked


def snapshot_difference(before: np.ndarray, after: np.ndarray) -> dict:
    """Explain how two outside-box coordinate sets differ."""
    def keys(array):
        if array.size == 0:
            return set()
        return set(map(tuple, array))

    before_keys, after_keys = keys(before), keys(after)
    only_before = before_keys - after_keys
    only_after = after_keys - before_keys
    return {
        "count_before": int(before.shape[0]),
        "count_after": int(after.shape[0]),
        "count_delta": int(after.shape[0] - before.shape[0]),
        "unique_before": len(before_keys),
        "unique_after": len(after_keys),
        "coords_only_in_before": len(only_before),
        "coords_only_in_after": len(only_after),
        "positions_identical": not only_before and not only_after,
        "sample_only_in_before": [list(c) for c in list(only_before)[:5]],
        "sample_only_in_after": [list(c) for c in list(only_after)[:5]],
    }


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def emission_material(name: str, colour):
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


def setup_world(strength: float = 0.0) -> None:
    world = bpy.data.worlds.get("StaffHoleWorld") or bpy.data.worlds.new("StaffHoleWorld")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = strength
    bpy.context.scene.world = world


def setup_lighting() -> None:
    """Key and fill suns so clay close-ups shade the hole wall and read as depth."""
    for name, rotation, energy in (
        ("StaffHoleKey", (math.radians(58.0), 0.0, math.radians(28.0)), 4.0),
        ("StaffHoleFill", (math.radians(72.0), 0.0, math.radians(203.0)), 1.6),
    ):
        if name in bpy.data.objects:
            continue
        lamp = bpy.data.lights.new(name, type="SUN")
        lamp.energy = energy
        obj = bpy.data.objects.new(name, lamp)
        obj.rotation_euler = rotation
        bpy.context.collection.objects.link(obj)


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


def aim_camera(camera, target: Vector, direction: Vector, distance: float, ortho_scale: float):
    camera.location = target - direction.normalized() * distance
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = ortho_scale
    bpy.context.view_layer.update()


def render(path: Path, material) -> None:
    scene = bpy.context.scene
    scene.view_layers[0].material_override = material
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    scene.view_layers[0].material_override = None


def load_rgba(path: Path) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        return np.flipud(pixels)
    finally:
        bpy.data.images.remove(image)


def alpha_mask(path: Path) -> np.ndarray:
    return load_rgba(path)[:, :, 3] > 0.5


def save_rgba(path: Path, rgba: np.ndarray) -> None:
    height, width = rgba.shape[:2]
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=True)
    try:
        image.pixels = np.flipud(rgba).astype(np.float32).ravel()
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def project_to_pixel(camera, point: Vector, resolution: int):
    coords = world_to_camera_view(bpy.context.scene, camera, point)
    return coords.x * resolution, (1.0 - coords.y) * resolution


# --------------------------------------------------------------------------------------
# ring detection
# --------------------------------------------------------------------------------------


def _run_lengths(mask: np.ndarray, axis: int, reverse: bool) -> np.ndarray:
    work = np.flip(mask, axis=axis) if reverse else mask
    counts = np.cumsum(work.astype(np.int32), axis=axis)
    reset = np.where(work, 0, counts)
    reset = np.maximum.accumulate(reset, axis=axis)
    result = counts - reset
    return np.flip(result, axis=axis) if reverse else result


def directional_radius(mask: np.ndarray) -> np.ndarray:
    """Vectorised minimum run length in the four axis directions."""
    return np.minimum.reduce(
        (
            _run_lengths(mask, 1, False),
            _run_lengths(mask, 1, True),
            _run_lengths(mask, 0, False),
            _run_lengths(mask, 0, True),
        )
    )


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
        world = vertex_world(obj)
        distance = np.square(world[:, 0] - x) + np.square(world[:, 2] - z)
        local_min = float(distance.min()) if len(distance) else float("inf")
        if local_min < winner_distance:
            winner_distance = local_min
            winner = obj
    if winner is None:
        raise RuntimeError("no mesh object could be associated with the staff ring")
    return winner, math.sqrt(winner_distance)


def disk_ratio(mask: np.ndarray, centre_px, inner: float, outer: float | None = None) -> float:
    col, row = centre_px
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    distance = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)
    region = distance <= inner if outer is None else ((distance >= inner) & (distance <= outer))
    return float(mask[region].mean()) if np.any(region) else 1.0


def blocked_inside_box(target, origin: Vector, direction: Vector, length: float,
                       centre, half, max_steps: int = 16) -> bool:
    """True when the ray strikes mesh inside the repair box (the ring itself).

    Hits outside the box are body or ornament geometry behind the ring and are skipped, so
    this measures whether the ring is open rather than whether anything is visible behind it.
    """
    matrix = target.matrix_world
    inverse = matrix.inverted()
    direction = direction.normalized()
    centre_v = Vector(centre)
    half_v = Vector(half)
    travelled = 0.0
    for _ in range(max_steps):
        remaining = length - travelled
        if remaining <= 1e-9:
            return False
        start_world = origin + direction * travelled
        start_local = inverse @ start_world
        direction_local = (inverse.to_3x3() @ direction).normalized()
        hit, location, _, _ = target.ray_cast(start_local, direction_local, distance=remaining)
        if not hit:
            return False
        hit_world = matrix @ location
        delta = hit_world - centre_v
        if all(abs(delta[i]) <= half_v[i] for i in range(3)):
            return True
        travelled = (hit_world - origin).dot(direction) + 1e-5
    return False


def open_ray_fraction(target, ring_centre: Vector, direction: Vector, radius: float,
                      span: float, centre, half, rings: int = 3, spokes: int = 12) -> float:
    """Fraction of a sampled ray bundle that passes clean through the ring opening."""
    direction = direction.normalized()
    helper = Vector((0.0, 0.0, 1.0))
    if abs(direction.dot(helper)) > 0.9:
        helper = Vector((1.0, 0.0, 0.0))
    u = direction.cross(helper).normalized()
    v = direction.cross(u).normalized()

    samples = [Vector(ring_centre)]
    for ring in range(1, rings + 1):
        r = radius * ring / float(rings)
        for spoke in range(spokes):
            angle = 2.0 * math.pi * spoke / spokes
            samples.append(ring_centre + u * (r * math.cos(angle)) + v * (r * math.sin(angle)))

    open_count = 0
    for point in samples:
        origin = point - direction * span
        if not blocked_inside_box(target, origin, direction, span * 2.0, centre, half):
            open_count += 1
    return open_count / float(len(samples))


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


# --------------------------------------------------------------------------------------
# localized patch surgery
# --------------------------------------------------------------------------------------


def select_patch_faces(target, centre_xyz, half_xyz, margin_rings: int) -> np.ndarray:
    """Faces touching the patch box, grown by a small adjacency margin."""
    world = vertex_world(target)
    delta = np.abs(world - np.asarray(centre_xyz, dtype=np.float64))
    vert_inside = np.all(delta <= np.asarray(half_xyz, dtype=np.float64), axis=1)

    count, loop_vert, poly_of_loop = polygon_topology(target)
    face_sel = np.bincount(poly_of_loop, weights=vert_inside[loop_vert], minlength=count) > 0
    for _ in range(margin_rings):
        vert_sel = np.zeros(world.shape[0], dtype=bool)
        vert_sel[loop_vert[face_sel[poly_of_loop]]] = True
        face_sel = np.bincount(poly_of_loop, weights=vert_sel[loop_vert], minlength=count) > 0
    return face_sel


def separate_patch(target, face_mask: np.ndarray):
    mesh = target.data
    loop_vert = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vert)
    _, _, poly_of_loop = polygon_topology(target)

    vert_sel = np.zeros(len(mesh.vertices), dtype=bool)
    vert_sel[loop_vert[face_mask[poly_of_loop]]] = True

    mesh.polygons.foreach_set("select", face_mask.astype(np.int8))
    mesh.vertices.foreach_set("select", vert_sel.astype(np.int8))
    mesh.edges.foreach_set("select", np.zeros(len(mesh.edges), dtype=np.int8))

    existing = {obj.name for obj in bpy.data.objects}
    select_only([target])
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")

    created = [obj for obj in bpy.data.objects if obj.name not in existing]
    if len(created) != 1:
        raise RuntimeError(f"patch separation produced {len(created)} objects, expected 1")
    return created[0]


def boundary_vertex_coords(obj) -> np.ndarray:
    """World coordinates of vertices on open boundary edges."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        indices = sorted({v.index for e in bm.edges if e.is_boundary for v in e.verts})
    finally:
        bm.free()
    if not indices:
        return np.empty((0, 3), dtype=np.float64)
    return vertex_world(obj)[np.asarray(indices, dtype=np.int64)]


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


def restore_corner_split(obj) -> None:
    """Split every edge so each face owns its own vertices, matching a triangle-soup source."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        bmesh.ops.split_edges(bm, edges=bm.edges[:])
        bm.to_mesh(obj.data)
    finally:
        bm.free()
    obj.data.update()


def join_and_weld(remainder, patch, boundary_world: np.ndarray, corner_split: bool) -> dict:
    select_only([remainder, patch])
    bpy.context.view_layer.objects.active = remainder
    bpy.ops.object.join()

    if corner_split:
        # Every triangle already owns its three vertices, so separation shares no vertex with
        # the remainder and there is no seam to stitch. Welding here would instead collapse
        # the mesh's existing corner splits and damage UV/normal attributes at the boundary.
        return {
            "skipped": True,
            "reason": "source mesh is corner-split (3 vertices per triangle); no shared seam",
            "welded_candidates": 0,
            "vertices_removed": 0,
        }

    if boundary_world.size == 0:
        return {"skipped": False, "welded_candidates": 0, "vertices_removed": 0}

    keys = {tuple(np.round(row, 9)) for row in boundary_world}
    world = vertex_world(remainder)
    rounded = np.round(world, 9)
    candidates = [i for i, row in enumerate(rounded) if tuple(row) in keys]

    before = len(remainder.data.vertices)
    bm = bmesh.new()
    try:
        bm.from_mesh(remainder.data)
        bm.verts.ensure_lookup_table()
        targets = [bm.verts[i] for i in candidates]
        if targets:
            bmesh.ops.remove_doubles(bm, verts=targets, dist=WELD_TOLERANCE)
        bm.to_mesh(remainder.data)
    finally:
        bm.free()
    remainder.data.update()
    return {
        "skipped": False,
        "welded_candidates": len(candidates),
        "vertices_removed": before - len(remainder.data.vertices),
    }


# --------------------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------------------


def annotate_detection(source: Path, destination: Path, resolution: int, ratio: float) -> None:
    rgba = load_rgba(source).copy()
    height, width = rgba.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - width * 0.5) ** 2 + (yy - height * 0.5) ** 2)
    outer_px = width / 5.5
    for radius, colour in ((outer_px, (1.0, 0.35, 0.1)), (outer_px * ratio, (0.1, 0.9, 1.0))):
        band = np.abs(distance - radius) <= max(1.5, width * 0.0022)
        rgba[band] = (*colour, 1.0)
    save_rgba(destination, rgba)


def render_view_set(camera, prefix: str, evidence: Path, target_point: Vector,
                    distance: float, ortho_scale: float, clay, silhouette,
                    backlit, resolution: int) -> dict:
    """Front, oblique and backlit renders around the staff ring."""
    views = {
        "front": Vector((0.0, 1.0, 0.0)),
        "oblique": Vector((0.62, 0.72, 0.31)),
    }
    produced = {}
    for name, direction in views.items():
        aim_camera(camera, target_point, direction, distance, ortho_scale)
        clay_path = evidence / f"{prefix}_staff_{name}.png"
        render(clay_path, clay)
        produced[f"{name}_clay"] = str(clay_path)

        mask_path = evidence / f"{prefix}_staff_{name}_mask.png"
        render(mask_path, silhouette)
        produced[f"{name}_mask"] = str(mask_path)

        centre_px = project_to_pixel(camera, target_point, resolution)
        produced[f"{name}_centre_px"] = [round(centre_px[0], 2), round(centre_px[1], 2)]

    # backlit silhouette: bright world behind, fully black subject
    setup_world(strength=6.0)
    bpy.context.scene.render.film_transparent = False
    aim_camera(camera, target_point, Vector((0.0, 1.0, 0.0)), distance, ortho_scale)
    backlit_path = evidence / f"{prefix}_staff_backlit.png"
    render(backlit_path, backlit)
    produced["backlit"] = str(backlit_path)
    bpy.context.scene.render.film_transparent = True
    setup_world(strength=CLAY_WORLD_STRENGTH)
    return produced


def backlit_centre_ratio(path: Path, centre_px, radius_px: float) -> float:
    """Fraction of the centre disk that is bright background (hole) rather than dark subject."""
    rgba = load_rgba(path)
    luminance = rgba[:, :, :3].mean(axis=2)
    col, row = centre_px
    yy, xx = np.ogrid[: luminance.shape[0], : luminance.shape[1]]
    distance = np.sqrt((xx - col) ** 2 + (yy - row) ** 2)
    region = distance <= radius_px
    if not np.any(region):
        return 0.0
    return float((luminance[region] > 0.5).mean())


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def build_context(args):
    """Import, orient, detect the ring and size the patch. Shared by both modes."""
    source = Path(args.input)
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    with Stage("import_mesh", IMPORT_BUDGET_SECONDS):
        reset_scene()
        objects = import_mesh(str(source))
        if not objects:
            raise RuntimeError("input GLB contains no mesh objects")
        original_matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
        triangles_before = triangle_count(objects)

    detect = Stage("preflight_detect", PREFLIGHT_BUDGET_SECONDS)
    detect.__enter__()
    orientation = orient(objects)

    setup_world(strength=CLAY_WORLD_STRENGTH)
    setup_lighting()
    camera, minimum, maximum, centre, ortho = setup_camera(objects, args.resolution)
    silhouette = emission_material("StaffHoleSilhouette", (1.0, 1.0, 1.0, 1.0))
    backlit = emission_material("StaffHoleBacklit", (0.0, 0.0, 0.0, 1.0))
    clay = clay_material()

    before_mask_path = evidence / "before_mask.png"
    render(before_mask_path, silhouette)
    before_mask = alpha_mask(before_mask_path)
    detection = detect_ring(before_mask)

    col, row = detection["centre_px"]
    x, z, world_per_pixel = pixel_to_world(col, row, centre, ortho, args.resolution)
    outer_world = detection["outer_radius_px"] * world_per_pixel
    hole_world = detection["hole_radius_px"] * world_per_pixel

    target, nearest_distance = nearest_object(objects, x, z)
    y_half = (maximum.y - minimum.y) * 0.5 + max(outer_world, 1e-4)
    y_centre = (maximum.y + minimum.y) * 0.5

    patch_centre = (x, y_centre, z)
    patch_half = (outer_world * 1.75, y_half, outer_world * 1.75)
    repair_centre = (x, y_centre, z)
    repair_half = (hole_world * 1.35, y_half, hole_world * 1.35)

    detection.update(
        {
            "centre_world_oriented": [x, y_centre, z],
            "outer_radius_world": outer_world,
            "hole_radius_world": hole_world,
            "world_per_pixel": world_per_pixel,
        }
    )

    # Ring wall thickness along the hole axis, measured in the annulus just outside the hole.
    # This sets how far a tilted ray may drift before it re-enters solid material.
    target_world = vertex_world(target)
    radial = np.sqrt(
        (target_world[:, 0] - x) ** 2 + (target_world[:, 2] - z) ** 2
    )
    band = (radial > hole_world * 1.05) & (radial < hole_world * 1.6)
    if band.any():
        band_y = target_world[band][:, 1]
        ring_thickness = float(band_y.max() - band_y.min())
        # The ring is not centred on the model's Y midpoint. Ray probes must start from the
        # ring's own centre, or a tilted bundle drifts sideways before it ever reaches the hole.
        ring_y_centre = float((band_y.max() + band_y.min()) * 0.5)
    else:
        ring_thickness = float("nan")
        ring_y_centre = y_centre

    mesh_data = target.data
    # Ratio, not equality: this mesh is 99.9% corner-split with a few thousand shared vertices.
    corner_split_ratio = len(mesh_data.vertices) / float(max(len(mesh_data.loops), 1))
    corner_split = corner_split_ratio >= 0.99

    face_mask = select_patch_faces(target, patch_centre, patch_half, args.margin_rings)
    patch_faces = int(face_mask.sum())
    total_faces = sum(len(obj.data.polygons) for obj in objects)
    fraction = patch_faces / float(total_faces) if total_faces else 1.0

    patch_world = vertex_world(target)
    _, loop_vert, poly_of_loop = polygon_topology(target)
    patch_verts = np.zeros(patch_world.shape[0], dtype=bool)
    patch_verts[loop_vert[face_mask[poly_of_loop]]] = True
    if patch_verts.any():
        selected = patch_world[patch_verts]
        patch_bbox = [selected.min(axis=0).tolist(), selected.max(axis=0).tolist()]
    else:
        patch_bbox = [[0, 0, 0], [0, 0, 0]]

    detect.__exit__()

    return {
        "source": source,
        "evidence": evidence,
        "objects": objects,
        "original_matrices": original_matrices,
        "triangles_before": triangles_before,
        "orientation": orientation,
        "camera": camera,
        "minimum": minimum,
        "maximum": maximum,
        "scene_centre": centre,
        "ortho": ortho,
        "silhouette": silhouette,
        "backlit": backlit,
        "clay": clay,
        "detection": detection,
        "target": target,
        "nearest_distance": nearest_distance,
        "x": x,
        "z": z,
        "y_centre": y_centre,
        "outer_world": outer_world,
        "hole_world": hole_world,
        "patch_centre": patch_centre,
        "patch_half": patch_half,
        "repair_centre": repair_centre,
        "repair_half": repair_half,
        "face_mask": face_mask,
        "patch_faces": patch_faces,
        "total_faces": total_faces,
        "patch_face_fraction": fraction,
        "patch_bbox_world_oriented": patch_bbox,
        "ring_thickness_world": ring_thickness,
        "ring_y_centre": ring_y_centre,
        "corner_split": corner_split,
        "corner_split_ratio": corner_split_ratio,
        "before_mask_path": before_mask_path,
        "before_mask": before_mask,
    }


def run_preflight(args) -> dict:
    ctx = build_context(args)

    evidence = ctx["evidence"]
    ring_point = Vector((ctx["x"], ctx["y_centre"], ctx["z"]))
    distance = max(ctx["maximum"].y - ctx["minimum"].y, ctx["ortho"]) * 2.5
    close_scale = ctx["outer_world"] * 5.5

    aim_camera(ctx["camera"], ring_point, Vector((0.0, 1.0, 0.0)), distance, close_scale)
    detected_path = evidence / "preflight_detected_region.png"
    render(detected_path, ctx["clay"])
    annotated = evidence / "preflight_detected_region_annotated.png"
    ratio = ctx["hole_world"] / max(ctx["outer_world"], 1e-9)
    annotate_detection(detected_path, annotated, args.resolution, ratio)

    report = {
        "mode": "preflight",
        "input": str(ctx["source"]),
        "input_sha256": sha256(ctx["source"]),
        "triangles_total": ctx["triangles_before"],
        "faces_total": ctx["total_faces"],
        "target_object": ctx["target"].name,
        "patch_faces": ctx["patch_faces"],
        "patch_face_fraction": ctx["patch_face_fraction"],
        "patch_face_fraction_limit": MAX_PATCH_FACE_FRACTION,
        "patch_within_limit": ctx["patch_face_fraction"] <= MAX_PATCH_FACE_FRACTION,
        "patch_bbox_world_oriented": ctx["patch_bbox_world_oriented"],
        "patch_box_centre": list(ctx["patch_centre"]),
        "patch_box_half_extent": list(ctx["patch_half"]),
        "repair_box_centre": list(ctx["repair_centre"]),
        "repair_box_half_extent": list(ctx["repair_half"]),
        "margin_rings": args.margin_rings,
        "detection": ctx["detection"],
        "nearest_target_vertex_distance_world": ctx["nearest_distance"],
        "orientation": ctx["orientation"],
        "evidence": {
            "before_mask": str(ctx["before_mask_path"]),
            "detected_region": str(detected_path),
            "detected_region_annotated": str(annotated),
        },
        "stage_seconds": dict(_STAGE_TIMINGS),
    }
    print(
        f"STAFF_PREFLIGHT patch_faces={ctx['patch_faces']} "
        f"fraction={ctx['patch_face_fraction']:.5f} total_faces={ctx['total_faces']} "
        f"triangles={ctx['triangles_before']}",
        flush=True,
    )
    return ctx, report


def run_localized_repair(args) -> dict:
    ctx, preflight = run_preflight(args)

    if not preflight["patch_within_limit"]:
        raise RuntimeError(
            "patch is too large for a localized repair: "
            f"{ctx['patch_faces']} faces = {ctx['patch_face_fraction']:.4%} of "
            f"{ctx['total_faces']} (limit {MAX_PATCH_FACE_FRACTION:.0%}). "
            "Refusing to fall back to a global Boolean."
        )

    source = ctx["source"]
    source_hash_before = sha256(source)
    objects = ctx["objects"]
    target = ctx["target"]
    evidence = ctx["evidence"]
    ring_point = Vector((ctx["x"], ctx["y_centre"], ctx["z"]))
    distance = max(ctx["maximum"].y - ctx["minimum"].y, ctx["ortho"]) * 2.5
    close_scale = ctx["outer_world"] * 5.5

    digest_before, snapshot_before = outside_box_snapshot(
        objects, ctx["repair_centre"], ctx["repair_half"]
    )

    before_views = render_view_set(
        ctx["camera"], "before", evidence, ring_point, distance, close_scale,
        ctx["clay"], ctx["silhouette"], ctx["backlit"], args.resolution,
    )
    full_before = evidence / "before_full_character.png"
    aim_camera(ctx["camera"], ctx["scene_centre"], Vector((0.0, 1.0, 0.0)),
               distance, ctx["ortho"])
    render(full_before, ctx["clay"])

    with Stage("localized_boolean", BOOLEAN_BUDGET_SECONDS):
        patch = separate_patch(target, ctx["face_mask"])
        patch_boundary = boundary_vertex_coords(patch)
        patch_triangles = triangle_count([patch])
        cutter_depth = max((ctx["maximum"].y - ctx["minimum"].y) * 3.0, ctx["hole_world"] * 8.0)
        cutter = add_cutter(ctx["x"], ctx["y_centre"], ctx["z"], ctx["hole_world"], cutter_depth)
        apply_boolean(patch, cutter)
        if ctx["corner_split"]:
            # Blender's EXACT solver welds its output. The source mesh is a triangle soup, so
            # restore that representation or the merge would silently destroy corner splits
            # (and with them UV/normal seams) across the whole patch.
            restore_corner_split(patch)
        patch_triangles_after = triangle_count([patch])
        weld = join_and_weld(target, patch, patch_boundary, ctx["corner_split"])

    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    triangles_after = triangle_count(objects)

    digest_after, snapshot_after = outside_box_snapshot(
        objects, ctx["repair_centre"], ctx["repair_half"]
    )
    outside_diff = snapshot_difference(snapshot_before, snapshot_after)
    # The source is a triangle soup, so duplicate multiplicity is bookkeeping, not geometry.
    # What must hold is that no position moved and no vertex was collapsed away.
    outside_unchanged = (
        outside_diff["positions_identical"] and outside_diff["count_delta"] >= 0
    )

    after_views = render_view_set(
        ctx["camera"], "after", evidence, ring_point, distance, close_scale,
        ctx["clay"], ctx["silhouette"], ctx["backlit"], args.resolution,
    )
    full_after = evidence / "after_full_character.png"
    aim_camera(ctx["camera"], ctx["scene_centre"], Vector((0.0, 1.0, 0.0)),
               distance, ctx["ortho"])
    render(full_after, ctx["clay"])

    # close-up front mask measurements, in close-up pixel space
    close_resolution = args.resolution
    hole_px_close = (ctx["hole_world"] / close_scale) * close_resolution
    centre_close = (close_resolution * 0.5, close_resolution * 0.5)

    front_mask = alpha_mask(Path(after_views["front_mask"]))
    centre_foreground_ratio = disk_ratio(front_mask, centre_close, hole_px_close * 0.62)
    annulus_foreground_ratio = disk_ratio(
        front_mask, centre_close, hole_px_close * 1.28, hole_px_close * 2.05
    )

    oblique_mask = alpha_mask(Path(after_views["oblique_mask"]))
    oblique_centre = after_views["oblique_centre_px"]
    # Reported for context only. A whole-model silhouette cannot gate openness at an oblique
    # angle because body and ornament geometry sits behind the foreshortened ring.
    oblique_centre_ratio = disk_ratio(oblique_mask, oblique_centre, hole_px_close * 0.45)

    backlit_ratio = backlit_centre_ratio(
        Path(after_views["backlit"]), centre_close, hole_px_close * 0.62
    )

    ray_blocked = ray_hits(target, ctx["x"], ctx["z"], ctx["minimum"], ctx["maximum"])

    probe_point = Vector((ctx["x"], ctx["ring_y_centre"], ctx["z"]))
    span = max(ctx["maximum"].y - ctx["minimum"].y, ctx["ortho"]) * 2.0
    axial_open = open_ray_fraction(
        target, probe_point, Vector((0.0, 1.0, 0.0)), ctx["hole_world"] * 0.62,
        span, ctx["repair_centre"], ctx["repair_half"],
    )
    # A ray tilted by theta drifts thickness*tan(theta) sideways while crossing the ring, so a
    # centre ray only clears while tan(theta) < hole_radius / thickness. Probe well inside that
    # measured limit; a closed recess still blocks every one of these rays.
    thickness = ctx["ring_thickness_world"]
    if thickness and thickness == thickness and thickness > 1e-9:
        limit_deg = math.degrees(math.atan2(ctx["hole_world"], thickness))
        tilt_deg = max(6.0, min(20.0, limit_deg * 0.6))
    else:
        limit_deg = float("nan")
        tilt_deg = 12.0
    # The ring plane need not be perpendicular to the Y cut axis, so scan azimuths around the
    # tilt cone instead of trusting a single arbitrary direction.
    tilt = math.radians(tilt_deg)
    oblique_scan = {}
    for step in range(8):
        azimuth = 2.0 * math.pi * step / 8.0
        direction = Vector(
            (
                math.sin(tilt) * math.cos(azimuth),
                math.cos(tilt),
                math.sin(tilt) * math.sin(azimuth),
            )
        )
        oblique_scan[round(math.degrees(azimuth))] = open_ray_fraction(
            target, probe_point, direction, ctx["hole_world"] * 0.25,
            span, ctx["repair_centre"], ctx["repair_half"],
        )
    oblique_open = max(oblique_scan.values())
    oblique_mean = sum(oblique_scan.values()) / len(oblique_scan)

    failures = []
    if centre_foreground_ratio > 0.08:
        failures.append(f"centre_foreground_ratio={centre_foreground_ratio:.4f} > 0.08")
    if annulus_foreground_ratio < 0.42:
        failures.append(f"annulus_foreground_ratio={annulus_foreground_ratio:.4f} < 0.42")
    if ray_blocked:
        failures.append("a direct ray through the ring centre still hits the mesh")
    if axial_open < 0.95:
        failures.append(f"axial_open_ray_fraction={axial_open:.4f} < 0.95")
    if oblique_open < 0.80:
        failures.append(
            f"oblique_open_ray_fraction={oblique_open:.4f} < 0.80 (recess, not a through-hole)"
        )
    if backlit_ratio < 0.60:
        failures.append(f"backlit_centre_ratio={backlit_ratio:.4f} < 0.60 (centre is not empty)")
    if not outside_diff["positions_identical"]:
        failures.append(
            "vertex positions outside the repair box changed: "
            f"{outside_diff['coords_only_in_before']} lost, "
            f"{outside_diff['coords_only_in_after']} added"
        )
    elif outside_diff["count_delta"] < 0:
        failures.append(
            f"{-outside_diff['count_delta']} vertices outside the repair box were collapsed; "
            "corner splits would be destroyed"
        )
    if failures:
        save_json(
            args.report,
            {
                "passed": False,
                "mode": "localized_repair",
                "failures": failures,
                "validation": {
                    "centre_foreground_ratio": centre_foreground_ratio,
                    "annulus_foreground_ratio": annulus_foreground_ratio,
                    "axial_open_ray_fraction": axial_open,
                    "oblique_open_ray_fraction": oblique_open, "oblique_open_ray_mean": oblique_mean, "oblique_azimuth_scan": oblique_scan,
                    "oblique_centre_ratio_context_only": oblique_centre_ratio,
                    "backlit_centre_ratio": backlit_ratio,
                    "ray_blocked_after_cut": ray_blocked,
                    "outside_repair_box_unchanged": outside_unchanged,
                },
                "outside_box_diff": outside_diff,
                "patch_faces": ctx["patch_faces"],
                "patch_face_fraction": ctx["patch_face_fraction"],
                "stage_seconds": dict(_STAGE_TIMINGS),
            },
        )
        raise RuntimeError("staff hole validation failed: " + "; ".join(failures))

    for obj in objects:
        if obj.name in ctx["original_matrices"]:
            obj.matrix_world = ctx["original_matrices"][obj.name]
    select_only(objects)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_glb(str(output), selected_only=True)

    source_hash_after = sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("input GLB changed during repair")

    report = {
        "passed": True,
        "mode": "localized_repair",
        "operation": "localized_staff_ring_through_hole",
        "input": str(source),
        "input_sha256": source_hash_after,
        "input_unchanged": True,
        "output": str(output),
        "output_sha256": sha256(output),
        "triangles_before": ctx["triangles_before"],
        "triangles_after": triangles_after,
        "triangle_delta": triangles_after - ctx["triangles_before"],
        "target_object": target.name,
        "patch": {
            "faces": ctx["patch_faces"],
            "face_fraction": ctx["patch_face_fraction"],
            "face_fraction_limit": MAX_PATCH_FACE_FRACTION,
            "triangles_before_cut": patch_triangles,
            "triangles_after_cut": patch_triangles_after,
            "boundary_vertices": int(patch_boundary.shape[0]),
            "bbox_world_oriented": ctx["patch_bbox_world_oriented"],
            "margin_rings": args.margin_rings,
            "weld": weld,
            "weld_tolerance": WELD_TOLERANCE,
        },
        "repair_box": {
            "centre": list(ctx["repair_centre"]),
            "half_extent": list(ctx["repair_half"]),
            "outside_digest_before": digest_before,
            "outside_digest_after": digest_after,
            "outside_unchanged": outside_unchanged,
            "outside_diff": outside_diff,
        },
        "detection": ctx["detection"],
        "validation": {
            "centre_foreground_ratio": centre_foreground_ratio,
            "annulus_foreground_ratio": annulus_foreground_ratio,
            "axial_open_ray_fraction": axial_open,
            "oblique_open_ray_fraction": oblique_open, "oblique_open_ray_mean": oblique_mean, "oblique_azimuth_scan": oblique_scan,
            "oblique_tilt_degrees": tilt_deg,
            "oblique_tilt_limit_degrees": limit_deg,
            "ring_thickness_world": thickness,
            "oblique_centre_ratio_context_only": oblique_centre_ratio,
            "backlit_centre_ratio": backlit_ratio,
            "ray_blocked_after_cut": ray_blocked,
            "outside_repair_box_unchanged": outside_unchanged,
            "real_through_hole_proven": True,
        },
        "evidence": dict(
            before_mask=str(ctx["before_mask_path"]),
            before_full_character=str(full_before),
            after_full_character=str(full_after),
            **{f"before_{k}": v for k, v in before_views.items()},
            **{f"after_{k}": v for k, v in after_views.items()},
        ),
        "orientation": ctx["orientation"],
        "preflight": preflight,
        "stage_seconds": dict(_STAGE_TIMINGS),
    }
    print(
        "STAFF_HOLE_REPAIR_PASS "
        f"target={target.name} triangles={ctx['triangles_before']}->{triangles_after} "
        f"patch_faces={ctx['patch_faces']} centre={centre_foreground_ratio:.4f} "
        f"annulus={annulus_foreground_ratio:.4f} oblique={oblique_centre_ratio:.4f} "
        f"backlit={backlit_ratio:.4f} outside_unchanged={outside_unchanged}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--margin-rings", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--localized-repair", action="store_true")
    args = parser.parse_args(argv_after_double_dash())

    started = time.monotonic()
    if args.preflight_only:
        _, report = run_preflight(args)
    else:
        if not args.output:
            parser.error("--localized-repair requires --output")
        report = run_localized_repair(args)

    report["total_seconds"] = round(time.monotonic() - started, 3)
    report["stage_seconds"] = dict(_STAGE_TIMINGS)
    save_json(args.report, report)
    print(f"STAGE_TIMINGS {json.dumps(report['stage_seconds'])}", flush=True)


if __name__ == "__main__":
    main()
