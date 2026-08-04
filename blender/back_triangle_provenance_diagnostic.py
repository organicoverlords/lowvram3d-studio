"""Exact review-camera rear triangle IDs and close-up diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds
from shaman_texture_review import configure_unlit, place_camera, setup_world


def image_from_array(name: str, array: np.ndarray, path: Path) -> None:
    rgba = np.asarray(array, dtype=np.float32)
    image = bpy.data.images.new(name, width=rgba.shape[1], height=rgba.shape[0], alpha=True)
    # Blender image pixels are stored bottom-up; PNG viewers are top-down.
    image.pixels = np.flipud(rgba).reshape(-1).tolist()
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def hsv_to_rgb(h: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0) * 6.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    p = np.full(h.shape, 0.75, np.float32)
    q = 1.0 - f * 0.25
    t = 0.75 + f * 0.25
    out = np.zeros((*h.shape, 3), np.float32)
    for k in range(6):
        mask = i == k
        if k == 0: out[mask] = np.stack([t[mask], np.ones(mask.sum()), p[mask]], axis=1)
        elif k == 1: out[mask] = np.stack([q[mask], np.ones(mask.sum()), p[mask]], axis=1)
        elif k == 2: out[mask] = np.stack([p[mask], t[mask], np.ones(mask.sum())], axis=1)
        elif k == 3: out[mask] = np.stack([p[mask], q[mask], np.ones(mask.sum())], axis=1)
        elif k == 4: out[mask] = np.stack([t[mask], p[mask], np.ones(mask.sum())], axis=1)
        else: out[mask] = np.stack([np.ones(mask.sum()), p[mask], q[mask]], axis=1)
    return out


def _rasterise(screen, depth, vertices, normals, tris, size):
    px = screen * float(size - 1)
    zbuffer = np.full((size, size), np.inf, dtype=np.float64)
    face_id = np.full((size, size), -1, dtype=np.int32)
    bary = np.zeros((size, size, 3), dtype=np.float32)
    position = np.zeros((size, size, 3), dtype=np.float32)
    normal = np.zeros((size, size, 3), dtype=np.float32)
    for triangle_id, tri in enumerate(tris):
        a = px[tri]
        x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x_hi < x_lo or y_hi < y_lo:
            continue
        xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        xs, ys = xs.ravel(), ys.ravel()
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / den
        w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        xs, ys = xs[inside], ys[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        d = weights @ depth[tri]
        closer = d < zbuffer[ys, xs]
        if not closer.any():
            continue
        xs, ys, weights, d = xs[closer], ys[closer], weights[closer], d[closer]
        zbuffer[ys, xs] = d
        face_id[ys, xs] = int(triangle_id)
        bary[ys, xs] = weights.astype(np.float32)
        position[ys, xs] = weights @ vertices[tri]
        normal[ys, xs] = weights @ normals[tri]
    silhouette = face_id >= 0
    normal[silhouette] /= np.maximum(np.linalg.norm(normal[silhouette], axis=1, keepdims=True), 1e-12)
    return face_id, bary, position, normal, zbuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.glb)
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"EXPECTED_ONE_MESH_OBJECT:{len(meshes)}")
    obj = meshes[0]
    obj.data.calc_loop_triangles()
    triangles = np.asarray([lt.vertices for lt in obj.data.loop_triangles], dtype=np.int32)
    positions = np.asarray([(obj.matrix_world @ v.co)[:] for v in obj.data.vertices], dtype=np.float64)
    normals = np.asarray([(obj.matrix_world.to_3x3() @ v.normal).normalized()[:] for v in obj.data.vertices], dtype=np.float64)

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    setup_world()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    camera_data = bpy.data.cameras.new("rear_exact_review")
    camera = bpy.data.objects.new("rear_exact_review", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    place_camera(camera, centre, radius, 180.0, 0.0, 1.0, centre, 1.0)

    projected = np.asarray([world_to_camera_view(scene, camera, obj.matrix_world @ v.co)[:] for v in obj.data.vertices], dtype=np.float64)
    screen = projected[:, :2].copy()
    screen[:, 1] = 1.0 - screen[:, 1]
    depth = projected[:, 2]
    face_id, bary, _position, _normal, zbuffer = _rasterise(
        screen, depth, positions, normals, triangles, args.resolution
    )
    visible = face_id >= 0

    # Review-camera selection: upper central geometry attached to the head/hood. The
    # central screen window removes the staff and hanging ornaments; visible height keeps
    # the selection on the upper back instead of the robe and feet.
    yy, xx = np.indices(face_id.shape)
    head_window = visible & (xx >= int(args.resolution * 0.25)) & (xx <= int(args.resolution * 0.75)) & (yy <= int(args.resolution * 0.58))
    selected = np.unique(face_id[head_window]).astype(np.int32)
    selected = selected[selected >= 0]
    head_mask = head_window & np.isin(face_id, selected)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "back_visible_triangle_ids.npy", face_id)
    np.save(out / "back_head_hood_triangle_ids.npy", selected)

    # Stable ID colours, with selected head/hood triangles highlighted yellow.
    overlay = np.zeros((args.resolution, args.resolution, 4), np.float32)
    if visible.any():
        ids = face_id[visible].astype(np.float32)
        colours = hsv_to_rgb(ids * 0.61803398875)
        overlay[visible, :3] = colours
        overlay[visible, 3] = 1.0
    overlay[head_mask, :3] = (1.0, 0.85, 0.05)
    overlay[head_mask, 3] = 1.0
    image_from_array("back_triangle_id_overlay", overlay, out / "back_texture_provenance_overlay.png")

    configure_unlit(objects)
    head = Vector((centre.x, centre.y, minimum.z + (maximum.z - minimum.z) * 0.86))
    place_camera(camera, centre, radius, 180.0, 0.0, 0.43, head, 1.0)
    close_path = out / "back_unlit_closeup.png"
    scene.render.filepath = str(close_path)
    bpy.ops.render.render(write_still=True)

    report = {
        "schema": "back_triangle_provenance_diagnostic_v1",
        "glb": str(args.glb),
        "camera": {"yaw_degrees": 180.0, "pitch_degrees": 0.0, "front_direction_gltf": "-z", "resolution": args.resolution},
        "triangle_count": int(len(triangles)),
        "visible_pixel_count": int(visible.sum()),
        "head_hood_selection": {
            "selection_rule": "review-camera visibility + upper central screen window; staff and hanging ornaments excluded by window",
            "triangle_count": int(len(selected)),
            "triangle_ids": selected.tolist(),
            "pixel_count": int(head_mask.sum()),
        },
        "outputs": {
            "back_visible_triangle_ids": str(out / "back_visible_triangle_ids.npy"),
            "back_head_hood_triangle_ids": str(out / "back_head_hood_triangle_ids.npy"),
            "back_texture_provenance_overlay": str(out / "back_texture_provenance_overlay.png"),
            "back_unlit_closeup": str(close_path),
        },
    }
    save_json(out / "back_triangle_provenance_report.json", report)
    print(f"BACK_ID visible={visible.sum()} head_hood_triangles={len(selected)} pixels={head_mask.sum()}", flush=True)


if __name__ == "__main__":
    main()
