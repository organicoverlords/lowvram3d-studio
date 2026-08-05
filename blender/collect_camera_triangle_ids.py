"""Collect exact visible triangle IDs for one review camera."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash, import_mesh, reset_scene, world_bounds
from shaman_texture_review import place_camera, setup_world


def raster(screen, depth, triangles, size):
    px = screen * float(size - 1)
    z = np.full((size, size), np.inf)
    ids = np.full((size, size), -1, np.int32)
    for tid, tri in enumerate(triangles):
        a = px[tri]
        lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if np.any(hi < lo):
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(den) < 1e-12:
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((y1-y2)*(fx-x2) + (x2-x1)*(fy-y2)) / den
        w1 = ((y2-y0)*(fx-x2) + (x0-x2)*(fy-y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        d = w0*depth[tri[0]] + w1*depth[tri[1]] + w2*depth[tri[2]]
        closer = inside & (d < z[ys, xs])
        z[ys[closer], xs[closer]] = d[closer]
        ids[ys[closer], xs[closer]] = tid
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--resolution", type=int, default=512)
    args = p.parse_args(argv_after_double_dash())
    reset_scene()
    objects = import_mesh(args.glb)
    meshes = [o for o in objects if o.type == "MESH"]
    obj = max(meshes, key=lambda o: len(o.data.polygons))
    obj.data.calc_loop_triangles()
    triangles = np.asarray([lt.vertices for lt in obj.data.loop_triangles], np.int32)
    positions = np.asarray([(obj.matrix_world @ v.co)[:] for v in obj.data.vertices], np.float64)
    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    setup_world()
    camera_data = bpy.data.cameras.new("id_camera")
    camera = bpy.data.objects.new("id_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    place_camera(camera, centre, radius, args.yaw, args.pitch, 1.0, centre, 1.0)
    projected = np.asarray([world_to_camera_view(bpy.context.scene, camera, obj.matrix_world @ v.co)[:] for v in obj.data.vertices], np.float64)
    screen = projected[:, :2].copy(); screen[:, 1] = 1.0 - screen[:, 1]
    ids = raster(screen, projected[:, 2], triangles, args.resolution)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); np.save(out, ids)
    print(f"CAMERA_IDS yaw={args.yaw} visible={int((ids>=0).sum())} unique={int(np.unique(ids[ids>=0]).size)} output={out}", flush=True)


if __name__ == "__main__":
    main()
