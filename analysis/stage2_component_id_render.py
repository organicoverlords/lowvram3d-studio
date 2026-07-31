"""STAGE 2: six orthographic component-ID / depth / mask buffers.

Deterministic numpy z-buffer rasteriser using the pipeline's ORTHO convention, so the result is
reproducible and independent of any renderer setting. Six views total -- never one render per
component.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

CACHE = sys.argv[1]
SOURCE_FRONT = sys.argv[2]
OUTDIR = sys.argv[3]
OUT_JSON = sys.argv[4]
SIZE = 512
ORTHO_SCALE = 2.6

VIEWS = {
    "front": ((0.0, -3.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 3.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-3.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((3.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 3.0), (0.0, 1.0, 0.0)),
    "underside": ((0.0, 0.0, -3.0), (0.0, 1.0, 0.0)),
}

data = np.load(CACHE)
# trimesh preserves the glTF Y-up convention; every camera here (and the whole texture pipeline)
# is expressed in Blender's Z-up world, so convert once: blender = (x, -z, y). Without this the
# "front" camera actually looks down the model's up-axis and the source-support test is meaningless.
raw_vertices = data["vertices"].astype(np.float64)
vertices = np.column_stack((raw_vertices[:, 0], -raw_vertices[:, 2], raw_vertices[:, 1]))
triangles = data["triangles"]
labels = data["labels"]
main_id = int(data["main_id"])
component_count = int(labels.max()) + 1


def basis(eye, up):
    forward = -np.asarray(eye, np.float64)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up, np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return right, true_up, forward


def rasterise(eye, up):
    right, true_up, forward = basis(eye, up)
    relative = vertices - np.asarray(eye, np.float64)
    cam_x = relative @ right
    cam_y = relative @ true_up
    depth = relative @ forward
    half = ORTHO_SCALE * 0.5
    screen_x = (cam_x / half * 0.5 + 0.5) * (SIZE - 1)
    screen_y = (1.0 - (cam_y / half * 0.5 + 0.5)) * (SIZE - 1)

    zbuffer = np.full((SIZE, SIZE), np.inf, np.float64)
    idbuffer = np.full((SIZE, SIZE), -1, np.int32)

    tri_x = screen_x[triangles]
    tri_y = screen_y[triangles]
    tri_z = depth[triangles]
    lo_x = np.maximum(np.floor(tri_x.min(axis=1)).astype(int), 0)
    hi_x = np.minimum(np.ceil(tri_x.max(axis=1)).astype(int), SIZE - 1)
    lo_y = np.maximum(np.floor(tri_y.min(axis=1)).astype(int), 0)
    hi_y = np.minimum(np.ceil(tri_y.max(axis=1)).astype(int), SIZE - 1)
    valid = (hi_x >= lo_x) & (hi_y >= lo_y)

    for index in np.flatnonzero(valid):
        x0, x1, x2 = tri_x[index]
        y0, y1, y2 = tri_y[index]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue
        xs, ys = np.meshgrid(
            np.arange(lo_x[index], hi_x[index] + 1),
            np.arange(lo_y[index], hi_y[index] + 1),
        )
        xs = xs.ravel()
        ys = ys.ravel()
        px = xs + 0.5
        py = ys + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        xs, ys = xs[inside], ys[inside]
        z = w0[inside] * tri_z[index, 0] + w1[inside] * tri_z[index, 1] + w2[inside] * tri_z[index, 2]
        closer = z < zbuffer[ys, xs]
        if closer.any():
            yy, xx = ys[closer], xs[closer]
            zbuffer[yy, xx] = z[closer]
            idbuffer[yy, xx] = labels[index]
    return idbuffer, zbuffer


source = cv2.imread(SOURCE_FRONT, cv2.IMREAD_UNCHANGED)
if source is not None and source.shape[2] == 4:
    source_mask = cv2.resize(source[..., 3], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 40
else:
    source_mask = np.zeros((SIZE, SIZE), bool)

os.makedirs(OUTDIR, exist_ok=True)
palette = np.random.default_rng(12345).integers(60, 255, size=(component_count, 3), dtype=np.uint8)
palette[main_id] = (120, 120, 120)

metrics = {c: {"visible_pixels": {}, "outside_dilated": {}, "island_views": 0,
               "gap_pixels": {}, "overlap_main": {}} for c in range(component_count)}
per_view_masks = {}

for name, (eye, up) in VIEWS.items():
    idbuffer, zbuffer = rasterise(eye, up)
    complete_mask = idbuffer >= 0
    main_mask = idbuffer == main_id
    dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    per_view_masks[name] = {"complete": complete_mask, "main": main_mask, "id": idbuffer}

    colour = np.zeros((SIZE, SIZE, 3), np.uint8)
    colour[complete_mask] = palette[idbuffer[complete_mask]]
    cv2.imwrite(os.path.join(OUTDIR, f"componentid_{name}.png"), cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTDIR, f"maskcomplete_{name}.png"), (complete_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(OUTDIR, f"maskmain_{name}.png"), (main_mask * 255).astype(np.uint8))
    np.save(os.path.join(OUTDIR, f"idbuffer_{name}.npy"), idbuffer)
    finite = np.isfinite(zbuffer)
    depth_png = np.zeros((SIZE, SIZE), np.uint8)
    if finite.any():
        lo, hi = zbuffer[finite].min(), zbuffer[finite].max()
        depth_png[finite] = (255 * (1.0 - (zbuffer[finite] - lo) / max(hi - lo, 1e-9))).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTDIR, f"depth_{name}.png"), depth_png)

    main_distance = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)
    for component in range(component_count):
        if component == main_id:
            continue
        pixels = idbuffer == component
        count = int(pixels.sum())
        metrics[component]["visible_pixels"][name] = count
        if count == 0:
            continue
        metrics[component]["outside_dilated"][name] = round(
            float((pixels & ~dilated_main).sum() / count * 100), 3
        )
        metrics[component]["overlap_main"][name] = round(
            float((pixels & dilated_main).sum() / count * 100), 3
        )
        metrics[component]["gap_pixels"][name] = round(float(main_distance[pixels].min()), 3)
        separate = cv2.connectedComponents(
            (pixels | main_mask).astype(np.uint8), connectivity=8
        )[1]
        component_labels = set(np.unique(separate[pixels])) - {0}
        main_labels = set(np.unique(separate[main_mask])) - {0}
        if component_labels.isdisjoint(main_labels):
            metrics[component]["island_views"] += 1

front_id = per_view_masks["front"]["id"]
for component in range(component_count):
    if component == main_id:
        continue
    pixels = front_id == component
    total = int(pixels.sum())
    metrics[component]["front_visible_pixels"] = total
    metrics[component]["source_support_percent"] = (
        round(float((pixels & source_mask).sum() / total * 100), 3) if total else 0.0
    )
    metrics[component]["views_visible"] = sum(
        1 for v in metrics[component]["visible_pixels"].values() if v > 0
    )

report = {
    "size": SIZE,
    "ortho_scale": ORTHO_SCALE,
    "views": list(VIEWS),
    "main_component_id": main_id,
    "component_count": component_count,
    "source_front_mask_pixels": int(source_mask.sum()),
    "components": {str(k): v for k, v in metrics.items() if k != main_id},
}
with open(OUT_JSON, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(f"STAGE2_ID_RENDER views={len(VIEWS)} components={component_count} main={main_id}", flush=True)
