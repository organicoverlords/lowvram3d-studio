"""STAGE 5: three bounded xatlas chart presets to cure UV fragmentation."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import trimesh
import xatlas

MESH_PATH = sys.argv[1]
OUTDIR = sys.argv[2]
RESOLUTION = 2048
PADDING = 8

PRESETS = {
    "A": {"max_cost": 2.0, "max_iterations": 2},
    "B": {"max_cost": 4.0, "max_iterations": 2},
    "C": {"max_cost": 8.0, "max_iterations": 2},
}

os.makedirs(OUTDIR, exist_ok=True)
loaded = trimesh.load(MESH_PATH, process=False)
if isinstance(loaded, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()])
else:
    mesh = loaded
vertices = np.asarray(mesh.vertices, np.float32)
faces = np.asarray(mesh.faces, np.uint32)
normals = np.asarray(mesh.vertex_normals, np.float32)

results = []
for name, options in PRESETS.items():
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces, normals)

    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = options["max_cost"]
    chart_options.max_iterations = options["max_iterations"]

    pack_options = xatlas.PackOptions()
    pack_options.resolution = RESOLUTION
    pack_options.padding = PADDING
    pack_options.bilinear = True
    pack_options.blockAlign = True
    pack_options.bruteForce = False
    pack_options.rotate_charts = True
    pack_options.rotate_charts_to_axis = True
    pack_options.create_image = True

    start = time.time()
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    elapsed = time.time() - start

    vmapping, indices, uvs = atlas[0]
    chart_count = int(atlas.chart_count)
    raw_utilization = atlas.utilization
    utilization = float(raw_utilization[0] if hasattr(raw_utilization, "__getitem__") else raw_utilization)

    # UV validity
    invalid = int((~np.isfinite(uvs)).any(axis=1).sum() + ((uvs < -1e-6) | (uvs > 1 + 1e-6)).any(axis=1).sum())

    # Island occupancy and tiny-island count, measured on a rasterised coverage mask.
    pixel_uv = uvs.copy()
    pixel_uv[:, 0] *= RESOLUTION - 1
    pixel_uv[:, 1] = (1.0 - pixel_uv[:, 1]) * (RESOLUTION - 1)
    import cv2

    occupancy = np.zeros((RESOLUTION, RESOLUTION), np.uint8)
    tri_uv = pixel_uv[indices]
    cv2.fillPoly(occupancy, [t.astype(np.int32) for t in tri_uv], 255)
    occupied = float((occupancy > 0).sum() / (RESOLUTION * RESOLUTION) * 100)
    island_count, island_labels, stats, _ = cv2.connectedComponentsWithStats(
        (occupancy > 0).astype(np.uint8), connectivity=8
    )
    sizes = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    tiny = int(np.sum((sizes[0] < 4) | (sizes[1] < 4)))
    min_dimension = int(min(sizes[0].min(), sizes[1].min())) if island_count > 1 else 0

    # Stretch: UV area vs 3D area ratio per triangle.
    p3 = vertices[faces.reshape(-1)].reshape(-1, 3, 3)
    area3d = np.linalg.norm(np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0]), axis=1) * 0.5
    uv3 = uvs[indices]
    area_uv = np.abs(
        (uv3[:, 1, 0] - uv3[:, 0, 0]) * (uv3[:, 2, 1] - uv3[:, 0, 1])
        - (uv3[:, 2, 0] - uv3[:, 0, 0]) * (uv3[:, 1, 1] - uv3[:, 0, 1])
    ) * 0.5
    scale = np.sqrt(np.maximum(area_uv, 1e-16) / np.maximum(area3d[: len(area_uv)], 1e-16))
    scale = scale[np.isfinite(scale) & (scale > 0)]
    normalised = scale / np.median(scale)
    stretch_p95 = float(np.percentile(normalised, 95))
    stretch_max = float(normalised.max())

    # Overlap: count atlas pixels covered by more than one triangle. Accumulated inside each
    # triangle's own bounding box -- allocating a full-resolution layer per triangle would be
    # ~41k x 4MB of pointless work.
    counter = np.zeros((RESOLUTION, RESOLUTION), np.uint16)
    for triangle in tri_uv:
        x_lo = max(int(np.floor(triangle[:, 0].min())), 0)
        x_hi = min(int(np.ceil(triangle[:, 0].max())), RESOLUTION - 1)
        y_lo = max(int(np.floor(triangle[:, 1].min())), 0)
        y_hi = min(int(np.ceil(triangle[:, 1].max())), RESOLUTION - 1)
        if x_hi < x_lo or y_hi < y_lo:
            continue
        local = np.zeros((y_hi - y_lo + 1, x_hi - x_lo + 1), np.uint8)
        shifted = triangle - np.array([x_lo, y_lo], np.float32)
        cv2.fillPoly(local, [shifted.astype(np.int32)], 1)
        counter[y_lo:y_hi + 1, x_lo:x_hi + 1] += local
    overlap_percent = float((counter > 1).sum() / max((counter > 0).sum(), 1) * 100)

    record = {
        "preset": name,
        "max_cost": options["max_cost"],
        "max_iterations": options["max_iterations"],
        "chart_count": chart_count,
        "atlas_utilization": round(utilization * 100, 3),
        "uv_island_occupancy_percent": round(occupied, 3),
        "overlap_percent": round(overlap_percent, 4),
        "invalid_uv_count": invalid,
        "stretch_p95": round(stretch_p95, 4),
        "stretch_max": round(stretch_max, 4),
        "screen_islands": int(island_count - 1),
        "islands_below_4px": tiny,
        "min_island_pixel_dimension": min_dimension,
        "unwrap_seconds": round(elapsed, 2),
        "vertices_out": int(len(vmapping)),
        "faces_out": int(len(indices)),
    }
    results.append(record)
    np.savez_compressed(
        os.path.join(OUTDIR, f"uv_preset_{name}.npz"),
        vmapping=vmapping, indices=indices, uvs=uvs,
    )
    print(f"PRESET {name}: charts={chart_count} util={utilization*100:.1f}% "
          f"occupancy={occupied:.1f}% overlap={overlap_percent:.3f}% invalid={invalid} "
          f"tiny={tiny} p95stretch={stretch_p95:.2f} {elapsed:.1f}s", flush=True)

with open(os.path.join(OUTDIR, "uv_rebuild_report.json"), "w", encoding="utf-8") as handle:
    json.dump({"resolution": RESOLUTION, "padding": PADDING, "presets": results}, handle, indent=2)
