"""Fill unseen UV texels without smearing distant front imagery onto the rear.

Observed source texels remain byte-for-byte unchanged. Nearby same-component donors may continue
across a short geodesic-like distance. Surfaces beyond that radius receive deterministic procedural
material colour based on component size, position, facing and local geometry, then the existing
AO/cavity detail pass adds measured high-frequency wear.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from mesh_io import read_glb, triangle_components

WELD = 4e-4
MAX_DONOR_RADIUS_FRACTION = 0.055


def rasterise_triangle_ids(uv: np.ndarray, tris: np.ndarray, size: int) -> np.ndarray:
    px = np.empty((len(uv), 2), np.float64)
    px[:, 0] = uv[:, 0] * (size - 1)
    px[:, 1] = uv[:, 1] * (size - 1)
    tri_id = np.full((size, size), -1, np.int32)
    for index, tri in enumerate(tris):
        a = px[tri]
        x0, y0 = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs, ys = xs.ravel(), ys.ravel()
        (ax, ay), (bx, by), (cx, cy) = a
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        fx, fy = xs + 0.5, ys + 0.5
        wa = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) / den
        wb = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) / den
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-4) & (wb >= -1e-4) & (wc >= -1e-4)
        tri_id[ys[inside], xs[inside]] = index
    return tri_id


def procedural_colour(centroid: np.ndarray, normal: np.ndarray, component_id: int,
                       component_size: int, lo: np.ndarray, extent: np.ndarray) -> np.ndarray:
    p = (centroid - lo) / np.maximum(extent, 1e-9)
    x, y, z = p
    outboard = abs(x - 0.5) > 0.39
    small = component_size < 1800
    high = z > 0.62
    central_head = abs(x - 0.5) < 0.18 and z > 0.55

    if outboard and not small:
        base = np.array([0.25, 0.18, 0.105], np.float32)  # weathered staff wood
    elif central_head or (high and not small):
        base = np.array([0.58, 0.53, 0.43], np.float32)   # bone / keratin
    elif small:
        palette = np.array([
            [0.46, 0.39, 0.28], [0.31, 0.39, 0.37], [0.53, 0.44, 0.31],
            [0.38, 0.29, 0.21], [0.57, 0.54, 0.46], [0.28, 0.24, 0.18],
        ], np.float32)
        base = palette[int(component_id) % len(palette)]
    else:
        upper = np.array([0.32, 0.34, 0.29], np.float32)
        lower = np.array([0.20, 0.145, 0.095], np.float32)
        base = lower * (1.0 - z) + upper * z              # layered weathered cloth

    facing = 0.5 + 0.5 * abs(float(normal[1]))
    vertical_wear = 0.90 + 0.12 * z
    deterministic = 0.96 + 0.06 * np.sin(component_id * 1.731 + x * 9.0 + z * 13.0)
    return np.clip(base * facing * vertical_wear * deterministic, 0.035, 0.82)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--neighbours", type=int, default=12)
    args = parser.parse_args()

    positions, _, uv, tris = read_glb(Path(args.mesh))
    positions = positions.astype(np.float64)
    uv = uv.astype(np.float64)
    image_bgr = cv2.imread(args.basecolor, cv2.IMREAD_COLOR)
    coverage = cv2.imread(args.coverage, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or coverage is None:
        raise RuntimeError("base colour or coverage image is missing")
    base = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    size = base.shape[0]
    if coverage.shape != base.shape[:2]:
        raise RuntimeError("coverage and base colour dimensions differ")

    tri_id = rasterise_triangle_ids(uv, tris, size)
    island = tri_id >= 0
    observed_texel = island & (coverage >= 255)
    total = len(tris)

    sums = np.zeros((total, 3), np.float64)
    counts = np.zeros(total, np.int64)
    np.add.at(sums, tri_id[observed_texel], base[observed_texel])
    np.add.at(counts, tri_id[observed_texel], 1)
    observed_tri = counts > 0
    colours = np.zeros((total, 3), np.float32)
    colours[observed_tri] = (sums[observed_tri] / counts[observed_tri, None]).astype(np.float32)

    edge1 = positions[tris[:, 1]] - positions[tris[:, 0]]
    edge2 = positions[tris[:, 2]] - positions[tris[:, 0]]
    normals = np.cross(edge1, edge2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centroids = positions[tris].mean(axis=1)
    component, _ = triangle_components(positions, tris, WELD)
    comp_sizes = np.bincount(component, minlength=int(component.max()) + 1)
    lo, hi = positions.min(0), positions.max(0)
    extent = hi - lo
    radius = float(np.linalg.norm(extent)) * MAX_DONOR_RADIUS_FRACTION

    tier = np.full(total, "observed", dtype=object)
    for cid in np.unique(component):
        members = np.flatnonzero(component == cid)
        donors = members[observed_tri[members]]
        targets = members[~observed_tri[members]]
        if targets.size == 0:
            continue
        nearby_mask = np.zeros(targets.size, bool)
        if donors.size:
            tree = cKDTree(centroids[donors])
            k = min(max(1, args.neighbours), donors.size)
            dist, nearest = tree.query(centroids[targets], k=k)
            if k == 1:
                dist, nearest = dist[:, None], nearest[:, None]
            donor_ids = donors[nearest]
            nearest_dist = dist[:, 0]
            nearby_mask = nearest_dist <= radius
            if nearby_mask.any():
                agreement = np.clip(np.einsum("ijk,ik->ij", normals[donor_ids[nearby_mask]],
                                              normals[targets[nearby_mask]]), 0.0, 1.0)
                weights = (0.15 + agreement) / np.maximum(dist[nearby_mask], 1e-5)
                weights /= weights.sum(axis=1, keepdims=True)
                colours[targets[nearby_mask]] = (
                    colours[donor_ids[nearby_mask]] * weights[..., None]
                ).sum(axis=1)
                tier[targets[nearby_mask]] = "nearby_same_component_donor"

        far_targets = targets[~nearby_mask]
        for tid in far_targets:
            colours[tid] = procedural_colour(
                centroids[tid], normals[tid], int(cid), int(comp_sizes[int(cid)]), lo, extent
            )
        tier[far_targets] = "procedural_unseen_material"

    repaint = island & ~observed_texel
    result = base.copy()
    result[repaint] = colours[tri_id[repaint]]
    cv2.imwrite(args.output, cv2.cvtColor((np.clip(result, 0, 1) * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

    labels, label_counts = np.unique(tier, return_counts=True)
    report = {
        "mesh": args.mesh,
        "output": args.output,
        "observed_triangles": int(observed_tri.sum()),
        "unseen_triangles": int((~observed_tri).sum()),
        "max_donor_radius_fraction": MAX_DONOR_RADIUS_FRACTION,
        "max_donor_radius": radius,
        "tier_triangle_counts": {str(k): int(v) for k, v in zip(labels, label_counts)},
        "observed_unchanged": bool(np.array_equal(base[observed_texel], result[observed_texel])),
        "remote_front_smearing_disabled": True,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"UNSEEN_FILL tiers={report['tier_triangle_counts']} observed_unchanged={report['observed_unchanged']}")


if __name__ == "__main__":
    main()
