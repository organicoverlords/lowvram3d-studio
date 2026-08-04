"""Stage 6 repair: replace flat component-prior fill with spatially varying donor continuation.

raster_project degrades unseen surface in tiers: constrained donor, then component-local prior,
then global prior. On this subject the component-local prior is the failure. The shaman welds into
one body component of 219k triangles, so "component-local" median is a single colour for the entire
robe, shoulders, back and head - 117,370 triangles received it. That is the flat grey.

The fix keeps the same principle - colour may only come from observed surface on the same connected
component - but removes the hard radius cut-off that pushed triangles onto the median in the first
place. Instead the nearest observed donors are always used, weighted by distance and by normal
agreement, so the back of a shoulder inherits from the nearest observed shoulder rather than from
the average of the whole character. Distance still matters, it just degrades smoothly instead of
falling off a cliff.

Observed texels are never rewritten. The front stays exactly as projected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mesh_io import read_glb, triangle_components

WELD = 4e-4
DONOR_NEIGHBOURS = 16
MIN_NORMAL_DOT = 0.10


def rasterise_triangle_ids(uv: np.ndarray, tris: np.ndarray, size: int) -> np.ndarray:
    """Map every atlas texel to the triangle that covers it, in glTF orientation."""
    px = np.empty((len(uv), 2), np.float64)
    px[:, 0] = uv[:, 0] * (size - 1)
    px[:, 1] = uv[:, 1] * (size - 1)
    tri_id = np.full((size, size), -1, np.int32)
    for index, tri in enumerate(tris):
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
        if inside.any():
            tri_id[ys[inside], xs[inside]] = index
    return tri_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--neighbours", type=int, default=DONOR_NEIGHBOURS)
    parser.add_argument("--protected-mask", default="")
    args = parser.parse_args()

    positions, _, uv, tris = read_glb(Path(args.mesh))
    positions = positions.astype(np.float64)
    uv = uv.astype(np.float64)

    basecolor = cv2.cvtColor(cv2.imread(args.basecolor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    coverage = cv2.imread(args.coverage, cv2.IMREAD_GRAYSCALE)
    size = basecolor.shape[0]
    if coverage.shape[0] != size:
        raise RuntimeError(f"coverage is {coverage.shape[0]}px but base colour is {size}px")

    tri_id = rasterise_triangle_ids(uv, tris, size)
    island = tri_id >= 0
    observed_texel = island & (coverage >= 255)
    protected = np.zeros_like(observed_texel)
    if args.protected_mask:
        protected_image = cv2.imread(args.protected_mask, cv2.IMREAD_GRAYSCALE)
        if protected_image is None or protected_image.shape != observed_texel.shape:
            raise RuntimeError("PROTECTED_FACE_MASK_DIMENSION_MISMATCH")
        protected = protected_image > 0

    total = len(tris)
    colour_sum = np.zeros((total, 3), np.float64)
    colour_count = np.zeros(total, np.int64)
    np.add.at(colour_sum, tri_id[observed_texel], basecolor[observed_texel].astype(np.float64))
    np.add.at(colour_count, tri_id[observed_texel], 1)
    has_observation = colour_count > 0
    tri_colour = np.zeros((total, 3), np.float32)
    tri_colour[has_observation] = (colour_sum[has_observation] / colour_count[has_observation, None]).astype(np.float32)

    edge1 = positions[tris[:, 1]] - positions[tris[:, 0]]
    edge2 = positions[tris[:, 2]] - positions[tris[:, 0]]
    normals = np.cross(edge1, edge2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centroids = positions[tris].mean(axis=1)

    component, welded = triangle_components(positions, tris, WELD)

    tier = np.where(has_observation, "observed", "unresolved").astype(object)
    global_prior = np.median(tri_colour[has_observation], axis=0) if has_observation.any() else np.zeros(3, np.float32)

    for value in np.unique(component):
        in_component = component == value
        donors = np.flatnonzero(in_component & has_observation)
        targets = np.flatnonzero(in_component & ~has_observation)
        if targets.size == 0:
            continue
        if donors.size == 0:
            tri_colour[targets] = global_prior
            tier[targets] = "global_prior_no_observation_on_component"
            continue
        tree = cKDTree(centroids[donors])
        k = int(min(args.neighbours, donors.size))
        distance, nearest = tree.query(centroids[targets], k=k)
        if k == 1:
            distance, nearest = distance[:, None], nearest[:, None]
        donor_ids = donors[nearest]
        agreement = np.einsum("ijk,ik->ij", normals[donor_ids], normals[targets])
        # No radius cut-off. Distance still dominates through the 1/d weight, but a triangle whose
        # nearest observed neighbour is far away now inherits from that neighbour instead of
        # collapsing to the median of a 219k-triangle component.
        acceptable = agreement >= MIN_NORMAL_DOT
        weight = np.where(acceptable, np.clip(agreement, 0.0, None) / np.maximum(distance, 1e-6), 0.0)
        weight_total = weight.sum(axis=1)
        strict = weight_total > 0

        blended = np.zeros((targets.size, 3), np.float32)
        if strict.any():
            normalised = weight[strict] / weight_total[strict, None]
            blended[strict] = (tri_colour[donor_ids[strict]] * normalised[..., None]).sum(axis=1)
        # Facing away from every nearby donor: fall back to plain inverse distance on the same
        # component rather than to a constant, which keeps the variation even if the normal test
        # cannot be satisfied.
        if (~strict).any():
            relaxed = 1.0 / np.maximum(distance[~strict], 1e-6)
            relaxed /= relaxed.sum(axis=1, keepdims=True)
            blended[~strict] = (tri_colour[donor_ids[~strict]] * relaxed[..., None]).sum(axis=1)

        tri_colour[targets] = blended
        assigned = np.where(strict, "donor_normal_compatible", "donor_distance_only")
        tier[targets] = assigned

    repaint = island & ~observed_texel & ~protected
    result = basecolor.copy()
    result[repaint] = tri_colour[tri_id[repaint]]

    before_flat = float(basecolor[repaint].std(axis=0).mean())
    after_flat = float(result[repaint].std(axis=0).mean())
    cv2.imwrite(args.output, cv2.cvtColor((np.clip(result, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    tiers, counts = np.unique(tier, return_counts=True)
    report = {
        "mesh": args.mesh,
        "basecolor": args.basecolor,
        "output": args.output,
        "triangles": int(total),
        "observed_triangles": int(has_observation.sum()),
        "repainted_texels": int(repaint.sum()),
        "island_texels": int(island.sum()),
        "tier_triangle_counts": {str(k): int(v) for k, v in zip(tiers, counts)},
        "colour_spread_before": round(before_flat, 5),
        "colour_spread_after": round(after_flat, 5),
        "observed_unchanged": bool(np.array_equal(basecolor[observed_texel], result[observed_texel])),
        "protected_face_texels": int(protected.sum()),
        "protected_face_texels_unchanged": bool(np.array_equal(basecolor[protected], result[protected])),
        "protected_face_texel_sha256_before": hashlib.sha256(basecolor[protected].tobytes()).hexdigest(),
        "protected_face_texel_sha256_after": hashlib.sha256(result[protected].tobytes()).hexdigest(),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"REPAINT tiers={report['tier_triangle_counts']} spread {before_flat:.5f} -> {after_flat:.5f} "
        f"observed_unchanged={report['observed_unchanged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
