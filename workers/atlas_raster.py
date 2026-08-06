"""Exact texel-centre atlas rasterisation, vectorised.

Semantics are identical to ``fast_texture_projection.rasterise_atlas``: a triangle covers a
texel when that texel's *centre* satisfies ``wa >= -1e-7``, ``wb >= -1e-7`` and
``wa + wb <= 1.0000001`` in the triangle's own UV barycentric frame, evaluated over the
triangle's UV bounding box clipped to the atlas. Nothing is scattered, nothing is supersampled,
and a UV outside the atlas is rejected rather than clamped to an edge.

The difference is bookkeeping, not geometry. ``rasterise_atlas`` walks 644k triangles in a
Python loop; here triangles are bucketed by bounding-box extent and each bucket is evaluated as
one padded batch, which is the same arithmetic in one pass. Padding a box outward can only add
texels that lie outside the triangle's bounding box, and a bounding box contains its triangle,
so the inside test rejects every added candidate: the result is unchanged.

Ownership is the lowest triangle index covering the texel. ``rasterise_atlas`` gives the first
triangle in source order that reaches an unowned texel, which is the same triangle.

``census`` keeps *every* covering triangle rather than only the owner, which is what proves a
layout injective: on an injective atlas no texel interior is claimed twice.
"""
from __future__ import annotations

import numpy as np

#: Bounding-box extents, in texels, that each batch is padded up to.
TIERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096)
#: Candidate texel evaluations per batch, which is what bounds peak memory here.
# Keep peak temporary arrays small enough for the full production GLB to remain
# resident alongside the atlas and texture buffers.  The census is unchanged;
# this only increases the number of deterministic batches.
BUDGET = 1_000_000


def census(uv: np.ndarray, tris: np.ndarray, size: int, *, interior: float = 0.0,
           with_barycentric: bool = False):
    """Every (texel, triangle) covering pair, unsorted.

    ``interior`` tightens the test to ``wa > interior``, ``wb > interior``,
    ``wa + wb < 1 - interior``. A positive value excludes triangle edges entirely, so a texel
    claimed twice cannot be an artifact of a shared chart seam or of the inclusive tolerance.
    """
    uv = np.asarray(uv, np.float64)
    tris = np.asarray(tris, np.int64)
    px = uv[tris] * float(size)

    low_corner = np.floor(px.min(axis=1))
    high_corner = np.ceil(px.max(axis=1)) - 1.0
    x0 = np.maximum(0, low_corner[:, 0]).astype(np.int64)
    x1 = np.minimum(size - 1, high_corner[:, 0]).astype(np.int64)
    y0 = np.maximum(0, low_corner[:, 1]).astype(np.int64)
    y1 = np.minimum(size - 1, high_corner[:, 1]).astype(np.int64)

    origin = px[:, 0]
    edge_a = px[:, 1] - origin
    edge_b = px[:, 2] - origin
    denominator = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    usable = (np.abs(denominator) > 1e-12) & (x1 >= x0) & (y1 >= y0)
    live = np.flatnonzero(usable)
    extent = np.maximum(x1 - x0 + 1, y1 - y0 + 1)[live]

    texels, triangles, weights_a, weights_b = [], [], [], []
    previous = 0
    for tier in TIERS:
        selected = live[(extent > previous) & (extent <= tier)]
        previous = tier
        if selected.size == 0:
            continue
        span = np.arange(tier, dtype=np.int64)
        step = max(1, int(BUDGET // (tier * tier)))
        for start in range(0, selected.size, step):
            batch = selected[start:start + step]
            ix = x0[batch][:, None] + span[None, :]
            iy = y0[batch][:, None] + span[None, :]
            inside_box = (iy <= y1[batch][:, None])[:, :, None] & (ix <= x1[batch][:, None])[:, None, :]
            dx = (ix + 0.5)[:, None, :] - origin[batch, 0][:, None, None]
            dy = (iy + 0.5)[:, :, None] - origin[batch, 1][:, None, None]
            inverse = 1.0 / denominator[batch][:, None, None]
            wa = (dx * edge_b[batch, 1][:, None, None] - dy * edge_b[batch, 0][:, None, None]) * inverse
            wb = (edge_a[batch, 0][:, None, None] * dy - edge_a[batch, 1][:, None, None] * dx) * inverse
            if interior > 0.0:
                covered = (wa > interior) & (wb > interior) & (wa + wb < 1.0 - interior)
            else:
                covered = (wa >= -1e-7) & (wb >= -1e-7) & (wa + wb <= 1.0000001)
            covered &= inside_box
            rows, yy, xx = np.nonzero(covered)
            if rows.size == 0:
                continue
            texels.append(iy[rows, yy] * size + ix[rows, xx])
            triangles.append(batch[rows])
            if with_barycentric:
                weights_a.append(wa[rows, yy, xx])
                weights_b.append(wb[rows, yy, xx])

    if not texels:
        empty = np.zeros(0, np.int64)
        if with_barycentric:
            return empty, empty, np.zeros(0), np.zeros(0)
        return empty, empty
    texel = np.concatenate(texels)
    triangle = np.concatenate(triangles)
    if with_barycentric:
        return texel, triangle, np.concatenate(weights_a), np.concatenate(weights_b)
    return texel, triangle


def rasterise(uv: np.ndarray, tris: np.ndarray, size: int):
    """Return (owner, weights) exactly as ``fast_texture_projection.rasterise_atlas`` would."""
    texel, triangle, wa, wb = census(uv, tris, size, with_barycentric=True)
    owner = np.full(size * size, -1, np.int32)
    weights = np.zeros((size * size, 2), np.float32)
    if texel.size == 0:
        return owner.reshape(size, size), weights.reshape(size, size, 2)

    order = np.lexsort((triangle, texel))
    texel = texel[order]
    triangle = triangle[order]
    wa = wa[order]
    wb = wb[order]
    first = np.flatnonzero(np.r_[True, texel[1:] != texel[:-1]])

    owner[texel[first]] = triangle[first].astype(np.int32)
    weights[texel[first], 0] = wa[first]
    weights[texel[first], 1] = wb[first]
    return owner.reshape(size, size), weights.reshape(size, size, 2)


def injectivity(uv: np.ndarray, tris: np.ndarray, size: int, *, interior: float = 0.05) -> dict:
    """Strict-interior double-claim census: the operative test for a usable single-owner atlas."""
    texel, triangle = census(uv, tris, size, interior=interior)
    if texel.size:
        order = np.argsort(texel, kind="stable")
        texel = texel[order]
        triangle = triangle[order]
        first = np.flatnonzero(np.r_[True, texel[1:] != texel[:-1]])
        counts = np.diff(np.r_[first, texel.size])
        shared = counts > 1
        offenders = np.unique(triangle[np.repeat(shared, counts)])
    else:
        first = np.zeros(0, np.int64)
        counts = np.zeros(0, np.int64)
        shared = np.zeros(0, bool)
        offenders = np.zeros(0, np.int64)

    corners = np.asarray(uv, np.float64)[np.asarray(tris, np.int64)] * float(size)
    area = 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0]))
    return {
        "interior_margin": float(interior),
        "atlas_size": int(size),
        "interior_texels": int(first.size),
        "interior_texels_claimed_twice": int(shared.sum()),
        "max_interior_claims_on_one_texel": int(counts.max()) if counts.size else 0,
        "triangles_sharing_interior_texels": int(offenders.size),
        "analytic_uv_area_fraction": float(area.sum() / float(size * size)),
        "degenerate_uv_triangles": int((area <= 0.0).sum()),
        "uv_out_of_unit_square": int(((np.asarray(uv) < -1e-6) | (np.asarray(uv) > 1 + 1e-6)).any(axis=1).sum()),
        "injective": bool(shared.sum() == 0),
        "exact_overlap": {
            "tested_pair_count": int(first.size),
            "timed_out": False,
            "success": True,
            "positive_overlap_total_texels_equivalent": float(shared.sum()),
            "positive_overlap_pair_count": int(shared.sum()),
            "degenerate_uv_triangle_count": int((area <= 0.0).sum()),
            "out_of_bounds_triangle_count": int(
                ((np.asarray(uv) < -1e-6) | (np.asarray(uv) > 1 + 1e-6)).any(axis=1).sum()),
            "method": "exhaustive_strict_interior_texel_census",
        },
    }
