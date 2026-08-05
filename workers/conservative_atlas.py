"""Diagnostic conservative UV-atlas coverage.

The production atlas owner stays point-sampled at texel centres.  This module is
separate on purpose: it measures pixel cells with positive-area triangle
intersection so sub-texel UV triangles can be classified as atlas-owned gaps
without mislabelling those cells as direct observations.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

DEFAULT_TIERS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


@dataclass(frozen=True)
class ConservativeCoverage:
    """Aggregated positive-area pixel-cell coverage."""

    claim_count: np.ndarray
    first_triangle: np.ndarray
    claims_per_triangle: np.ndarray
    tested_triangles: int
    positive_area_triangles: int
    clipped_out_triangles: int


@dataclass(frozen=True)
class ConservativeSupport:
    """A provenance-safe support layer kept separate from centre ownership."""

    owner: np.ndarray
    barycentric: np.ndarray
    chart_id: np.ndarray
    collision: np.ndarray
    same_chart_ambiguous: np.ndarray


def derive_uv_chart_ids(
    uv: np.ndarray,
    tris: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Derive deterministic UV charts from complete shared UV edges.

    The canonical mesh retains one UV entry per indexed corner, so an edge is
    connected only when the same two indexed UV corners are shared.  This is
    deliberately stricter than 3-D proximity and therefore keeps UV seams in
    separate charts.
    """
    uv_array = np.asarray(uv, dtype=np.float64)
    tri_array = np.asarray(tris, dtype=np.int64)
    if uv_array.ndim != 2 or uv_array.shape[1] != 2:
        raise ValueError("uv must have shape [vertex_count, 2]")
    if tri_array.ndim != 2 or tri_array.shape[1] != 3:
        raise ValueError("tris must have shape [triangle_count, 3]")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    count = int(tri_array.shape[0])
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    edges: dict[tuple[int, int], list[int]] = {}
    for triangle_id, corners in enumerate(tri_array):
        for corner in range(3):
            a = int(corners[corner])
            b = int(corners[(corner + 1) % 3])
            if a == b:
                continue
            # Index identity is the authoritative seam discriminator.  A
            # diagonal UV edge may be longer than one atlas unit, so do not
            # impose an arbitrary length bound here.
            key = (min(a, b), max(a, b))
            if not np.all(np.isfinite(uv_array[[a, b]])):
                continue
            edges.setdefault(key, []).append(triangle_id)
    for members in edges.values():
        first = members[0]
        for other in members[1:]:
            union(first, other)

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    unique_roots = sorted(set(int(value) for value in roots.tolist()))
    root_to_chart = {root: chart for chart, root in enumerate(unique_roots)}
    chart_ids = np.asarray([root_to_chart[int(root)] for root in roots], dtype=np.int32)
    triangles_per_chart = np.bincount(chart_ids, minlength=len(unique_roots))
    report = {
        "schema": "uv_chart_inventory_v1",
        "chart_count": int(len(unique_roots)),
        "triangle_count": int(count),
        "triangles_per_chart": triangles_per_chart.astype(int).tolist(),
        "shared_uv_edge_count": int(sum(1 for members in edges.values() if len(members) > 1)),
        "cross_chart_connectivity": 0,
        "tolerance": float(tolerance),
        "identity_rule": "same primitive indexed UV edge; no 3D proximity",
    }
    return chart_ids, report


def closest_point_on_uv_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return closest point, barycentrics and squared distance in 2-D."""
    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    a, b, c = tri
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.array([1.0, 0.0, 0.0]), float(np.dot(p - a, p - a))
    bp = p - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.array([0.0, 1.0, 0.0]), float(np.dot(p - b, p - b))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        t = d1 / denom if abs(denom) > 1e-15 else 0.0
        q = a + t * ab
        return q, np.array([1.0 - t, t, 0.0]), float(np.dot(p - q, p - q))
    cp = p - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.array([0.0, 0.0, 1.0]), float(np.dot(p - c, p - c))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        t = d2 / denom if abs(denom) > 1e-15 else 0.0
        q = a + t * ac
        return q, np.array([1.0 - t, 0.0, t]), float(np.dot(p - q, p - q))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        denom = float(np.dot(edge, edge))
        t = float(np.dot(p - b, edge) / denom) if denom > 1e-15 else 0.0
        t = min(1.0, max(0.0, t))
        q = b + t * edge
        return q, np.array([0.0, 1.0 - t, t]), float(np.dot(p - q, p - q))
    denom = float(np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac) ** 2)
    if abs(denom) <= 1e-15:
        return a.copy(), np.array([1.0, 0.0, 0.0]), float(np.dot(p - a, p - a))
    v = (np.dot(ac, ac) * np.dot(ab, ap) - np.dot(ab, ac) * np.dot(ac, ap)) / denom
    w = (np.dot(ab, ab) * np.dot(ac, ap) - np.dot(ab, ac) * np.dot(ab, ap)) / denom
    bary = np.array([1.0 - v - w, v, w])
    q = bary @ tri
    return q, bary, float(np.dot(p - q, p - q))


def resolve_conservative_support(
    uv: np.ndarray,
    tris: np.ndarray,
    size: int,
    direct_owner: np.ndarray,
    chart_ids: np.ndarray,
    *,
    triangle_ids: np.ndarray | None = None,
    budget: int = 2_000_000,
) -> ConservativeSupport:
    """Resolve positive-area cells without changing direct centre ownership."""
    owner = np.asarray(direct_owner, dtype=np.int32).reshape(size, size)
    charts = np.asarray(chart_ids, dtype=np.int32)
    support_owner = np.full((size, size), -1, dtype=np.int32)
    support_bary = np.zeros((size, size, 3), dtype=np.float32)
    support_chart = np.full((size, size), -1, dtype=np.int32)
    collision = np.zeros((size, size), dtype=bool)
    ambiguous = np.zeros((size, size), dtype=bool)
    best_distance = np.full((size, size), np.inf, dtype=np.float64)
    tri_array = np.asarray(tris, dtype=np.int64)
    uv_array = np.asarray(uv, dtype=np.float64)
    corners = uv_array[tri_array] * float(size)
    selected_triangles = None if triangle_ids is None else np.asarray(triangle_ids, dtype=np.int64)
    for texels, candidate_triangles in conservative_pairs(
        uv_array, tri_array, size, triangle_ids=selected_triangles, budget=budget
    ):
        ys = texels // size
        xs = texels - ys * size
        keep = owner[ys, xs] < 0
        if not np.any(keep):
            continue
        for texel, triangle_id in zip(texels[keep].tolist(), candidate_triangles[keep].tolist()):
            y, x = divmod(int(texel), size)
            chart = int(charts[int(triangle_id)])
            point = np.array([x + 0.5, y + 0.5], dtype=np.float64)
            _closest, bary, distance = closest_point_on_uv_triangle(point, corners[int(triangle_id)])
            previous_chart = int(support_chart[y, x])
            if previous_chart >= 0 and previous_chart != chart:
                collision[y, x] = True
                support_owner[y, x] = -1
                support_chart[y, x] = -1
                continue
            if previous_chart >= 0:
                ambiguous[y, x] = True
            if distance < best_distance[y, x] - 1e-15 or (
                abs(distance - best_distance[y, x]) <= 1e-15
                and (support_owner[y, x] < 0 or int(triangle_id) < int(support_owner[y, x]))
            ):
                best_distance[y, x] = distance
                support_owner[y, x] = int(triangle_id)
                support_chart[y, x] = chart
                support_bary[y, x] = bary.astype(np.float32)
    support_owner[owner >= 0] = -1
    support_bary[owner >= 0] = 0.0
    support_chart[owner >= 0] = -1
    collision[owner >= 0] = False
    ambiguous[owner >= 0] = False
    return ConservativeSupport(support_owner, support_bary, support_chart, collision, ambiguous)


def chart_local_gutter(
    source_owner: np.ndarray,
    source_chart: np.ndarray,
    *,
    radius: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dilate support only through a single chart at a time.

    Returns ``(gutter_owner, gutter_chart, cross_chart_collision)``.  Gutter
    texels are sampling support only and must never be counted as direct or
    conservative provenance.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    owner = np.asarray(source_owner, dtype=np.int32)
    chart = np.asarray(source_chart, dtype=np.int32)
    if owner.shape != chart.shape or owner.ndim != 2:
        raise ValueError("source owner and chart maps must have the same 2-D shape")
    gutter_owner = owner.copy()
    gutter_chart = chart.copy()
    collision = np.zeros(owner.shape, dtype=bool)
    for _ in range(int(radius)):
        next_owner = gutter_owner.copy()
        next_chart = gutter_chart.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ys0, ys1 = max(0, dy), min(owner.shape[0], owner.shape[0] + dy)
            xs0, xs1 = max(0, dx), min(owner.shape[1], owner.shape[1] + dx)
            src = (slice(ys0 - dy, ys1 - dy), slice(xs0 - dx, xs1 - dx))
            dst = (slice(ys0, ys1), slice(xs0, xs1))
            candidate_chart = gutter_chart[src]
            empty = next_owner[dst] < 0
            conflict = (~collision[dst]) & (candidate_chart >= 0) & (next_chart[dst] >= 0) & (next_chart[dst] != candidate_chart)
            valid = empty & ~collision[dst] & (candidate_chart >= 0)
            collision[dst] |= conflict
            if np.any(conflict):
                next_owner[dst][conflict] = -1
                next_chart[dst][conflict] = -1
            take = valid & ~conflict
            if np.any(take):
                next_owner[dst][take] = gutter_owner[src][take]
                next_chart[dst][take] = candidate_chart[take]
        gutter_owner, gutter_chart = next_owner, next_chart
    return gutter_owner, gutter_chart, collision


def _tier_indices(extent: np.ndarray, tiers: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(tiers, extent, side="left")
    if np.any(indices >= tiers.size):
        raise ValueError("coverage tier list does not reach the requested atlas size")
    return indices


def conservative_pairs(
    uv: np.ndarray,
    tris: np.ndarray,
    size: int,
    *,
    triangle_ids: np.ndarray | None = None,
    budget: int = 2_000_000,
    tiers: tuple[int, ...] = DEFAULT_TIERS,
    epsilon: float = 1e-12,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(linear_texel, triangle_id)`` positive-area intersections.

    A texel is treated as the half-open unit cell ``[x,x+1) × [y,y+1)`` in
    atlas-pixel coordinates.  Triangle/cell intersection is evaluated with a
    2-D separating-axis test.  Strict projection overlap removes boundary-only
    contacts, which avoids declaring a shared UV edge to be surface area in
    both adjacent cells.

    This is diagnostic occupancy, not a visibility or provenance observation.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    uv_array = np.asarray(uv, dtype=np.float64)
    tri_array = np.asarray(tris, dtype=np.int64)
    if uv_array.ndim != 2 or uv_array.shape[1] != 2:
        raise ValueError("uv must have shape [vertex_count, 2]")
    if tri_array.ndim != 2 or tri_array.shape[1] != 3:
        raise ValueError("tris must have shape [triangle_count, 3]")

    if triangle_ids is None:
        selected = np.arange(tri_array.shape[0], dtype=np.int64)
    else:
        selected = np.asarray(triangle_ids, dtype=np.int64).reshape(-1)
        if selected.size and (selected.min() < 0 or selected.max() >= tri_array.shape[0]):
            raise ValueError("triangle_ids contains an out-of-range index")
        selected = np.unique(selected)
    if selected.size == 0:
        return

    corners = uv_array[tri_array[selected]] * float(size)
    signed_twice_area = (
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )
    positive_area = np.abs(signed_twice_area) > epsilon

    low = np.floor(corners.min(axis=1)).astype(np.int64)
    high = (np.ceil(corners.max(axis=1)) - 1.0).astype(np.int64)
    x0 = np.maximum(0, low[:, 0])
    x1 = np.minimum(size - 1, high[:, 0])
    y0 = np.maximum(0, low[:, 1])
    y1 = np.minimum(size - 1, high[:, 1])
    live_mask = positive_area & (x1 >= x0) & (y1 >= y0)
    live = np.flatnonzero(live_mask)
    if live.size == 0:
        return

    width = x1[live] - x0[live] + 1
    height = y1[live] - y0[live] + 1
    tier_values = np.asarray(tuple(sorted(set(tiers) | {int(size)})), dtype=np.int64)
    width_tier = _tier_indices(width, tier_values)
    height_tier = _tier_indices(height, tier_values)
    groups = np.unique(np.stack([width_tier, height_tier], axis=1), axis=0)

    for width_index, height_index in groups:
        group_mask = (width_tier == width_index) & (height_tier == height_index)
        group = live[group_mask]
        tx = int(tier_values[int(width_index)])
        ty = int(tier_values[int(height_index)])
        cells_per_triangle = tx * ty
        step = max(1, int(budget // max(cells_per_triangle, 1)))
        span_x = np.arange(tx, dtype=np.int64)
        span_y = np.arange(ty, dtype=np.int64)

        for start in range(0, group.size, step):
            local = group[start:start + step]
            batch_corners = corners[local]
            ix = x0[local][:, None] + span_x[None, :]
            iy = y0[local][:, None] + span_y[None, :]
            inside_bbox = (
                (ix <= x1[local][:, None])[:, None, :]
                & (iy <= y1[local][:, None])[:, :, None]
            )
            center_x = ix[:, None, :].astype(np.float64) + 0.5
            center_y = iy[:, :, None].astype(np.float64) + 0.5
            covered = inside_bbox.copy()

            for edge_index in range(3):
                p0 = batch_corners[:, edge_index]
                p1 = batch_corners[:, (edge_index + 1) % 3]
                edge = p1 - p0
                normal_x = edge[:, 1]
                normal_y = -edge[:, 0]
                projections = (
                    batch_corners[..., 0] * normal_x[:, None]
                    + batch_corners[..., 1] * normal_y[:, None]
                )
                tri_min = projections.min(axis=1)[:, None, None]
                tri_max = projections.max(axis=1)[:, None, None]
                box_center = (
                    center_x * normal_x[:, None, None]
                    + center_y * normal_y[:, None, None]
                )
                radius = 0.5 * (np.abs(normal_x) + np.abs(normal_y))[:, None, None]
                covered &= (box_center + radius > tri_min + epsilon)
                covered &= (box_center - radius < tri_max - epsilon)
                if not covered.any():
                    break

            rows, yy, xx = np.nonzero(covered)
            if rows.size == 0:
                continue
            texels = iy[rows, yy] * int(size) + ix[rows, xx]
            triangle_ids_out = selected[local[rows]]
            yield texels.astype(np.int64, copy=False), triangle_ids_out.astype(np.int64, copy=False)


def conservative_coverage(
    uv: np.ndarray,
    tris: np.ndarray,
    size: int,
    *,
    triangle_ids: np.ndarray | None = None,
    budget: int = 2_000_000,
) -> ConservativeCoverage:
    """Aggregate conservative claims into atlas-sized maps and per-triangle counts."""

    tri_array = np.asarray(tris, dtype=np.int64)
    if triangle_ids is None:
        selected = np.arange(tri_array.shape[0], dtype=np.int64)
    else:
        selected = np.unique(np.asarray(triangle_ids, dtype=np.int64).reshape(-1))

    uv_array = np.asarray(uv, dtype=np.float64)
    selected_corners = uv_array[tri_array[selected]] * float(size) if selected.size else np.empty((0, 3, 2))
    signed_twice_area = (
        (selected_corners[:, 1, 0] - selected_corners[:, 0, 0])
        * (selected_corners[:, 2, 1] - selected_corners[:, 0, 1])
        - (selected_corners[:, 1, 1] - selected_corners[:, 0, 1])
        * (selected_corners[:, 2, 0] - selected_corners[:, 0, 0])
    ) if selected.size else np.empty(0)
    positive = np.abs(signed_twice_area) > 1e-12
    if selected.size:
        low = np.floor(selected_corners.min(axis=1)).astype(np.int64)
        high = (np.ceil(selected_corners.max(axis=1)) - 1.0).astype(np.int64)
        clipped_live = (
            positive
            & (np.minimum(size - 1, high[:, 0]) >= np.maximum(0, low[:, 0]))
            & (np.minimum(size - 1, high[:, 1]) >= np.maximum(0, low[:, 1]))
        )
    else:
        clipped_live = np.empty(0, dtype=bool)

    claims = np.zeros(size * size, dtype=np.uint32)
    first = np.full(size * size, -1, dtype=np.int32)
    per_triangle = np.zeros(tri_array.shape[0], dtype=np.uint32)
    for texels, triangle_chunk in conservative_pairs(
        uv_array, tri_array, size, triangle_ids=selected, budget=budget
    ):
        was_empty = claims[texels] == 0
        if was_empty.any():
            first[texels[was_empty]] = triangle_chunk[was_empty].astype(np.int32)
        np.add.at(claims, texels, 1)
        np.add.at(per_triangle, triangle_chunk, 1)

    return ConservativeCoverage(
        claim_count=claims.reshape(size, size),
        first_triangle=first.reshape(size, size),
        claims_per_triangle=per_triangle,
        tested_triangles=int(selected.size),
        positive_area_triangles=int(positive.sum()),
        clipped_out_triangles=int((positive & ~clipped_live).sum()),
    )
