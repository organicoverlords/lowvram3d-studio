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
