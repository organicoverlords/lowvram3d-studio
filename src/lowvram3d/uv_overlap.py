"""Exact positive-area UV overlap detection.

Two earlier metrics were unsound and must never be used as an acceptance gate:

* rasterising every triangle and counting pixels covered more than once counts the shared edge
  between adjacent triangles, which is ordinary connectivity rather than overlap;
* comparing summed analytic UV area against rasterised coverage conflates boundary rounding with
  genuine surface-on-surface overlap.

Both are retained by callers only as named diagnostics. The gate uses exact convex polygon
clipping: two triangles overlap only when their intersection has positive area. Shared vertices
and shared edges clip to zero area and therefore never count. Topological neighbours are *not*
excluded before testing, because adjacent triangles can still be folded over one another.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

GRID_SIZE = 256
AREA_EPSILON_UV = 0.25 / (2048 * 2048)
MAX_CANDIDATE_PAIRS = 5_000_000
TIMEOUT_SECONDS = 60.0
COORD_EPSILON = 1e-10
UV_LOWER_BOUND = -1e-6
UV_UPPER_BOUND = 1.000001


@dataclass
class UvOverlapReport:
    candidate_pair_count: int = 0
    tested_pair_count: int = 0
    positive_overlap_pair_count: int = 0
    positive_overlap_total_area_uv: float = 0.0
    positive_overlap_max_area_uv: float = 0.0
    positive_overlap_total_texels_equivalent: float = 0.0
    degenerate_uv_triangle_count: int = 0
    out_of_bounds_triangle_count: int = 0
    ignored_noise_intersection_count: int = 0
    timed_out: bool = False
    success: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "candidate_pair_count": self.candidate_pair_count,
            "tested_pair_count": self.tested_pair_count,
            "positive_overlap_pair_count": self.positive_overlap_pair_count,
            "positive_overlap_total_area_uv": self.positive_overlap_total_area_uv,
            "positive_overlap_max_area_uv": self.positive_overlap_max_area_uv,
            "positive_overlap_total_texels_equivalent": self.positive_overlap_total_texels_equivalent,
            "degenerate_uv_triangle_count": self.degenerate_uv_triangle_count,
            "out_of_bounds_triangle_count": self.out_of_bounds_triangle_count,
            "ignored_noise_intersection_count": self.ignored_noise_intersection_count,
            "timed_out": self.timed_out,
            "success": self.success,
            "errors": list(self.errors),
        }


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a convex subject polygon by a convex clipper triangle."""
    # Orient the clipper counter-clockwise so the inside test has a consistent sign.
    area = (clipper[1, 0] - clipper[0, 0]) * (clipper[2, 1] - clipper[0, 1]) - (
        clipper[2, 0] - clipper[0, 0]
    ) * (clipper[1, 1] - clipper[0, 1])
    if area < 0:
        clipper = clipper[::-1]

    output = subject
    for index in range(len(clipper)):
        if len(output) == 0:
            return np.empty((0, 2), np.float64)
        start = clipper[index]
        end = clipper[(index + 1) % len(clipper)]
        edge = end - start
        # Positive when the point lies to the left of the directed edge, i.e. inside.
        distance = edge[0] * (output[:, 1] - start[1]) - edge[1] * (output[:, 0] - start[0])
        inside = distance >= -COORD_EPSILON

        clipped: list[np.ndarray] = []
        count = len(output)
        for current in range(count):
            following = (current + 1) % count
            if inside[current]:
                clipped.append(output[current])
            if inside[current] != inside[following]:
                denominator = distance[current] - distance[following]
                if abs(denominator) > COORD_EPSILON:
                    t = distance[current] / denominator
                    clipped.append(output[current] + t * (output[following] - output[current]))
        output = np.asarray(clipped, np.float64) if clipped else np.empty((0, 2), np.float64)
    return output


def positive_area_uv_overlaps(
    uv_triangles: np.ndarray,
    atlas_resolution: int,
    *,
    timeout_seconds: float = TIMEOUT_SECONDS,
    max_candidate_pairs: int = MAX_CANDIDATE_PAIRS,
) -> UvOverlapReport:
    report = UvOverlapReport()
    triangles = np.asarray(uv_triangles, np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        report.errors.append("uv_triangles must have shape (F, 3, 2)")
        return report

    if not np.isfinite(triangles).all():
        report.errors.append("non-finite UV coordinates present")
        return report

    below = (triangles < UV_LOWER_BOUND).any(axis=(1, 2))
    above = (triangles > UV_UPPER_BOUND).any(axis=(1, 2))
    out_of_bounds = below | above
    report.out_of_bounds_triangle_count = int(out_of_bounds.sum())

    signed_area = 0.5 * (
        (triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    )
    degenerate = np.abs(signed_area) <= AREA_EPSILON_UV
    report.degenerate_uv_triangle_count = int(degenerate.sum())

    testable = ~(degenerate | out_of_bounds)
    indices = np.flatnonzero(testable)
    if indices.size == 0:
        report.success = not report.errors
        return report

    low = triangles[indices].min(axis=1)
    high = triangles[indices].max(axis=1)

    cell_low = np.clip((low * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)
    cell_high = np.clip((high * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)

    buckets: dict[int, list[int]] = {}
    for position, triangle_index in enumerate(indices):
        for cell_x in range(cell_low[position, 0], cell_high[position, 0] + 1):
            for cell_y in range(cell_low[position, 1], cell_high[position, 1] + 1):
                buckets.setdefault(cell_x * GRID_SIZE + cell_y, []).append(int(triangle_index))

    started = time.monotonic()
    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for first in range(len(members) - 1):
            for second in range(first + 1, len(members)):
                a, b = members[first], members[second]
                candidate_pairs.add((a, b) if a < b else (b, a))
                if len(candidate_pairs) > max_candidate_pairs:
                    report.candidate_pair_count = len(candidate_pairs)
                    report.errors.append(
                        f"candidate pair count exceeded {max_candidate_pairs}; failing closed"
                    )
                    return report
        if time.monotonic() - started > timeout_seconds:
            report.timed_out = True
            report.errors.append("timed out building candidate pairs; failing closed")
            return report

    report.candidate_pair_count = len(candidate_pairs)

    total_area = 0.0
    max_area = 0.0
    overlap_pairs = 0
    ignored = 0
    tested = 0

    for first, second in candidate_pairs:
        if time.monotonic() - started > timeout_seconds:
            report.timed_out = True
            report.errors.append("timed out during intersection testing; failing closed")
            return report
        a = triangles[first]
        b = triangles[second]
        if (
            a[:, 0].max() < b[:, 0].min() - COORD_EPSILON
            or b[:, 0].max() < a[:, 0].min() - COORD_EPSILON
            or a[:, 1].max() < b[:, 1].min() - COORD_EPSILON
            or b[:, 1].max() < a[:, 1].min() - COORD_EPSILON
        ):
            continue
        tested += 1
        intersection = _clip_convex(a.copy(), b.copy())
        area = _polygon_area(intersection)
        if area > AREA_EPSILON_UV:
            overlap_pairs += 1
            total_area += area
            max_area = max(max_area, area)
        elif area > 0.0:
            ignored += 1

    report.tested_pair_count = tested
    report.positive_overlap_pair_count = overlap_pairs
    report.positive_overlap_total_area_uv = total_area
    report.positive_overlap_max_area_uv = max_area
    report.positive_overlap_total_texels_equivalent = total_area * float(atlas_resolution) ** 2
    report.ignored_noise_intersection_count = ignored
    report.success = not report.errors and not report.timed_out
    return report
