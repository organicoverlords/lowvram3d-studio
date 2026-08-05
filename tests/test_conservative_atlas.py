from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from atlas_raster import injectivity, rasterise  # noqa: E402
from conservative_atlas import conservative_coverage  # noqa: E402


def _clip_polygon_axis(
    polygon: list[np.ndarray], axis: int, limit: float, keep_greater: bool
) -> list[np.ndarray]:
    if not polygon:
        return []
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= limit if keep_greater else previous[axis] <= limit
    for current in polygon:
        current_inside = current[axis] >= limit if keep_greater else current[axis] <= limit
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > 1e-15:
                amount = (limit - previous[axis]) / denominator
                output.append(previous + amount * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _triangle_cell_positive_area(corners: np.ndarray, x: int, y: int) -> bool:
    polygon = [point.copy() for point in corners]
    polygon = _clip_polygon_axis(polygon, 0, float(x), True)
    polygon = _clip_polygon_axis(polygon, 0, float(x + 1), False)
    polygon = _clip_polygon_axis(polygon, 1, float(y), True)
    polygon = _clip_polygon_axis(polygon, 1, float(y + 1), False)
    if len(polygon) < 3:
        return False
    points = np.asarray(polygon, dtype=np.float64)
    area_twice = abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    return bool(area_twice > 1e-10)


def test_subpixel_triangle_without_center_hit_is_recovered_as_gap() -> None:
    uv = np.array([[0.05, 0.05], [0.15, 0.05], [0.05, 0.15]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    owner, weights = rasterise(uv, tris, 4)
    assert owner.shape == (4, 4)
    assert weights.shape == (4, 4, 2)
    assert not np.any(owner == 0)
    coverage = conservative_coverage(uv, tris, 4)
    assert coverage.claims_per_triangle[0] > 0
    assert np.any(coverage.claim_count > 0)


def test_empty_center_census_reports_zero_interior_texels() -> None:
    uv = np.array([[0.05, 0.05], [0.15, 0.05], [0.05, 0.15]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    report = injectivity(uv, tris, 4)
    assert report["interior_texels"] == 0
    assert report["interior_texels_claimed_twice"] == 0
    assert report["max_interior_claims_on_one_texel"] == 0
    assert report["injective"] is True
    assert report["exact_overlap"]["tested_pair_count"] == 0


def test_center_samples_are_inside_positive_area_conservative_cells() -> None:
    uv = np.array(
        [[0.05, 0.05], [0.95, 0.10], [0.15, 0.90], [0.60, 0.55], [0.93, 0.60], [0.80, 0.93]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    owner, _ = rasterise(uv, tris, 16)
    coverage = conservative_coverage(uv, tris, 16)
    assert np.all(coverage.claim_count[owner >= 0] > 0)


def test_overlap_is_reported_not_silently_resolved() -> None:
    uv = np.array(
        [[0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.15, 0.15], [0.85, 0.15], [0.15, 0.85]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    coverage = conservative_coverage(uv, tris, 16)
    assert int(coverage.claim_count.max()) == 2
    assert np.any(coverage.claim_count > 1)


def test_boundary_only_contact_is_not_positive_area() -> None:
    uv = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    coverage = conservative_coverage(uv, tris, 2)
    assert coverage.claim_count[0, 0] == 1
    assert coverage.claim_count[0, 1] == 0
    assert coverage.claim_count[1, 0] == 0


def test_conservative_claims_match_bruteforce_polygon_clipping() -> None:
    rng = np.random.default_rng(20260805)
    size = 8
    triangle_count = 80
    uv = rng.uniform(-0.2, 1.2, size=(triangle_count * 3, 2))
    tris = np.arange(triangle_count * 3, dtype=np.int64).reshape(-1, 3)
    coverage = conservative_coverage(uv, tris, size)
    expected = np.zeros((size, size), dtype=np.uint32)
    corners = uv[tris] * float(size)
    for triangle in corners:
        signed_twice_area = (
            (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1]) * (triangle[2, 0] - triangle[0, 0])
        )
        if abs(signed_twice_area) <= 1e-12:
            continue
        for y in range(size):
            for x in range(size):
                if _triangle_cell_positive_area(triangle, x, y):
                    expected[y, x] += 1
    np.testing.assert_array_equal(coverage.claim_count, expected)


def test_degenerate_triangle_has_no_positive_area_claim() -> None:
    uv = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    coverage = conservative_coverage(uv, tris, 16)
    assert coverage.positive_area_triangles == 0
    assert int(coverage.claim_count.sum()) == 0
    assert int(coverage.claims_per_triangle[0]) == 0


def test_out_of_bounds_triangle_is_clipped_without_edge_clamping() -> None:
    uv = np.array([[-0.4, -0.4], [-0.2, -0.4], [-0.4, -0.2]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    coverage = conservative_coverage(uv, tris, 16)
    assert coverage.clipped_out_triangles == 1
    assert int(coverage.claim_count.sum()) == 0
