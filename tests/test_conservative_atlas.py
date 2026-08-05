from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from atlas_raster import injectivity, rasterise  # noqa: E402
from conservative_atlas import conservative_coverage  # noqa: E402


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
