from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from atlas_raster import rasterise  # noqa: E402
from conservative_atlas import (  # noqa: E402
    chart_local_gutter,
    closest_point_on_uv_triangle,
    derive_uv_chart_ids,
    resolve_conservative_support,
)


def test_chart_local_gutter_never_crosses_chart_boundary() -> None:
    owner = np.full((3, 5), -1, dtype=np.int32)
    chart = np.full((3, 5), -1, dtype=np.int32)
    owner[1, 0] = 10; chart[1, 0] = 0
    owner[1, 4] = 20; chart[1, 4] = 1
    gutter_owner, gutter_chart, collision = chart_local_gutter(owner, chart, radius=2)
    assert gutter_chart[1, 1] == 0
    assert gutter_chart[1, 3] == 1
    assert gutter_owner[1, 2] == -1
    assert collision[1, 2]


def test_chart_identity_uses_complete_index_shared_edges() -> None:
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5]], dtype=np.int64)
    chart_ids, report = derive_uv_chart_ids(uv, tris)
    assert chart_ids.tolist() == [0, 0, 1]
    assert report["cross_chart_connectivity"] == 0


def test_closest_point_handles_interior_edge_and_vertex() -> None:
    tri = np.array([[0, 0], [4, 0], [0, 4]], dtype=float)
    point, bary, distance = closest_point_on_uv_triangle(np.array([1, 1.0]), tri)
    np.testing.assert_allclose(point, [1, 1])
    np.testing.assert_allclose(bary, [0.5, 0.25, 0.25])
    assert distance == 0.0
    point, bary, _ = closest_point_on_uv_triangle(np.array([2, -1.0]), tri)
    np.testing.assert_allclose(point, [2, 0])
    np.testing.assert_allclose(bary, [0.5, 0.5, 0])
    point, bary, _ = closest_point_on_uv_triangle(np.array([-1, -1.0]), tri)
    np.testing.assert_allclose(point, [0, 0])
    np.testing.assert_allclose(bary, [1, 0, 0])


def test_support_preserves_direct_owner_and_separates_provenance() -> None:
    uv = np.array([[0.05, 0.05], [0.15, 0.05], [0.05, 0.15]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    direct, _ = rasterise(uv, tris, 4)
    charts, _ = derive_uv_chart_ids(uv, tris)
    support = resolve_conservative_support(uv, tris, 4, direct, charts)
    assert np.all(support.owner[direct >= 0] == -1)
    assert np.any(support.owner >= 0)
    assert np.all(support.barycentric[support.owner < 0] == 0)


def test_cross_chart_support_is_rejected() -> None:
    uv = np.array(
        [[0.05, 0.05], [0.45, 0.05], [0.05, 0.45],
         [0.05, 0.05], [0.45, 0.05], [0.05, 0.45]], dtype=float
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    direct = np.full((4, 4), -1, dtype=np.int32)
    charts = np.array([0, 1], dtype=np.int32)
    support = resolve_conservative_support(uv, tris, 4, direct, charts)
    assert np.any(support.collision)
    assert np.all(support.owner[ support.collision ] == -1)
