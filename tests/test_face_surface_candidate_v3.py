from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from face_surface_candidate_v3 import (  # noqa: E402
    anchor_aware_component,
    anchor_from_layers,
    bounded_fill_to_floor,
    build_edge_costs,
    candidate_ids_from_layers,
    clamp_triangle_count,
    connect_anchors_into_core,
    face_normals,
    fill_enclosed_interior,
    grow_geodesic_surface,
    largest_connected_component,
    points_in_polygon,
    score_map,
    weld_face_adjacency,
    weld_mesh_vertices,
    writable_texel_mask,
)


def test_points_in_polygon_inside_outside() -> None:
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)
    points = np.array([[5.0, 5.0], [-1.0, 5.0], [11.0, 5.0], [5.0, -1.0], [5.0, 11.0]], dtype=np.float64)
    inside = points_in_polygon(polygon, points)
    assert inside.tolist() == [True, False, False, False, False]


def test_points_in_polygon_complex_shape() -> None:
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]], dtype=np.float64)
    points = np.array([[5.0, 3.0], [2.0, 1.0], [8.0, 1.0], [5.0, 8.0]], dtype=np.float64)
    inside = points_in_polygon(polygon, points)
    assert inside[0]
    assert inside[1]
    assert inside[2]
    assert inside[3]


def test_candidate_ids_from_layers_filters_rank_and_facing() -> None:
    layers = {
        "triangle_ids": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        "offsets": np.array([0, 2, 4, 6], dtype=np.int64),
        "normal_facing": np.array([0.8, 0.6, 0.02, 0.9, 0.7, 0.01], dtype=np.float64),
    }
    candidate_ids = candidate_ids_from_layers(layers, max_rank=2, minimum_facing=0.05)
    assert set(candidate_ids.tolist()) == {1, 2, 4, 5}


def test_candidate_ids_from_layers_empty_layers() -> None:
    layers = {
        "triangle_ids": np.array([], dtype=np.int64),
        "offsets": np.array([0], dtype=np.int64),
        "normal_facing": np.array([], dtype=np.float64),
    }
    candidate_ids = candidate_ids_from_layers(layers)
    assert candidate_ids.size == 0


def test_anchor_from_layers_selects_front_most_hit() -> None:
    layers = {
        "pixels_xy": np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float64),
        "offsets": np.array([0, 2, 4, 6], dtype=np.int64),
        "triangle_ids": np.array([1, 5, 2, 6, 3, 7], dtype=np.int64),
        "barycentric": np.array(
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5],
             [1.0, 0.0, 0.0], [0.1, 0.4, 0.5],
             [1.0, 0.0, 0.0], [0.3, 0.3, 0.4]], dtype=np.float64
        ),
        "depth": np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        "normal_facing": np.array([0.8, 0.9, 0.8, 0.9, 0.8, 0.9], dtype=np.float64),
    }
    landmark = {"name": "nose", "source_xy": [10.0, 10.0]}
    anchor = anchor_from_layers(layers, landmark, minimum_facing=0.05, neighbour_rays=96)
    assert anchor["triangle_id"] == 1
    assert abs(sum(anchor["barycentric"]) - 1.0) < 1e-9


def test_anchor_from_layers_raises_when_no_valid_hit() -> None:
    layers = {
        "pixels_xy": np.array([[10.0, 10.0]], dtype=np.float64),
        "offsets": np.array([0, 2], dtype=np.int64),
        "triangle_ids": np.array([1, 2], dtype=np.int64),
        "barycentric": np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]], dtype=np.float64),
        "depth": np.array([1.0, 2.0], dtype=np.float64),
        "normal_facing": np.array([0.0, 0.01], dtype=np.float64),
    }
    landmark = {"name": "nose", "source_xy": [10.0, 10.0]}
    try:
        anchor_from_layers(layers, landmark, minimum_facing=0.05)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "FACE_ANCHOR_NOT_FOUND" in str(e)


def test_face_normals_unit_length() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    normals = face_normals(positions, triangles)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.allclose(normals[0], [0.0, 0.0, 1.0])


def test_weld_mesh_vertices_deterministic() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    welded1 = weld_mesh_vertices(positions, tolerance=1e-6)
    welded2 = weld_mesh_vertices(positions, tolerance=1e-6)
    assert np.array_equal(welded1, welded2)
    assert welded1[1] == welded1[3]
    assert welded1[2] == welded1[5]


def test_weld_mesh_vertices_different_tolerance() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0]], dtype=np.float64)
    welded_coarse = weld_mesh_vertices(positions, tolerance=1e-4)
    welded_fine = weld_mesh_vertices(positions, tolerance=1e-8)
    assert len(np.unique(welded_coarse)) == 1
    assert len(np.unique(welded_fine)) == 3


def test_weld_face_adjacency_3d_not_uv() -> None:
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0],
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    welded = weld_mesh_vertices(positions)
    candidate_ids = np.array([0, 1, 2], dtype=np.int64)
    adjacency = weld_face_adjacency(triangles, welded, candidate_ids)
    assert 0 in adjacency and 1 in adjacency[0]
    assert 2 not in adjacency[0]
    assert 2 not in adjacency[1]


def test_build_edge_costs_base_geodesic() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    projected_depths = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    assert abs(edge_costs[(0, 1)] - 1.0) < 1e-9
    assert abs(edge_costs[(1, 2)] - 1.0) < 1e-9


def test_build_edge_costs_depth_penalty() -> None:
    adjacency = {0: {1}, 1: {0}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    projected_depths = np.array([1.0, 2.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths, depth_penalty=1.0)
    assert edge_costs[(0, 1)] == 2.0


def test_build_edge_costs_chart_penalty() -> None:
    adjacency = {0: {1}, 1: {0}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    projected_depths = np.array([1.0, 1.0], dtype=np.float64)
    chart_ids = np.array([0, 1], dtype=np.int64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths, chart_ids=chart_ids, chart_penalty=5.0)
    assert edge_costs[(0, 1)] == 6.0


def test_grow_geodesic_surface_deterministic_multi_source() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float64)
    projected_depths = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    grown = grow_geodesic_surface([0, 3], adjacency, edge_costs, target_count=4)
    assert grown == {0, 1, 2, 3}


def test_grow_geodesic_surface_respects_target_count() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
    centroids = np.array([[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0]] * 5, dtype=np.float64)
    projected_depths = np.array([1.0] * 5, dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    grown = grow_geodesic_surface([0], adjacency, edge_costs, target_count=3)
    assert len(grown) == 3
    assert 0 in grown


def test_largest_connected_component() -> None:
    adjacency = {0: {1}, 1: {0}, 2: {3, 4}, 3: {2, 4}, 4: {2, 3}}
    selected = {0, 1, 2, 3}
    largest = largest_connected_component(selected, adjacency)
    # Both components {0,1} and {2,3} have size 2; tie-break by min element -> {0,1}
    assert largest == {0, 1}


def test_anchor_aware_component_prefers_anchored() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}, 3: {4}, 4: {3}}
    selected = {0, 1, 2, 3, 4}
    result = anchor_aware_component(selected, adjacency, [0])
    assert 0 in result
    assert 1 in result
    assert 3 not in result


def test_connect_anchors_into_core_bridges_missing() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float64)
    projected_depths = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    core = {0}
    result = connect_anchors_into_core(core, adjacency, edge_costs, [2])
    assert result == {0, 1, 2}


def test_bounded_fill_to_floor_reaches_minimum() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    core = {0}
    result = bounded_fill_to_floor(core, adjacency, min_triangles=3, max_triangles=4)
    assert len(result) >= 3
    assert 0 in result


def test_bounded_fill_to_floor_respects_maximum() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
    core = {0, 1, 2, 3, 4}
    result = bounded_fill_to_floor(core, adjacency, min_triangles=2, max_triangles=3)
    # Core already exceeds max_triangles; function returns it unchanged
    assert len(result) == 5


def test_clamp_triangle_count_raises_undersized() -> None:
    selected = {0, 1}
    try:
        clamp_triangle_count(selected, min_triangles=5, max_triangles=10)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "FACE_SURFACE_UNDERSIZED" in str(e)


def test_clamp_triangle_count_raises_oversized() -> None:
    selected = {0, 1, 2, 3, 4, 5}
    try:
        clamp_triangle_count(selected, min_triangles=2, max_triangles=4)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "FACE_SURFACE_OVERSIZED" in str(e)


def test_clamp_triangle_count_passes_within_bounds() -> None:
    selected = {0, 1, 2}
    result = clamp_triangle_count(selected, min_triangles=2, max_triangles=5)
    assert result == {0, 1, 2}


def test_fill_enclosed_interior_adds_enclosed() -> None:
    adjacency = {0: {1}, 1: {0, 2, 3}, 2: {1}, 3: {1}}
    # classification: 0 = STRICT_INTERIOR, 1 = BOUNDARY
    # Triangle 1 has two selected neighbours and is eligible for filling.
    classification = np.array([0, 0, 0, 0], dtype=np.int64)
    core = {0, 2}
    result = fill_enclosed_interior(core, adjacency, classification, max_additions=10)
    assert 1 in result


def test_fill_enclosed_interior_respects_max_additions() -> None:
    adjacency = {0: {1, 2}, 1: {0}, 2: {0, 3}, 3: {2, 4}, 4: {3}}
    classification = np.zeros(5, dtype=np.int64)
    core = {0, 3}
    result = fill_enclosed_interior(core, adjacency, classification, max_additions=1)
    assert len(result) == 3


def test_writable_texel_mask_exact_rasterization() -> None:
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    selected_ids = np.array([0], dtype=np.int64)
    mask = writable_texel_mask(uv, triangles, selected_ids, size=8)
    assert mask.sum() > 0
    assert mask.shape == (8, 8)


def test_score_map_computes_mean_edge_cost() -> None:
    adjacency = {0: {1}, 1: {0, 2}, 2: {1}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float64)
    projected_depths = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    selected_ids = np.array([0, 1], dtype=np.int64)
    scores = score_map(selected_ids, adjacency, edge_costs)
    assert abs(scores[0] - 1.0) < 1e-9
    assert abs(scores[1] - 1.0) < 1e-9


def test_candidate_ids_from_layers_rank_boundary() -> None:
    layers = {
        "triangle_ids": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
        "offsets": np.array([0, 2, 4, 6, 8], dtype=np.int64),
        "normal_facing": np.array([0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.04, 0.9], dtype=np.float64),
    }
    candidate_ids = candidate_ids_from_layers(layers, max_rank=2, minimum_facing=0.05)
    assert set(candidate_ids.tolist()) == {1, 2, 3, 4, 5, 6, 8}


def test_anchor_from_layers_distance_ordering() -> None:
    layers = {
        "pixels_xy": np.array([[10.0, 10.0], [100.0, 100.0]], dtype=np.float64),
        "offsets": np.array([0, 2, 4], dtype=np.int64),
        "triangle_ids": np.array([1, 2, 3, 4], dtype=np.int64),
        "barycentric": np.array(
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5],
             [0.3, 0.3, 0.4], [0.4, 0.3, 0.3]], dtype=np.float64
        ),
        "depth": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
        "normal_facing": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64),
    }
    landmark = {"name": "left_eye", "source_xy": [10.0, 10.0]}
    anchor = anchor_from_layers(layers, landmark, minimum_facing=0.05, neighbour_rays=96)
    assert anchor["triangle_id"] == 1


def test_grow_geodesic_surface_tie_break_by_triangle_id() -> None:
    adjacency = {0: {1, 2}, 1: {0}, 2: {0}}
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    face_normals_arr = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float64)
    projected_depths = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    edge_costs = build_edge_costs(adjacency, centroids, face_normals_arr, projected_depths)
    grown = grow_geodesic_surface([0], adjacency, edge_costs, target_count=2)
    assert grown == {0, 1}
