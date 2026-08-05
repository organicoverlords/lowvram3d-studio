"""Topology and semantics tests for the bounded bar repair."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from mesh_io import write_glb  # noqa: E402
from prove_bar_repair import direction_key  # noqa: E402
from prove_camera_semantics import mirror_scores  # noqa: E402
from repair_bar_local_closure import (  # noqa: E402
    aspect_ratios,
    boundary_loops,
    fit_plane,
    primitive_table,
    read_container,
    topology_stats,
    triangulate_loop,
)


def _open_box() -> tuple[np.ndarray, np.ndarray]:
    """A unit box whose +z face is missing, so it has one 4-edge boundary loop."""
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
    triangles = np.array(
        [[0, 2, 1], [0, 3, 2],           # -z
         [0, 1, 5], [0, 5, 4],           # -y
         [1, 2, 6], [1, 6, 5],           # +x
         [2, 3, 7], [2, 7, 6],           # +y
         [3, 0, 4], [3, 4, 7]], np.int64)  # -x
    return positions, triangles


def test_open_box_exposes_one_square_boundary_loop():
    positions, triangles = _open_box()
    loops = boundary_loops(triangles, {4, 5, 6, 7})
    assert len(loops) == 1
    assert sorted(loops[0]) == [4, 5, 6, 7]


def test_closure_seals_the_loop_and_keeps_orientation_consistent():
    positions, triangles = _open_box()
    loop = boundary_loops(triangles, {4, 5, 6, 7})[0]
    closure = np.asarray(triangulate_loop(list(reversed(loop)), positions), dtype=np.int64)
    assert len(closure) == len(loop) - 2

    sealed = np.concatenate((triangles, closure), axis=0)
    assert boundary_loops(sealed, {4, 5, 6, 7}) == []

    directed = set()
    for a, b, c in sealed:
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            assert edge not in directed, f"directed edge {edge} repeats"
            directed.add(edge)

    # The cap must face outward like the rest of the shell, i.e. along +z.
    normals = np.cross(positions[closure[:, 1]] - positions[closure[:, 0]],
                       positions[closure[:, 2]] - positions[closure[:, 0]])
    assert np.all(normals[:, 2] > 0)


def test_closure_never_increases_boundary_or_component_counts():
    positions, triangles = _open_box()
    loop = boundary_loops(triangles, {4, 5, 6, 7})[0]
    closure = np.asarray(triangulate_loop(list(reversed(loop)), positions), dtype=np.int64)
    before = topology_stats(positions, triangles)
    after = topology_stats(positions, np.concatenate((triangles, closure), axis=0))
    assert after["boundary_edges"] < before["boundary_edges"]
    assert after["non_manifold_edges"] <= before["non_manifold_edges"]
    assert after["components"] <= before["components"]
    assert after["zero_area_faces"] == 0
    assert after["duplicate_faces"] == 0


def test_uneven_loop_is_triangulated_without_a_sliver_fan():
    """A rim with one very short edge is exactly where greedy ear clipping leaves slivers."""
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    radius = np.where(np.arange(12) == 3, 0.985, 1.0)
    positions = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(12)], axis=1).astype(np.float32)
    loop = list(range(12))
    closure = np.asarray(triangulate_loop(loop, positions), dtype=np.int64)
    assert len(closure) == 10
    assert float(aspect_ratios(positions, closure).max()) < 4.0


def test_boundary_loops_ignores_unrelated_seams():
    """Only edges with both endpoints in the neighbourhood may join the chain."""
    positions, triangles = _open_box()
    detached_positions = np.concatenate(
        (positions, np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], np.float32)), axis=0)
    detached = np.concatenate((triangles, np.array([[8, 9, 10]], np.int64)), axis=0)
    loops = boundary_loops(detached, {4, 5, 6, 7})
    assert len(loops) == 1
    assert sorted(loops[0]) == [4, 5, 6, 7]


def test_fit_plane_reports_planarity():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    assert fit_plane(points)[4] == pytest.approx(0.0, abs=1e-12)
    tilted = points.copy()
    tilted[0, 2] = 0.5
    assert fit_plane(tilted)[4] > 0.05


def test_primitive_table_tracks_face_and_vertex_ranges(tmp_path):
    positions, triangles = _open_box()
    path = tmp_path / "box.glb"
    write_glb(path, positions, np.tile([0.0, 0.0, 1.0], (len(positions), 1)), None, triangles)
    meta, _binary = read_container(path)
    table = primitive_table(meta)
    assert len(table) == 1
    assert table[0]["face_start"] == 0
    assert table[0]["triangle_count"] == len(triangles)
    assert table[0]["vertex_end_exclusive"] == len(positions)


def test_direction_key_pairs_relabelled_bundles():
    """Raw index 0 in two bundles need not be the same camera; direction is the identity."""
    source = {"index": 3, "camera_direction": [-1.0, -0.0, 0.0], "semantic_name": "front"}
    repaired = {"index": 1, "camera_direction": [-0.9999, 1e-9, -0.0],
                "semantic_name": "horizontal_1"}
    assert direction_key(source) == direction_key(repaired) == (-1, 0, 0)
    assert direction_key({"camera_direction": [0.0, 0.0, 1.0]}) == (0, 0, 1)


def test_mirror_scores_separate_a_symmetric_shape_from_a_skewed_one():
    symmetric = np.zeros((64, 64), bool)
    symmetric[16:48, 24:40] = True
    skewed = np.zeros((64, 64), bool)
    skewed[16:48, 24:40] = True
    skewed[16:24, 40:56] = True
    assert mirror_scores(symmetric)["vertical_axis_mirror_iou"] == pytest.approx(1.0)
    assert mirror_scores(skewed)["vertical_axis_mirror_iou"] < 0.9
