from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from face_surface_candidate_v2 import (  # noqa: E402
    border_connected_foreground,
    fixture_face_mask,
    select_patch_union,
    welded_surface_patches,
)


def test_border_connected_background_preserves_enclosed_white_object() -> None:
    image = np.full((80, 80, 3), 250, dtype=np.uint8)
    image[20:60, 20:60] = [70, 50, 30]
    image[30:50, 30:50] = [248, 248, 246]
    foreground, alpha, report = border_connected_foreground(image)
    assert not foreground[0, 0]
    assert foreground[25, 25]
    assert foreground[40, 40]
    assert alpha[40, 40] > 0.9
    assert report["foreground_pixels"] >= 1600


def test_fixture_face_mask_is_polygon_bounded() -> None:
    fixture = {"face_polygon": [[10, 10], [30, 10], [30, 30], [10, 30]]}
    mask = fixture_face_mask((50, 50), fixture)
    assert mask[20, 20]
    assert not mask[5, 5]
    assert int(mask.sum()) > 300


def test_welded_surface_connects_duplicate_vertex_indices() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0],
        ], dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    patches = welded_surface_patches(positions, triangles, np.asarray([0, 1, 2]), weld_tolerance=1e-6)
    assert sorted(len(patch) for patch in patches) == [1, 2]
    connected = next(patch for patch in patches if len(patch) == 2)
    assert set(connected.tolist()) == {0, 1}


def test_patch_union_keeps_depth_compatible_landmark_patches() -> None:
    records = [
        {
            "patch_id": 0, "triangle_ids": np.asarray([1, 2]), "landmark_support_count": 5,
            "ray_coverage": 0.20, "depth_median": 1.0, "depth_mad": 0.01,
            "score": 10.0,
        },
        {
            "patch_id": 1, "triangle_ids": np.asarray([3]), "landmark_support_count": 1,
            "ray_coverage": 0.03, "depth_median": 1.02, "depth_mad": 0.005,
            "score": 3.0,
        },
        {
            "patch_id": 2, "triangle_ids": np.asarray([4]), "landmark_support_count": 2,
            "ray_coverage": 0.05, "depth_median": 2.0, "depth_mad": 0.01,
            "score": 4.0,
        },
    ]
    selected, decision = select_patch_union(records, landmark_count=7)
    assert set(selected.tolist()) == {1, 2, 3}
    assert decision["selected_patch_ids"] == [0, 1]
    assert decision["landmark_support_count"] == 5
