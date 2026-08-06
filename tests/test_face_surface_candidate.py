from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from face_surface_candidate import (  # noqa: E402
    anchors_from_layers,
    camera_from_projection_fit,
    derive_face_mask,
    landmarks_from_face_mask,
)


def test_camera_conversion_matches_existing_projection() -> None:
    positions = np.asarray([[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]])
    fit = {"matrix": np.eye(3), "centre": [0.0, 0.0, 0.0], "offset": [10.0, 20.0], "scale": 4.0}
    camera = camera_from_projection_fit(fit, positions, 100, 120)
    points = np.asarray([[2.0, 3.0, 1.0], [-1.0, -2.0, 0.0]])
    projected, _depth = camera.project(points)
    assert np.allclose(projected, points[:, :2] * 4.0 + np.asarray([10.0, 20.0]))


def test_face_mask_is_bounded_upper_central_foreground() -> None:
    foreground = np.zeros((100, 100), dtype=bool)
    foreground[10:90, 20:80] = True
    mask = derive_face_mask(foreground)
    yy, xx = np.nonzero(mask)
    assert yy.max() < 50
    assert xx.min() > 20
    assert xx.max() < 80
    assert int(mask.sum()) > 64


def test_landmarks_snap_to_face_mask() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:30, 10:30] = True
    landmarks = landmarks_from_face_mask(mask)
    assert len(landmarks) == 7
    for item in landmarks:
        x, y = map(int, item["source_xy"])
        assert mask[y, x]


def test_anchors_choose_selected_deeper_surface() -> None:
    layers = {
        "pixels_xy": np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        "offsets": np.asarray([0, 2, 4, 6]),
        "triangle_ids": np.asarray([1, 5, 2, 6, 3, 7]),
        "barycentric": np.asarray(
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5],
             [1.0, 0.0, 0.0], [0.1, 0.4, 0.5],
             [1.0, 0.0, 0.0], [0.3, 0.3, 0.4]],
        ),
        "depth": np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
    }
    landmarks = [
        {"name": "a", "source_xy": [10.0, 10.0]},
        {"name": "b", "source_xy": [20.0, 20.0]},
        {"name": "c", "source_xy": [30.0, 30.0]},
    ]
    anchors = anchors_from_layers(layers, np.asarray([5, 6, 7]), landmarks)
    assert [anchor["triangle_id"] for anchor in anchors] == [5, 6, 7]
    assert all(abs(sum(anchor["barycentric"]) - 1.0) < 1e-9 for anchor in anchors)
