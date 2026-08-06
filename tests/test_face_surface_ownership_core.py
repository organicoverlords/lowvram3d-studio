from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from face_patch_texture import evaluate_tps, fit_tps, sample_premultiplied  # noqa: E402
from face_surface_ownership import (  # noqa: E402
    Camera,
    build_bvh,
    connected_patches,
    score_surface_patches,
    select_face_patch,
    trace_mask_layers,
    trace_ray,
)


def _layered_squares() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [-1.0, 1.0, 2.0],
        ], dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64)
    normals = np.tile(np.asarray([[0.0, 0.0, -1.0]]), (len(positions), 1))
    return positions, normals, triangles


def _camera() -> Camera:
    return Camera.from_dict(
        {
            "origin": [0.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "width": 5,
            "height": 5,
            "projection": "orthographic",
            "ortho_width": 2.0,
            "ortho_height": 2.0,
        }
    )


def test_camera_center_ray_and_projection_round_trip() -> None:
    camera = _camera()
    origin, direction = camera.pixel_rays(np.asarray([[2, 2]], dtype=np.float64))
    projected, depth = camera.project(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64))
    assert np.allclose(origin[0], [0.0, 0.0, 0.0])
    assert np.allclose(direction[0], [0.0, 0.0, 1.0])
    assert np.allclose(projected[0], [2.0, 2.0])
    assert np.allclose(depth, [1.0])


def test_multi_hit_raycast_preserves_depth_order_and_barycentrics() -> None:
    positions, _normals, triangles = _layered_squares()
    bvh = build_bvh(positions, triangles, leaf_size=1)
    triangle_ids, data = trace_ray(
        np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0]),
        positions, triangles, bvh,
    )
    assert len(triangle_ids) == 4
    assert np.all(np.diff(data[:, 0]) >= 0.0)
    assert np.allclose(np.unique(np.round(data[:, 0], 6)), [1.0, 2.0])
    assert np.allclose(data[:, 1:].sum(axis=1), 1.0)


def test_mask_layer_trace_keeps_near_and_deep_surfaces() -> None:
    positions, normals, triangles = _layered_squares()
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    layers = trace_mask_layers(
        positions, normals, triangles, _camera(), mask,
        stride=1, max_hits=8, leaf_size=1,
    )
    assert layers["offsets"].tolist() == [0, 4]
    assert np.allclose(np.unique(layers["depth"]), [1.0, 2.0])


def test_connected_patches_separate_unconnected_depth_layers() -> None:
    _positions, _normals, triangles = _layered_squares()
    patches = connected_patches(triangles, np.arange(4, dtype=np.int64))
    assert sorted(len(patch) for patch in patches) == [2, 2]


def test_patch_scoring_prefers_nearer_connected_surface() -> None:
    positions, normals, triangles = _layered_squares()
    layers = trace_mask_layers(
        positions, normals, triangles, _camera(), np.ones((5, 5), dtype=bool),
        stride=1, max_hits=8, leaf_size=1,
    )
    records = score_surface_patches(
        layers, triangles, positions, normals,
        np.zeros(len(triangles), dtype=np.int64), _camera(),
    )
    selected = select_face_patch(records, minimum_ray_coverage=0.1, maximum_side_wrap=1.0)
    assert set(selected["triangle_ids"].tolist()) == {0, 1}


def test_tps_interpolates_anchor_controls() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.asarray([[2.0, 3.0], [4.0, 3.0], [2.0, 5.0], [4.0, 5.0]])
    model = fit_tps(source, target, regularization=0.0)
    assert np.allclose(evaluate_tps(model, source), target, atol=1e-8)


def test_tps_rejects_duplicate_controls() -> None:
    source = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    target = np.asarray([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    try:
        fit_tps(source, target)
    except ValueError as error:
        assert str(error) == "TPS_DUPLICATE_CONTROL_POINT"
    else:
        raise AssertionError("duplicate TPS controls were accepted")


def test_premultiplied_sampling_does_not_bleed_white_background() -> None:
    image = np.asarray(
        [[[255, 255, 255], [255, 0, 0]], [[255, 255, 255], [255, 0, 0]]],
        dtype=np.uint8,
    )
    alpha = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    sampled, sampled_alpha = sample_premultiplied(image, alpha, np.asarray([[0.5, 0.5]]))
    assert np.allclose(sampled[0], [255.0, 0.0, 0.0], atol=1e-5)
    assert np.isclose(sampled_alpha[0], 0.5)
