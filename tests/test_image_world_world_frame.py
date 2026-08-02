import numpy as np
import pytest

from lowvram3d.image_world.contracts import ContractError
from lowvram3d.image_world.surface_projection import project_moge_surface, robust_xy_bounds
from lowvram3d.image_world.world_frame import (
    camera_to_world_rotation,
    estimate_world_up_from_normals,
    transform_camera_vectors,
)


def test_opencv_image_up_rotation_is_right_handed_and_z_up():
    rotation = camera_to_world_rotation(np.array([0.0, -1.0, 0.0]))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    point = transform_camera_vectors(np.array([1.0, 2.0, 3.0]), rotation)
    assert point == pytest.approx([1.0, 3.0, -2.0])


def test_world_up_estimator_recovers_spatially_broad_consensus():
    rng = np.random.default_rng(42)
    height, width = 60, 80
    true_up = np.array([0.0, -0.8, -0.6], dtype=np.float64)
    true_up /= np.linalg.norm(true_up)
    normals = np.zeros((height, width, 3), dtype=np.float64)
    normals[:] = true_up + rng.normal(0.0, 0.015, normals.shape)

    # Add strong but spatially narrow wall noise that must not win.
    normals[20:58, 4:14] = np.array([1.0, -0.05, 0.0])
    valid = np.ones((height, width), dtype=bool)
    estimate = estimate_world_up_from_normals(normals, valid)
    recovered = np.asarray(estimate.up_camera)
    assert float(np.dot(recovered, true_up)) > 0.995
    assert estimate.fallback_used is False
    assert estimate.spatial_coverage > 0.75
    assert estimate.support_fraction > 0.70


def test_world_up_estimator_fails_closed_without_horizontal_consensus():
    normals = np.zeros((20, 20, 3), dtype=np.float64)
    normals[..., 0] = 1.0
    valid = np.ones((20, 20), dtype=bool)
    with pytest.raises(ContractError):
        estimate_world_up_from_normals(normals, valid)


def test_world_up_fallback_is_explicit_and_zero_confidence():
    normals = np.zeros((20, 20, 3), dtype=np.float64)
    normals[..., 0] = 1.0
    valid = np.ones((20, 20), dtype=bool)
    estimate = estimate_world_up_from_normals(normals, valid, allow_fallback=True)
    assert estimate.fallback_used is True
    assert estimate.method == "opencv_image_up_fallback"
    assert estimate.confidence == 0.0


def test_robust_bounds_ignore_single_extreme_outlier():
    points = np.zeros((10, 10, 3), dtype=np.float64)
    yy, xx = np.mgrid[0:10, 0:10]
    points[..., 0] = xx
    points[..., 1] = yy
    points[0, 0, :2] = 1_000_000.0
    mask = np.ones((10, 10), dtype=bool)
    xmin, xmax, ymin, ymax = robust_xy_bounds(points, mask, padding_fraction=0.0)
    assert xmin >= 0.0
    assert xmax < 10_000.0
    assert ymin >= 0.0
    assert ymax < 10_000.0


def test_flat_moge_surface_produces_finite_unclassified_baseline():
    height, width = 24, 32
    image_y, image_x = np.mgrid[0:height, 0:width]
    points = np.zeros((height, width, 3), dtype=np.float64)
    points[..., 0] = (image_x - width / 2) * 0.1
    points[..., 1] = 0.0
    points[..., 2] = 2.0 + image_y * 0.1
    normals = np.zeros_like(points)
    normals[..., 1] = -1.0
    valid = np.ones((height, width), dtype=bool)

    result = project_moge_surface(
        points,
        normals,
        valid,
        grid_size=17,
        smoothing_iterations=2,
        stream_minimum_cells=4,
    )
    assert result.classification == "UNCLASSIFIED_SURFACE_BASELINE_NOT_TERRAIN_PROOF"
    assert result.frame.fallback_used is False
    assert np.isfinite(result.completed.height).all()
    assert np.isfinite(result.flow_accumulation).all()
    assert result.observation.observed_mask.any()
    assert np.array_equal(
        result.completed.height[result.observation.observed_mask],
        result.observation.height[result.observation.observed_mask],
    )


def test_surface_projection_rejects_shape_mismatch():
    points = np.zeros((8, 8, 3), dtype=np.float64)
    normals = np.zeros((8, 7, 3), dtype=np.float64)
    valid = np.ones((8, 8), dtype=bool)
    with pytest.raises(ContractError):
        project_moge_surface(points, normals, valid, grid_size=9)
