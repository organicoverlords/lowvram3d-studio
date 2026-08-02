import numpy as np
import pytest

from workers.raster_project import projection_triangle_gate


def test_projection_gate_requires_depth_visibility_and_front_facing_normal():
    visibility = np.array([True, True, False, True])
    facing = np.array([0.9, 0.1, 0.9, np.nan], dtype=np.float32)
    assert projection_triangle_gate(visibility, facing).tolist() == [True, False, False, False]


def test_projection_gate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        projection_triangle_gate(np.array([True, False]), np.array([0.5]))
