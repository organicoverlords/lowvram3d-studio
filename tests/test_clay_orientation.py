"""Clay-render orientation must stand a subject up from any exporter convention.

The rotation is checked as pure vector algebra, without Blender, because the failure it guards
against was a wrong sign in hand-derived per-axis cases: the detector reported the right axis and
sign, and the renderer still laid the subject on its side.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender"))


def gltf_axis_in_blender(axis: int, sign: float) -> np.ndarray:
    """Mirror of the renderer's mapping: glTF (x, y, z) imports as Blender (x, -z, y)."""
    if axis == 0:
        return np.array([sign, 0.0, 0.0])
    if axis == 1:
        return np.array([0.0, 0.0, sign])
    return np.array([0.0, -sign, 0.0])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (np.linalg.norm(v) ** 2))


def stand_up(up_axis: int, up_sign: float, lateral_axis: int) -> np.ndarray:
    up = gltf_axis_in_blender(up_axis, up_sign)
    rotation = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    lateral = rotation @ gltf_axis_in_blender(lateral_axis, 1.0)
    flat = np.array([lateral[0], lateral[1], 0.0])
    if np.linalg.norm(flat) > 1e-6:
        angle = -math.atan2(flat[1], flat[0])
        spin = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
        rotation = spin @ rotation
    return rotation


CASES = [
    (1, 1.0, 0),    # Y-up, as our pipeline exports
    (2, -1.0, 0),   # Z-up sign-flipped, as the good online reference is stored
    (2, 1.0, 0),
    (0, 1.0, 1),
    (0, -1.0, 2),
    (1, -1.0, 2),
]


@pytest.mark.parametrize("up_axis,up_sign,lateral_axis", CASES)
def test_measured_up_always_ends_pointing_up(up_axis, up_sign, lateral_axis):
    rotation = stand_up(up_axis, up_sign, lateral_axis)
    result = rotation @ gltf_axis_in_blender(up_axis, up_sign)
    assert result[2] == pytest.approx(1.0, abs=1e-6), f"up landed at {result}"


@pytest.mark.parametrize("up_axis,up_sign,lateral_axis", CASES)
def test_lateral_ends_on_x_so_the_front_faces_the_camera(up_axis, up_sign, lateral_axis):
    rotation = stand_up(up_axis, up_sign, lateral_axis)
    result = rotation @ gltf_axis_in_blender(lateral_axis, 1.0)
    assert abs(result[0]) == pytest.approx(1.0, abs=1e-6), f"lateral landed at {result}"
    assert result[2] == pytest.approx(0.0, abs=1e-6)


def test_pipeline_convention_is_left_untouched():
    """Our own Y-up exports already stand correctly; the fix must not disturb them."""
    assert np.allclose(stand_up(1, 1.0, 0), np.eye(3), atol=1e-9)


def test_the_reference_convention_is_actually_rotated():
    """The Z-up sign-flipped case is the one that previously rendered lying down."""
    assert not np.allclose(stand_up(2, -1.0, 0), np.eye(3), atol=1e-6)
