"""Orientation normalisation for geometry-quality metrics.

These exist because assuming Y-up produced confident, wrong measurements: a Z-up reference had its
flank measured as its head band, and a sign-flipped one rendered upside down while every number
still looked plausible.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

pytest.importorskip("scipy")

from geometry_quality_metrics import canonical_axes, up_axis_sign  # noqa: E402


def figure(up_axis: int, lateral_axis: int, flip: bool = False):
    """A crude standing figure: a wide heavy base and a narrow light head, mirrored laterally."""
    rng = np.random.default_rng(7)
    depth_axis = [a for a in range(3) if a not in (up_axis, lateral_axis)][0]

    base = rng.uniform(-1.0, 1.0, size=(4000, 3))
    base[:, up_axis] = rng.uniform(0.0, 0.35, size=4000)
    head = rng.uniform(-0.18, 0.18, size=(300, 3))
    head[:, up_axis] = rng.uniform(0.80, 1.0, size=300)

    points = np.vstack([base, head])
    points[:, depth_axis] *= 0.35
    # Enforce lateral symmetry by mirroring the cloud onto itself.
    mirror = points.copy()
    mirror[:, lateral_axis] = -mirror[:, lateral_axis]
    points = np.vstack([points, mirror])
    if flip:
        points[:, up_axis] = -points[:, up_axis]
    return points


def triangles_for(points):
    count = (len(points) // 3) * 3
    return np.arange(count).reshape(-1, 3)


@pytest.mark.parametrize("up_axis", [0, 1, 2])
def test_up_axis_is_detected_regardless_of_convention(up_axis):
    lateral = (up_axis + 1) % 3
    points = figure(up_axis, lateral)
    _, detected_up, _ = canonical_axes(points)
    assert detected_up == up_axis


def test_lateral_axis_is_the_symmetric_one():
    points = figure(up_axis=2, lateral_axis=0)
    detected_lateral, _, _ = canonical_axes(points)
    assert detected_lateral == 0


def test_up_sign_follows_surface_area_not_width():
    """The heavy end is down. Width would answer wrongly for a subject widest at the top."""
    points = figure(up_axis=2, lateral_axis=0)
    tris = triangles_for(points)
    assert up_axis_sign(points, tris, 2) == 1.0

    flipped = figure(up_axis=2, lateral_axis=0, flip=True)
    assert up_axis_sign(flipped, triangles_for(flipped), 2) == -1.0
