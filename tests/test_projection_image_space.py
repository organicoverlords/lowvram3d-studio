"""The camera search must rasterise +Y up as row 0, not row N.

Model space has +Y up; an image has row indices increasing downward. Every
consumer of `rotation()` maps the rotated (x, y) straight onto (col, row), so a
missing negation rasterises the silhouette upside down. The search still returns
a pose -- the least-bad tilt that hides the inversion -- so nothing raises and
nothing logs. The only symptom is a flat score landscape, which reads as "this
subject is hard to register" rather than as a bug. It survived a full texture
run on the Mini Turbo shaman and produced a camouflage-patchwork atlas.

The obvious test is circular: rasterise the mesh through `project()` at yaw 0,
feed that back as the mask, and assert the search recovers yaw 0. Under the bug
the mask is generated flipped too, so it passes.

So the mask here is built by hand, with the convention written out explicitly:
the mesh carries a knob at high +Y, and the mask puts that knob in the TOP rows.
Nothing but `rotation()` decides whether those agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

pytest.importorskip("cv2")

import fast_texture_projection as F  # noqa: E402


def _lollipop():
    """A thin vertical bar with a knob at the +Y end. Asymmetric top-to-bottom."""
    vertices, triangles = [], []

    def box(cx, cy, half_x, half_y, half_z):
        base = len(vertices)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    vertices.append([cx + sx * half_x, cy + sy * half_y, sz * half_z])
        for a, b, c in ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
                        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)):
            triangles.append([base + a, base + b, base + c])

    box(0.0, 0.0, 0.06, 0.50, 0.06)     # the stick, centred
    box(0.0, 0.62, 0.22, 0.16, 0.06)    # the knob, unambiguously at +Y

    # `fit_camera` samples a fixed handful of barycentric points per triangle, so
    # 24 coarse faces rasterise as scattered dots and cap the IoU at ~0.04 no
    # matter how good the pose is. Subdivide until each face is small relative to
    # a probe texel, which is the regime a real mesh is already in.
    import trimesh

    vertices, triangles = trimesh.remesh.subdivide_to_size(
        np.asarray(vertices, np.float64), np.asarray(triangles, np.int64),
        max_edge=0.02)
    return np.asarray(vertices, np.float64), np.asarray(triangles, np.int32)


def _upright_mask(size=192):
    """Hand-drawn: knob in the top rows, stick below it. Row 0 is the top."""
    mask = np.zeros((size, size), bool)
    mid = size // 2
    # Knob: upper band, wide.
    mask[int(size * 0.06):int(size * 0.22),
         mid - int(size * 0.20):mid + int(size * 0.20)] = True
    # Stick: everything below it, narrow.
    mask[int(size * 0.22):int(size * 0.94),
         mid - int(size * 0.06):mid + int(size * 0.06)] = True
    return mask


def test_image_space_basis_flips_y():
    """The basis applied to every pose must negate y and leave x, z alone."""
    assert np.allclose(F.IMAGE_SPACE, np.diag([1.0, -1.0, 1.0]))
    # z untouched is what keeps depth ordering and the front-facing gate honest:
    # `facing` reads view_normal[:, 2] only.
    assert np.allclose(F.rotation(37.0, 11.0, 5.0)[2],
                       (np.diag([1.0, 1.0, 1.0]) @ F.rotation(37.0, 11.0, 5.0))[2])


def test_identity_pose_projects_plus_y_to_the_top_rows():
    """At yaw=pitch=roll=0, the knob must land in the upper half of the image."""
    vertices, _ = _lollipop()
    mask = _upright_mask()
    matrix = F.rotation(0.0, 0.0, 0.0)
    rotated = vertices @ matrix.T
    scale, offset = F.fit_to_mask(rotated, mask)
    rows = rotated[:, 1] * scale + offset[1]

    knob = vertices[:, 1] > 0.4
    assert knob.any()
    assert rows[knob].mean() < rows[~knob].mean(), (
        "the +Y knob rasterised BELOW the stick: image-space y is not negated")


def test_camera_search_recovers_the_upright_pose():
    """Against an independently drawn upright mask, the solve must be sharp."""
    vertices, triangles = _lollipop()
    mask = _upright_mask()

    camera = F.fit_camera(vertices, triangles, mask)
    iou = camera["silhouette_iou"]

    # A vertically asymmetric subject registered against its own upright
    # silhouette should score decisively. The inverted basis measured 0.585 on a
    # real subject whose whole 360 sweep sat between 0.475 and 0.585, so the
    # threshold is set where a flat landscape cannot reach.
    assert iou > 0.80, f"camera search only reached IoU {iou:.4f}"

    # And it should be found near upright rather than by tilting to hide a flip.
    pitch, roll = abs(camera["pitch"]), abs(camera["roll"])
    assert pitch <= 15.0 and roll <= 15.0, (
        f"solved by tilting: pitch={camera['pitch']} roll={camera['roll']}")
