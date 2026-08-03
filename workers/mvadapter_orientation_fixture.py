"""Genuinely asymmetric six-side orientation fixture.

The fixture exists to prove *semantic* six-view orientation from actually
rendered triangle evidence.  Direction vectors alone cannot prove that image
index N shows a particular side, nor that the horizontal sweep is left-handed
or right-handed, nor that the top and bottom views are rotated as declared.

Geometry is deliberately not rotationally or reflectionally symmetric:

    front   -Y  long narrow spike
    rear    +Y  short broad block
    right   +X  triangular fin
    left    -X  rectangular plate
    top     +Z  pyramid
    bottom  -Z  offset foot (offset in +X and +Y so no mirror plane survives)

A central core occludes the opposite side from every camera, so "which
components are visible" is a hard, falsifiable statement rather than a
tautology.  Each protruding component is narrower than the core in the two
axes orthogonal to its own axis, which is what makes the occlusion exact.

This module owns geometry and expectations only.  It never rasterises, never
imports torch, and never touches the production mesh.
"""
from __future__ import annotations

from typing import Any

import numpy as np


CORE_HALF = 0.32
"""Half-extent of the central occluder cube."""


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    tris = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, tris


def _pyramid(half: float, base_z: float, apex_z: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-half, -half, base_z], [half, -half, base_z], [half, half, base_z], [-half, half, base_z],
            [0.0, 0.0, apex_z],
        ],
        dtype=np.float64,
    )
    tris = np.asarray(
        [[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3], [3, 4, 0]],
        dtype=np.int64,
    )
    return vertices, tris


def _spike(half: float, base_y: float, apex_y: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-half, base_y, -half], [half, base_y, -half], [half, base_y, half], [-half, base_y, half],
            [0.0, apex_y, 0.0],
        ],
        dtype=np.float64,
    )
    tris = np.asarray(
        [[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=np.int64,
    )
    return vertices, tris


def _fin(x_base: float, x_tip: float, y_half: float, z_half: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangular prism: broad at the core, tapering to a point at +X."""
    vertices = np.asarray(
        [
            [x_base, -y_half, -z_half], [x_base, -y_half, z_half], [x_tip, -y_half, 0.0],
            [x_base, y_half, -z_half], [x_base, y_half, z_half], [x_tip, y_half, 0.0],
        ],
        dtype=np.float64,
    )
    tris = np.asarray(
        [
            [0, 1, 2], [3, 5, 4],
            [0, 2, 5], [0, 5, 3],
            [1, 4, 5], [1, 5, 2],
            [0, 3, 4], [0, 4, 1],
        ],
        dtype=np.int64,
    )
    return vertices, tris


# name -> (builder result, axis the component proves)
_COMPONENT_SPECS: tuple[tuple[str, tuple[np.ndarray, np.ndarray]], ...] = (
    ("core", _box((-CORE_HALF, -CORE_HALF, -CORE_HALF), (CORE_HALF, CORE_HALF, CORE_HALF))),
    ("front_spike", _spike(0.04, -CORE_HALF + 0.01, -1.00)),
    ("rear_block", _box((-0.28, CORE_HALF - 0.01, -0.28), (0.28, 0.72, 0.28))),
    ("right_fin", _fin(CORE_HALF - 0.01, 0.95, 0.05, 0.28)),
    ("left_plate", _box((-0.85, -0.22, -0.28), (-CORE_HALF + 0.01, 0.22, 0.28))),
    ("top_pyramid", _pyramid(0.25, CORE_HALF - 0.01, 1.00)),
    ("bottom_foot", _box((0.02, 0.02, -0.95), (0.28, 0.28, -CORE_HALF + 0.01))),
)

COMPONENT_IDS: dict[str, int] = {name: index for index, (name, _geometry) in enumerate(_COMPONENT_SPECS)}
COMPONENT_NAMES: tuple[str, ...] = tuple(name for name, _geometry in _COMPONENT_SPECS)

#: Component that must be *closest to the camera* for each world-space axis.
AXIS_SIGNATURE_COMPONENT: dict[str, str] = {
    "front": "front_spike",
    "rear": "rear_block",
    "right": "right_fin",
    "left": "left_plate",
    "top": "top_pyramid",
    "bottom": "bottom_foot",
}

#: Component that the core must fully occlude for each world-space axis.
AXIS_OCCLUDED_COMPONENT: dict[str, str] = {
    "front": "rear_block",
    "rear": "front_spike",
    "right": "left_plate",
    "left": "right_fin",
    "top": "bottom_foot",
    "bottom": "top_pyramid",
}

#: Outward world direction of each signature component, used to derive the
#: expected image-side placement from the declared camera basis.
COMPONENT_WORLD_DIRECTION: dict[str, list[float]] = {
    "front_spike": [0.0, -1.0, 0.0],
    "rear_block": [0.0, 1.0, 0.0],
    "right_fin": [1.0, 0.0, 0.0],
    "left_plate": [-1.0, 0.0, 0.0],
    "top_pyramid": [0.0, 0.0, 1.0],
    "bottom_foot": [0.0, 0.0, -1.0],
}


def build_fixture() -> dict[str, Any]:
    """Return the asymmetric fixture as vertices, triangles and per-triangle ids."""
    vertices: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    component_of_tri: list[np.ndarray] = []
    offset = 0
    for name, (component_vertices, component_tris) in _COMPONENT_SPECS:
        vertices.append(component_vertices)
        tris.append(component_tris + offset)
        component_of_tri.append(np.full(len(component_tris), COMPONENT_IDS[name], dtype=np.int32))
        offset += len(component_vertices)
    stacked_vertices = np.concatenate(vertices, axis=0)
    stacked_tris = np.concatenate(tris, axis=0).astype(np.int64)
    stacked_components = np.concatenate(component_of_tri, axis=0)
    if len(stacked_tris) != len(stacked_components):
        raise RuntimeError("FIXTURE_COMPONENT_ID_LENGTH_MISMATCH")
    if not _is_asymmetric(stacked_vertices):
        raise RuntimeError("FIXTURE_NOT_ASYMMETRIC")
    return {
        "vertices": stacked_vertices,
        "triangles": stacked_tris,
        "component_of_triangle": stacked_components,
        "component_ids": dict(COMPONENT_IDS),
        "component_names": list(COMPONENT_NAMES),
        "triangle_count": int(len(stacked_tris)),
        "vertex_count": int(len(stacked_vertices)),
    }


def _is_asymmetric(vertices: np.ndarray) -> bool:
    """Reject any fixture that survives a mirror or a 90/180 degree spin.

    The check is deliberately coarse but real: a symmetric point cloud maps
    onto itself under the transform, so the sorted coordinate tables match.
    """
    reference = np.sort(vertices, axis=0)
    transforms = [
        np.diag([-1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, 1.0]),
        np.diag([1.0, 1.0, -1.0]),
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    for transform in transforms:
        if np.allclose(np.sort(vertices @ transform.T, axis=0), reference, atol=1e-9):
            return False
    return True


def normalise(vertices: np.ndarray) -> np.ndarray:
    """Apply the same +/-0.5 normalisation the production control builder uses."""
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centred = vertices - centre
    largest = float(np.max(np.abs(centred)))
    if largest <= 1e-12:
        raise RuntimeError("FIXTURE_DEGENERATE")
    return centred * (0.5 / largest)
