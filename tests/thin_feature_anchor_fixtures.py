"""Small deterministic meshes for the thin-feature anchor contract tests."""
from __future__ import annotations

import numpy as np
import trimesh


def ordinary_body() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def detached_singleton(*, reverse_objects: bool = False) -> trimesh.Trimesh:
    body = ordinary_body()
    fragment = trimesh.Trimesh(
        vertices=np.asarray(
            ((1.35, -0.10, 0.25), (1.55, 0.10, 0.25), (1.45, 0.0, 0.48)),
            dtype=np.float64,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    order = (fragment, body) if reverse_objects else (body, fragment)
    return trimesh.util.concatenate(order)


def occluded_singleton() -> trimesh.Trimesh:
    body = ordinary_body()
    fragment = trimesh.Trimesh(
        vertices=np.asarray(
            ((-0.10, -0.10, 0.0), (0.10, -0.10, 0.0), (0.0, 0.10, 0.0)),
            dtype=np.float64,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    return trimesh.util.concatenate((body, fragment))


def attached_narrow_strip() -> trimesh.Trimesh:
    """A closed box with a narrow rectangular protrusion sharing its top annulus edges."""
    vertices = np.asarray(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-0.08, -0.08, 1.0),
            (0.08, -0.08, 1.0),
            (0.08, 0.08, 1.0),
            (-0.08, 0.08, 1.0),
            (-0.08, -0.08, 3.0),
            (0.08, -0.08, 3.0),
            (0.08, 0.08, 3.0),
            (-0.08, 0.08, 3.0),
        ),
        dtype=np.float64,
    )
    quads = (
        (0, 3, 2, 1),  # bottom
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (4, 5, 9, 8),  # top annulus
        (5, 6, 10, 9),
        (6, 7, 11, 10),
        (7, 4, 8, 11),
        (8, 9, 13, 12),  # protrusion sides
        (9, 10, 14, 13),
        (10, 11, 15, 14),
        (11, 8, 12, 15),
        (12, 13, 14, 15),  # protrusion cap
    )
    faces = []
    for a, b, c, d in quads:
        faces.extend(((a, b, c), (a, c, d)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
