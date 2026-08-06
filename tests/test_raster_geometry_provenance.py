from __future__ import annotations

import numpy as np
import pytest

from lowvram3d.anchor_provenance import (
    GEOMETRY_HASH_FRAME,
    geometry_sha256,
    provenance_record,
)
from lowvram3d.raster_route import verified_cleanup_geometry_hash


def _triangle(offset: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(offset, dtype=np.float64) + np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    return vertices, np.asarray([[0, 1, 2]], dtype=np.int64)


def test_geometry_hash_translation_matches_centered_raster_frame() -> None:
    source_vertices, triangles = _triangle((12.5, -3.0, 7.25))
    centered_vertices, _ = _triangle((0.0, 0.0, 0.0))
    assert geometry_sha256(source_vertices, triangles) == geometry_sha256(centered_vertices, triangles)
    assert provenance_record(
        receipt_sha256="a" * 64,
        anchor_ids=[],
        input_geometry_sha256="b" * 64,
        output_geometry_sha256="b" * 64,
        geometry_unchanged=True,
    )["geometry_hash_frame"] == GEOMETRY_HASH_FRAME


def test_cleanup_geometry_mutation_rejects_promotion() -> None:
    expected = "1" * 64
    report = {
        "success": True,
        "provenance": {
            "input_geometry_sha256": expected,
            "output_geometry_sha256": "2" * 64,
        },
    }
    with pytest.raises(ValueError, match="geometry hash mismatch"):
        verified_cleanup_geometry_hash(report, expected)
