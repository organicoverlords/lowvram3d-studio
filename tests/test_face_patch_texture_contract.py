from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

import face_patch_texture as face_patch  # noqa: E402
from face_surface_ownership import Camera  # noqa: E402


def test_face_patch_reconstructs_third_barycentric_component(monkeypatch) -> None:
    owner = np.full((4, 4), -1, dtype=np.int32)
    barycentric_ab = np.zeros((4, 4, 2), dtype=np.float32)
    owner[1, 1] = 0
    barycentric_ab[1, 1] = (0.0, 0.0)
    owner[1, 2] = 0
    barycentric_ab[1, 2] = (1.0, 0.0)
    owner[2, 1] = 1
    barycentric_ab[2, 1] = (0.0, 0.0)

    def fake_rasterise(_uv: np.ndarray, _triangles: np.ndarray, _size: int):
        return owner, barycentric_ab

    monkeypatch.setattr(face_patch, "rasterise", fake_rasterise)
    baseline = np.full((4, 4, 3), 10, dtype=np.uint8)
    source = np.full((4, 4, 3), [200, 100, 50], dtype=np.uint8)
    alpha = np.ones((4, 4), dtype=np.float32)
    positions = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    camera = Camera.from_dict(
        {
            "origin": [0.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "width": 4,
            "height": 4,
            "projection": "orthographic",
            "ortho_width": 4.0,
            "ortho_height": 4.0,
        }
    )
    controls = np.asarray([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    output, report, writable = face_patch.build_face_patch_atlas(
        baseline, source, alpha, positions, uv, triangles, np.asarray([0]),
        camera, controls, controls, minimum_alpha=0.1,
    )
    assert report["atlas_raster_barycentric_components"] == 2
    assert report["accepted_face_texels"] == 2
    assert writable[1, 1] and writable[1, 2]
    assert not writable[2, 1]
    assert np.array_equal(output[~writable], baseline[~writable])


def test_face_patch_rejects_wrong_raster_weight_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        face_patch,
        "rasterise",
        lambda _uv, _triangles, size: (
            np.full((size, size), -1, dtype=np.int32),
            np.zeros((size, size, 3), dtype=np.float32),
        ),
    )
    baseline = np.zeros((4, 4, 3), dtype=np.uint8)
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    alpha = np.ones((4, 4), dtype=np.float32)
    positions = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    camera = Camera.from_dict(
        {
            "origin": [0.0, 0.0, 0.0], "right": [1.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0], "forward": [0.0, 0.0, 1.0],
            "width": 4, "height": 4, "projection": "orthographic",
            "ortho_width": 4.0, "ortho_height": 4.0,
        }
    )
    controls = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    try:
        face_patch.build_face_patch_atlas(
            baseline, source, alpha, positions, uv, triangles, np.asarray([0]),
            camera, controls, controls,
        )
    except ValueError as error:
        assert str(error) == "ATLAS_RASTER_BARYCENTRIC_CONTRACT_INVALID"
    else:
        raise AssertionError("wrong barycentric shape was accepted")
