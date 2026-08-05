from pathlib import Path

import numpy as np

from workers.surface_registration_forensics import _camera_projection


def test_camera_projection_is_model_agnostic():
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    result = _camera_projection(points, {
        "name": "front",
        "right": [1.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "origin": [0.0, 0.0, 0.0],
        "width": 256,
        "height": 256,
    })
    assert result["x"] == [0.0, 1.0]
    assert result["y"] == [0.0, 2.0]


def test_generic_utility_contains_no_character_specific_exception_table():
    source = Path("workers/surface_registration_forensics.py").read_text(encoding="utf-8").lower()
    for forbidden in ("panda", "red_panda", "owner_face", "approved_unknown"):
        assert forbidden not in source
