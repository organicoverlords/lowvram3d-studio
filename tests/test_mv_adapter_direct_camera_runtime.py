from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from mv_adapter_i2mv_camera_runtime import (  # noqa: E402
    AZIMUTHS,
    build_camera_to_world,
    build_orthographic_control_images,
    preprocess_rgba,
)


def test_camera_and_control_images_are_finite_without_optional_raster_stack() -> None:
    torch = pytest.importorskip("torch", reason="camera tensor runtime requires optional torch")

    c2w = build_camera_to_world(len(AZIMUTHS), "cpu")
    controls = build_orthographic_control_images(len(AZIMUTHS), 16, "cpu")

    assert tuple(c2w.shape) == (6, 4, 4)
    assert tuple(controls.shape) == (6, 6, 16, 16)
    assert torch.isfinite(c2w).all()
    assert torch.isfinite(controls).all()
    assert float(controls.min()) >= 0.0
    assert float(controls.max()) <= 1.0
    assert "nvdiffrast" not in sys.modules
    assert "triton" not in sys.modules


def test_rgba_preprocess_centres_subject_on_neutral_background() -> None:
    rgba = np.zeros((32, 16, 4), dtype=np.uint8)
    rgba[4:28, 3:13, :3] = np.array([120, 80, 40], dtype=np.uint8)
    rgba[4:28, 3:13, 3] = 255

    output = preprocess_rgba(Image.fromarray(rgba, mode="RGBA"), 64, 64)
    pixels = np.asarray(output)

    assert output.mode == "RGB"
    assert output.size == (64, 64)
    assert pixels.std() > 1.0
    assert np.allclose(pixels[0, 0], np.array([127, 127, 127]), atol=1)
