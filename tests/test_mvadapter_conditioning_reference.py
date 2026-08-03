from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from prepare_mvadapter_conditioning_reference import prepare  # noqa: E402


def test_conditioning_reference_is_official_style_and_mesh_independent(tmp_path: Path) -> None:
    source = np.full((40, 24, 4), (240, 240, 240, 0), np.uint8)
    source[4:36, 5:19] = (30, 110, 180, 255)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    output = tmp_path / "conditioning.png"
    report = tmp_path / "conditioning.json"
    result = prepare(source_path, output, report, 64)
    assert result["passed"] is True
    assert 0.88 <= result["occupancy_fraction"] <= 0.92
    assert result["mesh_silhouette_comparison_used"] is False
    assert result["mirror_used"] is False
    assert result["non_rigid_warp_used"] is False
    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image.shape == (64, 64, 4)
    assert np.all(image[0, 0, :3] == 127)
    assert image[0, 0, 3] == 0


def test_conditioning_reference_is_deterministic(tmp_path: Path) -> None:
    source = np.zeros((20, 20, 4), np.uint8)
    source[2:18, 5:15] = (10, 40, 90, 255)
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    ra, rb = tmp_path / "a.json", tmp_path / "b.json"
    prepare(source_path, a, ra, 64)
    prepare(source_path, b, rb, 64)
    assert a.read_bytes() == b.read_bytes()
