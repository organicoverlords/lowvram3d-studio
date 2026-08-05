from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

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


def test_conditioning_reference_preserves_aspect_and_centres_subject(tmp_path: Path) -> None:
    source = np.zeros((90, 30, 4), np.uint8)
    source[5:85, 6:24] = (20, 60, 200, 255)
    source_path = tmp_path / "tall.png"
    cv2.imwrite(str(source_path), source)
    result = prepare(source_path, tmp_path / "out.png", tmp_path / "out.json", 128)
    width, height = result["resized_subject_dimensions"]
    # 18x80 subject scaled to a 128 canvas: longer side is 90%, aspect held.
    assert height == round(128 * 0.90)
    assert abs((18 / 80) - (width / height)) < 0.02
    assert result["aspect_ratio_preserved"] is True
    assert result["mesh_silhouette_iou_required"] is False
    centre_x, centre_y = result["subject_centre_fraction"]
    assert abs(centre_x - 0.5) <= 0.01 and abs(centre_y - 0.5) <= 0.01
    offset_x, offset_y = result["placement_offset"]
    assert offset_x == (128 - width) // 2 and offset_y == (128 - height) // 2
    assert 0.88 <= result["longer_dimension_fraction"] <= 0.92


def test_conditioning_reference_rejects_empty_foreground(tmp_path: Path) -> None:
    source = np.zeros((32, 32, 4), np.uint8)
    source_path = tmp_path / "empty.png"
    cv2.imwrite(str(source_path), source)
    with pytest.raises(RuntimeError, match="ALPHA_EMPTY"):
        prepare(source_path, tmp_path / "o.png", tmp_path / "o.json", 64)


def test_conditioning_background_is_neutral_gray_under_alpha(tmp_path: Path) -> None:
    source = np.zeros((40, 40, 4), np.uint8)
    source[8:32, 8:32] = (10, 220, 30, 255)
    source_path = tmp_path / "s.png"
    cv2.imwrite(str(source_path), source)
    prepare(source_path, tmp_path / "o.png", tmp_path / "o.json", 64)
    image = cv2.imread(str(tmp_path / "o.png"), cv2.IMREAD_UNCHANGED)
    background = image[image[:, :, 3] == 0][:, :3]
    assert background.size > 0
    assert np.all(background == 127)


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
