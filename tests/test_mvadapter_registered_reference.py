from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from register_mvadapter_front_reference import register  # noqa: E402


def test_registered_reference_uses_rgba_and_neutral_background(tmp_path: Path) -> None:
    source = np.full((64, 64, 4), 0, np.uint8)
    source[12:52, 20:44, :3] = (20, 80, 180)
    source[12:52, 20:44, 3] = 255
    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)
    target = np.zeros((64, 64), np.uint8)
    target[12:52, 20:44] = 255
    target_path = tmp_path / "target.png"
    cv2.imwrite(str(target_path), target)
    output = tmp_path / "registered.png"
    report_path = tmp_path / "receipt.json"
    report = register(source_path, target_path, output, report_path, size=64)
    result = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert report["silhouette_iou"] >= 0.85
    assert report["mirror_used"] is False
    assert report["non_rigid_warp_used"] is False
    assert result.shape == (64, 64, 4)
    assert np.all(result[0, 0, :3] == 127)
