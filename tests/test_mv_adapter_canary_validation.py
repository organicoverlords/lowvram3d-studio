from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from run_mv_adapter_fp32_canary import inspect_image, validate_inspections  # noqa: E402


def test_black_identical_views_fail_closed() -> None:
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    inspections = [inspect_image(image, index) for index in range(6)]
    failures = validate_inspections(inspections)
    assert "PIXEL_IDENTICAL_OUTPUT_VIEWS" in failures
    assert "VIEW_0_NEAR_CONSTANT" in failures
    assert "VIEW_0_LOW_DYNAMIC_RANGE" in failures
    assert "VIEW_0_INSUFFICIENT_COLOUR_DIVERSITY" in failures


def test_distinct_high_variance_views_pass_numeric_gates() -> None:
    y, x = np.mgrid[:64, :64]
    inspections = []
    for index in range(6):
        rgb = np.stack(
            [
                (x * (index + 3) + y * 5) % 256,
                (y * (index + 7) + x * 11) % 256,
                ((x + y) * (index + 13)) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        inspections.append(inspect_image(Image.fromarray(rgb, mode="RGB"), index))
    assert validate_inspections(inspections) == []
