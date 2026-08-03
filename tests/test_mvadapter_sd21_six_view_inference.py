from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from mvadapter_sd21_six_view_inference import (  # noqa: E402
    _is_cuda_oom,
    _registered_iou,
    qa_outputs,
    validate_preflight,
)


def _images_and_masks(tmp_path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    names = ["horizontal_0", "horizontal_1", "horizontal_2", "horizontal_3", "top", "bottom"]
    for index, name in enumerate(names):
        mask = np.zeros((32, 32), np.uint8)
        mask[8:24, 8:24] = 255
        Image.fromarray(mask).save(tmp_path / f"{name}_mask.png")
        image = np.zeros((32, 32, 3), np.uint8)
        yy, xx = np.indices((16, 16))
        if index == 2:
            image[8:24, 8:24] = np.stack(
                [np.full((16, 16), 50), 220 - yy * 8, np.full((16, 16), 50)], axis=2
            )
        else:
            image[8:24, 8:24] = np.stack(
                [120 + xx * 5, np.full((16, 16), 50 + index * 8), np.full((16, 16), 40)],
                axis=2,
            )
        images.append(Image.fromarray(image, "RGB"))
    return images


def test_real_cuda_oom_is_distinguished_from_other_failures() -> None:
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not _is_cuda_oom(RuntimeError("attention shape mismatch"))


def test_illegal_memory_access_blocks_oom_fallback() -> None:
    config = ROOT / "configs" / "texture" / "gpu_panda_mvadapter_ig2mv_sd21_inference.json"
    try:
        validate_preflight(config, "oom-fallback")
    except RuntimeError as exc:
        assert "CONFIG_STATUS_INVALID" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("illegal-memory-access manifest authorized an OOM fallback")


def test_bounded_registration_reports_a_similarity_transform() -> None:
    mask = np.zeros((32, 32), bool)
    mask[8:24, 8:24] = True
    score, transform = _registered_iou(mask, mask)
    assert score == 1.0
    assert set(transform) == {"scale", "dx", "dy"}


def test_six_view_qa_requires_six_outputs_and_records_rear_numeric_gate(tmp_path: Path) -> None:
    images = _images_and_masks(tmp_path)
    report = qa_outputs(images, tmp_path, 32, ["front", "right", "rear", "left", "top", "bottom"])
    assert len(report["views"]) == 6
    assert report["structural_gate_passed"]
    assert report["colour_gate_passed"]
    assert report["semantic_gate"] == "USER_REVIEW_REQUIRED"
    assert report["rear_numeric_gate_passed"]
