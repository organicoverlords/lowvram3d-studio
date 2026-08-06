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


#: The panda's real contract, abbreviated. `camera_direction` is not decoration:
#: `qa_outputs` derives the true opposite from it, and a fixture without it
#: raises KeyError. The previous fixture supplied only `camera_position`, so the
#: geometry-derived opposite shipped with no passing test behind it.
#:
#: Note the labels: index 0 "front" points [-1,0,0] and its true opposite
#: [1,0,0] is index 3, labelled "left". The horizontal names are rotated by one.
_PANDA_CONTRACT = [
    ("front",  [-1.0, 0.0, 0.0], [1.8, 0.0, 0.0]),
    ("right",  [0.0, -1.0, 0.0], [0.0, 1.8, 0.0]),
    ("rear",   [0.0, 1.0, 0.0],  [0.0, -1.8, 0.0]),
    ("left",   [1.0, 0.0, 0.0],  [-1.8, 0.0, 0.0]),
    ("top",    [0.0, 0.0, -1.0], [0.0, 0.0, 1.8]),
    ("bottom", [0.0, 0.0, 1.0],  [0.0, 0.0, -1.8]),
]


def _panda_style_images(tmp_path: Path) -> list[Image.Image]:
    """Six tiles standing in for the known-bad panda.

    The point of the fixture is the *shape* of that failure, which is not "the
    rear duplicates the front pixel for pixel". It is "the rear looks nothing
    like the front by any pixel statistic, and still has a face on it": 0.162
    direct, 0.113 mirrored. So index 3 -- the true opposite -- is given a
    visually unrelated pattern here. A fixture where the rear correlates highly
    with the front would be testing a defect the pipeline does not have, and
    would pass under the old numeric gate too.
    """
    images: list[Image.Image] = []
    names = ["horizontal_0", "horizontal_1", "horizontal_2", "horizontal_3", "top", "bottom"]
    yy, xx = np.indices((16, 16))
    for index, name in enumerate(names):
        mask = np.zeros((32, 32), np.uint8)
        mask[8:24, 8:24] = 255
        Image.fromarray(mask).save(tmp_path / f"{name}_mask.png")
        image = np.zeros((32, 32, 3), np.uint8)
        if index == 3:
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


def _panda_style_camera_views() -> list[dict]:
    mask_names = ["horizontal_0", "horizontal_1", "horizontal_2", "horizontal_3", "top", "bottom"]
    return [
        {
            "index": index,
            "proven_semantic": label,
            "axis_label": label,
            "semantic_name": label,
            "camera_position": position,
            "camera_direction": direction,
            "control_mask_filename": f"{mask_name}_mask.png",
        }
        for index, ((label, direction, position), mask_name)
        in enumerate(zip(_PANDA_CONTRACT, mask_names))
    ]


def test_six_view_qa_requires_six_outputs_and_records_rear_numeric_gate(tmp_path: Path) -> None:
    images = _images_and_masks(tmp_path)
    labels = ["front", "right", "rear", "left", "top", "bottom"]
    report = qa_outputs(images, tmp_path, 32, labels, _panda_style_camera_views())
    assert len(report["views"]) == 6
    assert report["structural_gate_passed"]
    assert report["colour_gate_passed"]
    assert report["rear_correlation_is_diagnostic_only"]


def test_low_rear_correlation_alone_cannot_pass_a_run(tmp_path: Path) -> None:
    """The known-bad panda regression.

    That asset has a full second face on its true rear and scores 0.162 direct /
    0.113 mirrored -- comfortably under the old 0.82 threshold, which is why it
    shipped green. Structural and colour gates pass on it too. The only thing
    standing between such a run and promotion must be the human verdict, so a
    run with no verdict file must not pass no matter how good the numbers are.
    """
    images = _panda_style_images(tmp_path)
    labels = ["front", "right", "rear", "left", "top", "bottom"]
    report = qa_outputs(images, tmp_path, 32, labels, _panda_style_camera_views())

    assert report["true_opposite_index"] == 3, "opposite must come from geometry, not label order"
    assert report["front_rear_direct_correlation"] < 0.82
    assert report["structural_gate_passed"] and report["colour_gate_passed"]
    assert report["true_rear_visual_verdict"] == "AMBIGUOUS"
    assert report["true_rear_visual_verdict_source"] == "MISSING"
    assert not report["passed"], "a run nobody reviewed must never report pass"
    assert "TRUE_REAR_VISUAL_VERDICT_AMBIGUOUS" in report["blocked_reason"]


def test_only_face_free_promotes_and_ambiguous_fails_closed(tmp_path: Path) -> None:
    import json

    images = _panda_style_images(tmp_path)
    labels = ["front", "right", "rear", "left", "top", "bottom"]
    camera_views = _panda_style_camera_views()
    verdict_path = tmp_path / "true_rear_verdict.json"

    for verdict, expected in (("FACE_PRESENT", False), ("AMBIGUOUS", False),
                              ("NONSENSE", False), ("FACE_FREE", True)):
        verdict_path.write_text(json.dumps({"verdict": verdict, "reviewer": "test"}),
                                encoding="utf-8")
        report = qa_outputs(images, tmp_path, 32, labels, camera_views, run_dir=tmp_path)
        assert report["passed"] is expected, f"{verdict} should {'' if expected else 'not '}pass"

    # An unreadable verdict file is not a pass either.
    verdict_path.write_text("{ not json", encoding="utf-8")
    assert not qa_outputs(images, tmp_path, 32, labels, camera_views, run_dir=tmp_path)["passed"]
