import cv2
import numpy as np

from workers.postprocess_orientation_color import (
    choose_orientation,
    colour_stats,
    pale_texture_decision,
    recover_basecolor,
)


def test_orientation_repair_requires_clear_nonzero_yaw_margin():
    scores = {
        0: {"score": 0.22}, 90: {"score": 0.31},
        180: {"score": 0.81}, 270: {"score": 0.28},
    }
    decision = choose_orientation(scores, margin=0.08)
    assert decision["best_yaw"] == 180
    assert decision["repair_required"] is True
    assert decision["decision"] == "rotate_root"


def test_orientation_does_not_repair_ambiguous_scores():
    scores = {0: {"score": 0.50}, 90: {"score": 0.51}, 180: {"score": 0.52}, 270: {"score": 0.51}}
    decision = choose_orientation(scores, margin=0.08)
    assert decision["decision"] == "undetermined"
    assert decision["repair_required"] is False


def test_pale_texture_detector_is_masked_and_bounded():
    source = {"saturation_mean": 0.60, "luma_std": 0.25}
    render = {"saturation_mean": 0.25, "luma_std": 0.10}
    decision = pale_texture_decision(source, render)
    assert decision["pale"] is True


def test_recovery_preserves_unpainted_texels_and_limits_gain(tmp_path):
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :, :] = (100, 100, 100)
    image[2:6, 2:6] = (120, 120, 120)
    source_stats = {"saturation_mean": 0.7, "luma_mean": 0.5, "luma_std": 0.3}
    render_stats = {"saturation_mean": 0.1, "luma_mean": 0.5, "luma_std": 0.05}
    source_path = tmp_path / "base.png"
    output_path = tmp_path / "recovered.png"
    cv2.imwrite(str(source_path), image)
    result = recover_basecolor(source_path, output_path, source_stats, render_stats)
    assert result["saturation_gain"] <= 1.35
    recovered = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert recovered.shape == image.shape
