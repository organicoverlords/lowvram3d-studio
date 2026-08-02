import cv2
import numpy as np

from workers.gpu_texture_repair_sequence import (
    colour_stats,
    foreground_mask,
    luma_correlation,
    mask_iou,
)


def subject_image(*, mirrored=False, gray=False):
    image = np.zeros((96, 96, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    image[18:82, 24:72, 3] = 255
    image[18:82, 24:72, :3] = (110, 70, 30) if not gray else (120, 120, 120)
    image[26:38, 32:42, :3] = (230, 230, 230)
    image[26:38, 54:64, :3] = (20, 20, 20)
    image[52:72, 42:58, :3] = (40, 120, 210)
    return image[:, ::-1].copy() if mirrored else image


def test_alpha_foreground_mask_is_used():
    image = subject_image()
    mask = foreground_mask(image)
    assert mask.sum() == 64 * 48
    assert not mask[0, 0]


def test_mask_iou_handles_resize():
    a = np.zeros((32, 32), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[8:24, 8:24] = True
    b[16:48, 16:48] = True
    assert mask_iou(a, b) == 1.0


def test_front_like_rear_is_detectable_direct_and_mirrored():
    front = subject_image()
    same = subject_image()
    mirrored = subject_image(mirrored=True)
    assert luma_correlation(front, same) > 0.99
    assert luma_correlation(front, mirrored, mirror_b=True) > 0.99


def test_colour_stats_rejects_flat_gray_signal():
    colourful = subject_image()
    gray = subject_image(gray=True)
    colourful_stats = colour_stats(colourful, foreground_mask(colourful))
    gray_stats = colour_stats(gray, foreground_mask(gray))
    assert colourful_stats["saturation_mean"] > gray_stats["saturation_mean"]
    assert gray_stats["saturation_mean"] == 0.0
