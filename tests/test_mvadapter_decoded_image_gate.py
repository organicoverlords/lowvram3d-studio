"""Decoded-image gate: did the pipeline produce content at all?

Deliberately weaker than ``qa_outputs``, and recorded before it, because the production QA gate
raises on a blank image and a numerical repair still needs the measured numbers when it does. The
failure this exists to catch is the one that already happened: twenty completed steps, a completed
decode, and six PNGs sharing one SHA-256.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from mvadapter_sd21_six_view_inference import _decoded_image_gate  # noqa: E402

SIZE = 32


def _noise(seed: int) -> Image.Image:
    generator = np.random.default_rng(seed)
    return Image.fromarray(generator.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8), "RGB")


def _flat(value: int) -> Image.Image:
    return Image.fromarray(np.full((SIZE, SIZE, 3), value, dtype=np.uint8), "RGB")


def test_six_distinct_noisy_views_pass():
    gate = _decoded_image_gate([_noise(index) for index in range(6)])

    assert gate["passed"] is True
    assert gate["distinct_image_hashes"] == 6
    assert gate["all_views_identical"] is False


def test_six_black_views_fail():
    gate = _decoded_image_gate([_flat(0) for _ in range(6)])

    assert gate["passed"] is False
    assert gate["all_non_black"] is False
    assert gate["all_views_identical"] is True


def test_a_single_black_view_fails_the_whole_gate():
    images = [_noise(index) for index in range(5)] + [_flat(0)]
    gate = _decoded_image_gate(images)

    assert gate["passed"] is False
    assert gate["all_non_black"] is False
    assert gate["views"][5]["non_black"] is False


def test_flat_mid_grey_views_fail_even_though_they_are_not_black():
    gate = _decoded_image_gate([_flat(128) for _ in range(6)])

    assert gate["all_non_black"] is True
    assert gate["all_non_flat"] is False
    assert gate["passed"] is False


def test_six_copies_of_one_good_image_fail():
    """Content is present, but the six views carry no independent information."""
    single = _noise(7)
    gate = _decoded_image_gate([single.copy() for _ in range(6)])

    assert gate["all_non_black"] is True
    assert gate["all_non_flat"] is True
    assert gate["all_views_identical"] is True
    assert gate["passed"] is False


@pytest.mark.parametrize("count", [0, 5, 7])
def test_a_wrong_view_count_fails(count):
    gate = _decoded_image_gate([_noise(index) for index in range(count)])

    assert gate["passed"] is False
    assert gate["image_count"] == count
