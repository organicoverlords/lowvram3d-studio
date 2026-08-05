"""Regression tests for the MV-Adapter NaN-to-black false-success defect.

Job 26a37e41 recorded ``"success": true`` over six byte-identical pure-black PNGs: the SD2.1 UNet
emitted non-finite latents on Turing fp16 and diffusers' ``(images * 255).astype("uint8")`` turned
NaN into 0. These tests pin each condition that must now block a success receipt.
"""
from __future__ import annotations

import numpy as np
import pytest

from lowvram3d.view_validation import (
    is_effectively_blank,
    validate_generated_views,
    view_statistics,
)

REQUIRED = ("front", "right", "back", "left")


def textured_view(seed: int, size: int = 64) -> np.ndarray:
    """A plausible render: smoothly varying, plenty of distinct colours, clearly not blank."""
    rng = np.random.default_rng(seed)
    base = rng.random((size, size, 3))
    ramp = np.linspace(0.15, 0.9, size)[:, None, None]
    return np.clip(base * 0.35 + ramp, 0.0, 1.0)


def view_set(**overrides: np.ndarray) -> dict[str, np.ndarray]:
    views = {name: textured_view(index) for index, name in enumerate(REQUIRED)}
    views.update(overrides)
    return views


def test_healthy_view_set_passes() -> None:
    assert validate_generated_views(view_set(), REQUIRED) == []


def test_nan_view_is_rejected() -> None:
    poisoned = textured_view(0)
    poisoned[10, 10, 0] = np.nan
    problems = validate_generated_views(view_set(front=poisoned), REQUIRED)
    assert any("non-finite" in problem for problem in problems)


def test_infinity_view_is_rejected() -> None:
    poisoned = textured_view(1)
    poisoned[0, 0, 1] = np.inf
    problems = validate_generated_views(view_set(right=poisoned), REQUIRED)
    assert any("non-finite" in problem for problem in problems)


def test_all_black_view_set_is_rejected() -> None:
    """The exact observed failure: NaN latents already cast down to a uniform zero image."""
    black = {name: np.zeros((64, 64, 3)) for name in REQUIRED}
    problems = validate_generated_views(black, REQUIRED)
    assert any("effectively black or constant" in problem for problem in problems)


def test_byte_identical_views_are_rejected() -> None:
    """Six identical outputs are a generation failure even when the pixels are not black."""
    shared = textured_view(7)
    identical = {name: shared.copy() for name in REQUIRED}
    problems = validate_generated_views(identical, REQUIRED)
    assert any("byte-identical" in problem for problem in problems)


def test_constant_nonblack_view_set_is_rejected() -> None:
    grey = {name: np.full((64, 64, 3), 0.5) for name in REQUIRED}
    problems = validate_generated_views(grey, REQUIRED)
    assert any("effectively black or constant" in problem for problem in problems)


def test_single_blank_view_is_reported_without_claiming_all_blank() -> None:
    problems = validate_generated_views(view_set(back=np.zeros((64, 64, 3))), REQUIRED)
    assert problems
    assert not any("all required views" in problem for problem in problems)


def test_missing_required_view_is_rejected() -> None:
    views = view_set()
    del views["left"]
    problems = validate_generated_views(views, REQUIRED)
    assert any("missing required views" in problem for problem in problems)


def test_uint8_black_is_detected_like_float_black() -> None:
    """The saved-PNG form of the defect must be caught identically to the in-memory form."""
    black = {name: np.zeros((64, 64, 3), dtype=np.uint8) for name in REQUIRED}
    problems = validate_generated_views(black, REQUIRED)
    assert any("effectively black or constant" in problem for problem in problems)


def test_statistics_do_not_hide_nan_behind_a_finite_mean() -> None:
    poisoned = textured_view(3)
    poisoned[5, 5, 2] = np.nan
    statistics = view_statistics(poisoned)
    assert statistics["all_finite"] is False
    assert statistics["nonfinite_count"] == 1
    assert np.isfinite(statistics["mean"])


@pytest.mark.parametrize("value", [0.0, 1.0, 0.25])
def test_uniform_images_are_blank_at_any_level(value: float) -> None:
    assert is_effectively_blank(view_statistics(np.full((32, 32, 3), value)))


def test_textured_view_is_not_blank() -> None:
    assert not is_effectively_blank(view_statistics(textured_view(11)))
