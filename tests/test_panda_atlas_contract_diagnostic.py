"""Focused unit tests for the image-independent panda atlas contract diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from panda_atlas_contract_diagnostic import (  # noqa: E402
    DEBUG_UNOWNED_RGB,
    build_unique_atlas,
    support_categories,
    triangle_debug_colors,
    triangle_support_counts,
)


def test_triangle_debug_colors_avoid_black_and_debug_magenta() -> None:
    colors = triangle_debug_colors(100_000)
    assert colors.shape == (100_000, 3)
    assert int(colors.min()) >= 40
    assert int(colors.max()) <= 200
    debug = np.asarray(DEBUG_UNOWNED_RGB)
    assert not np.any(np.all(colors == debug[None, :], axis=1))


def test_support_counts_and_categories_are_exact() -> None:
    owner = np.asarray([
        [-1, 0, 0, 1],
        [2, 2, 2, 3],
        [3, 3, 3, 3],
    ], dtype=np.int32)
    counts = triangle_support_counts(owner, 6)
    assert counts.tolist() == [2, 1, 3, 5, 0, 0]
    assert support_categories(counts) == {
        "zero": 2,
        "one": 1,
        "critical_1_to_3": 3,
        "low_4_to_8": 1,
        "adequate_9_or_more": 0,
        "under_4": 5,
        "under_9": 6,
    }


def test_unique_atlas_uses_magenta_only_for_unowned_space() -> None:
    owner = np.asarray([[-1, 0], [1, -1]], dtype=np.int32)
    colors = triangle_debug_colors(2)
    atlas = build_unique_atlas(owner, colors)
    debug = np.asarray(DEBUG_UNOWNED_RGB)
    assert np.array_equal(atlas[0, 0], debug)
    assert np.array_equal(atlas[1, 1], debug)
    assert np.array_equal(atlas[0, 1], colors[0])
    assert np.array_equal(atlas[1, 0], colors[1])
    assert int(np.all(atlas == debug[None, None, :], axis=2).sum()) == 2


def test_negative_triangle_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        triangle_debug_colors(-1)
