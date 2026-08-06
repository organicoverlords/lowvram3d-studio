from __future__ import annotations

import numpy as np
import pytest

from workers.front_face_registration_lock import (
    FaceRegistrationError,
    chart_local_support,
    estimate_bounded_similarity,
    luma_preserving_grade,
)


def test_bounded_similarity_reports_transform_and_rejects_large_translation():
    source = np.asarray([[10, 10], [30, 10], [20, 30]], dtype=np.float64)
    target = source + np.asarray([3, -2])
    matrix, report = estimate_bounded_similarity(source, target, max_translation=8)
    assert report["passed"] is True
    assert np.allclose(matrix[:, 2], [3, -2], atol=1e-6)
    with pytest.raises(FaceRegistrationError, match="TRANSLATION_LIMIT"):
        estimate_bounded_similarity(source, source + [20, 0], max_translation=8)


def test_chart_support_never_overwrites_direct_and_gutter_is_chart_local():
    atlas = np.zeros((5, 7, 3), np.uint8)
    atlas[2, 1] = (10, 20, 30)
    atlas[2, 5] = (200, 210, 220)
    direct = np.full((5, 7), -1, np.int32)
    direct[2, 1], direct[2, 5] = 1, 2
    conservative = np.full((5, 7), -1, np.int32)
    conservative[2, 2] = 7
    charts = np.zeros((5, 7), np.int32)
    charts[:, 4:] = 2
    result, gutter = chart_local_support(atlas, direct, conservative, charts, radius=1)
    assert tuple(result[2, 1]) == (10, 20, 30)
    assert tuple(result[2, 2]) == (10, 20, 30)
    assert not gutter[2, 3]  # chart 0 does not leak into chart 2


def test_luma_grade_is_bounded_and_does_not_touch_unmasked_pixels():
    atlas = np.full((2, 2, 3), 80, np.uint8)
    reference = np.full((2, 2, 3), 120, np.uint8)
    mask = np.asarray([[True, False], [False, False]])
    result, report = luma_preserving_grade(atlas, reference, mask, max_gain=0.1)
    assert tuple(result[0, 1]) == (80, 80, 80)
    assert report["gain"] <= 1.1
    assert result[0, 0, 0] > atlas[0, 0, 0]
