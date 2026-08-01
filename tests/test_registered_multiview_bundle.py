from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

import build_registered_multiview_bundle as bundle  # noqa: E402


def test_required_view_order_is_stable() -> None:
    assert bundle.VIEW_NAMES == ("front", "right", "back", "left")


def test_choose_pair_selects_direct_assignment(monkeypatch) -> None:
    positive = np.array([0.0, 0.0, 1.0])
    negative = np.array([0.0, 0.0, -1.0])
    calls = iter([0.90, 0.88, 0.22, 0.20])

    def fake_score(*args, **kwargs):
        return {"iou": next(calls)}

    monkeypatch.setattr(bundle, "_score_direction", fake_score)
    directions, report = bundle._choose_pair(
        "front",
        "back",
        np.zeros((1, 1), dtype=bool),
        np.zeros((1, 1), dtype=bool),
        positive,
        negative,
        np.zeros((1, 3)),
        np.zeros((1, 3), dtype=np.int32),
        1.0,
        16,
    )

    assert report["selected"] == "direct"
    assert np.array_equal(directions["front"], positive)
    assert np.array_equal(directions["back"], negative)
    assert report["direct_score"] == 1.78
    assert report["swapped_score"] == 0.42


def test_choose_pair_selects_swapped_assignment(monkeypatch) -> None:
    positive = np.array([1.0, 0.0, 0.0])
    negative = np.array([-1.0, 0.0, 0.0])
    calls = iter([0.31, 0.34, 0.86, 0.83])

    def fake_score(*args, **kwargs):
        return {"iou": next(calls)}

    monkeypatch.setattr(bundle, "_score_direction", fake_score)
    directions, report = bundle._choose_pair(
        "right",
        "left",
        np.zeros((1, 1), dtype=bool),
        np.zeros((1, 1), dtype=bool),
        positive,
        negative,
        np.zeros((1, 3)),
        np.zeros((1, 3), dtype=np.int32),
        1.0,
        16,
    )

    assert report["selected"] == "swapped"
    assert np.array_equal(directions["right"], negative)
    assert np.array_equal(directions["left"], positive)
    assert report["direct_score"] == 0.65
    assert report["swapped_score"] == 1.69


def test_bundle_thresholds_are_fail_closed() -> None:
    assert bundle.DEFAULT_MIN_REGISTRATION_IOU >= 0.50
    assert bundle.DEFAULT_MIN_VISIBLE_UNION_PERCENT >= 45.0
