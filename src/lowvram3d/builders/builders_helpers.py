"""Small shared helpers for deterministic builder input normalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def spline_segments(points: Any) -> list[tuple[list[float], list[float]]]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) < 2:
        raise ValueError("spline points require at least two points")
    normalized = [[float(value) for value in point] for point in points]
    if any(len(point) != 3 for point in normalized):
        raise ValueError("spline points must be three-vectors")
    return list(zip(normalized, normalized[1:]))
