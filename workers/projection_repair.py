"""Pure guards shared by the bounded projection-repair lane.

These helpers intentionally do not alter geometry or UVs.  They make the
source-sample contract explicit so a raster implementation cannot treat a
depth-visible triangle as eligible when its sampled pixel belongs to another
triangle or lies outside the observed source mask.
"""
from __future__ import annotations

import numpy as np


def gated_sample_mask(
    *,
    depth_visible: bool,
    facing_score: float,
    face_id_match: np.ndarray,
    source_mask_valid: np.ndarray,
    confidence: np.ndarray,
    facing_threshold: float = 0.15,
    confidence_threshold: float = 0.20,
) -> np.ndarray:
    """Return the exact per-sample source-projection gate.

    All inputs after the scalar triangle gates are one-dimensional and must
    have identical lengths.  Invalid values fail closed.
    """
    face_id_match = np.asarray(face_id_match, dtype=bool)
    source_mask_valid = np.asarray(source_mask_valid, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float32)
    if face_id_match.ndim != 1 or source_mask_valid.ndim != 1 or confidence.ndim != 1:
        raise ValueError("sample gates must be one-dimensional")
    if not (face_id_match.shape == source_mask_valid.shape == confidence.shape):
        raise ValueError("sample gates must have matching shapes")
    triangle_ok = bool(depth_visible) and np.isfinite(float(facing_score)) and (
        float(facing_score) > float(facing_threshold)
    )
    if not triangle_ok:
        return np.zeros_like(face_id_match, dtype=bool)
    return (
        face_id_match
        & source_mask_valid
        & np.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )


def facial_source_region(
    source_mask: np.ndarray,
    *,
    top_fraction: float = 0.45,
    central_width_fraction: float = 0.70,
) -> np.ndarray:
    """Build a conservative source-facing head/face region from the matte.

    This is a bounded semantic safety mask, not a character-specific landmark
    model: it is the upper central portion of the observed foreground box.
    """
    mask = np.asarray(source_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("source_mask must be a 2-D array")
    ys, xs = np.nonzero(mask)
    result = np.zeros_like(mask, dtype=bool)
    if len(xs) == 0:
        return result
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width = (x1 - x0) * float(central_width_fraction)
    cx = (x0 + x1) * 0.5
    fx0 = max(x0, int(round(cx - width * 0.5)))
    fx1 = min(x1, int(round(cx + width * 0.5)))
    fy1 = min(y1, y0 + int(round((y1 - y0) * float(top_fraction))))
    result[y0:fy1, fx0:fx1] = mask[y0:fy1, fx0:fx1]
    return result


def rear_face_provenance_violations(
    rear_dominant: np.ndarray,
    winning_view: np.ndarray,
    winning_facial: np.ndarray,
) -> np.ndarray:
    """Return triangles that illegally receive front facial provenance."""
    rear_dominant = np.asarray(rear_dominant, dtype=bool)
    winning_view = np.asarray(winning_view, dtype=np.int32)
    winning_facial = np.asarray(winning_facial, dtype=bool)
    if not (rear_dominant.shape == winning_view.shape == winning_facial.shape):
        raise ValueError("provenance arrays must have matching shapes")
    return rear_dominant & (winning_view >= 0) & winning_facial

