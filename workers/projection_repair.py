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


def face_id_matches_within_radius(
    face_id: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    triangle_id: int,
    radius: int = 0,
) -> np.ndarray:
    """Match a sampled point to a rendered face ID with raster quantization tolerance.

    The tolerance is measured in rendered pixels and is intentionally small.
    It handles a UV sample landing on an adjacent pixel when the source mesh
    triangle is subpixel-sized, without accepting an unrelated face outside
    the local raster footprint.
    """
    ids = np.asarray(face_id, dtype=np.int32)
    x = np.asarray(sample_x, dtype=np.int64)
    y = np.asarray(sample_y, dtype=np.int64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("sample coordinates must be matching 1-D arrays")
    if ids.ndim != 2:
        raise ValueError("face_id must be a 2-D array")
    result = np.zeros(x.shape, dtype=bool)
    radius = max(int(radius), 0)
    for dy in range(-radius, radius + 1):
        yy = np.clip(y + dy, 0, ids.shape[0] - 1)
        for dx in range(-radius, radius + 1):
            xx = np.clip(x + dx, 0, ids.shape[1] - 1)
            result |= ids[yy, xx] == int(triangle_id)
    return result
