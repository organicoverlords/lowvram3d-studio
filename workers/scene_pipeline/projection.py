"""Small deterministic camera and UV helpers for the MoGe source frame."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def project_points(points: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("POINTS_SHAPE_INVALID")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("INTRINSICS_INVALID")
    z = points[:, 2]
    if np.any(z <= 0) or not np.isfinite(z).all():
        raise ValueError("POINTS_NOT_POSITIVE_DEPTH")
    normalized = (intrinsics @ points.T).T
    normalized[:, 0] /= normalized[:, 2]
    normalized[:, 1] /= normalized[:, 2]
    return np.stack([normalized[:, 0] * width - 0.5, normalized[:, 1] * height - 0.5], axis=1)


def source_uv(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack([xx / max(width - 1, 1), 1.0 - yy / max(height - 1, 1)], axis=-1)


def reprojection_metrics(projected: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    error = np.linalg.norm(np.asarray(projected) - np.asarray(expected), axis=1)
    return {
        "count": int(len(error)),
        "median_px": float(np.median(error)) if len(error) else math.inf,
        "p99_px": float(np.percentile(error, 99)) if len(error) else math.inf,
        "max_px": float(error.max()) if len(error) else math.inf,
        "mean_px": float(error.mean()) if len(error) else math.inf,
    }
