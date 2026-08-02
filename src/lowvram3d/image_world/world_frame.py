"""Estimate a deterministic Z-up world frame from OpenCV camera-space normals.

MoGe point and normal maps use the OpenCV camera convention: +X right,
+Y down and +Z forward.  Terrain rasterization needs a world frame whose +Z
axis is up.  The estimator below searches for a spatially broad normal cluster,
which is usually supplied by water, ground or other approximately horizontal
surfaces.  It never silently calls the result proven terrain orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import ContractError


@dataclass(frozen=True)
class WorldFrameEstimate:
    up_camera: tuple[float, float, float]
    rotation_camera_to_world: tuple[tuple[float, float, float], ...]
    method: str
    confidence: float
    support_fraction: float
    spatial_coverage: float
    median_angle_degrees: float
    sampled_normal_count: int
    fallback_used: bool


def estimate_world_up_from_normals(
    normal_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lower_start_fraction: float = 0.20,
    angular_tolerance_degrees: float = 14.0,
    minimum_image_up_alignment: float = 0.10,
    minimum_support_fraction: float = 0.025,
    minimum_spatial_coverage: float = 0.08,
    max_samples: int = 20_000,
    max_candidates: int = 256,
    spatial_bins: int = 10,
    allow_fallback: bool = False,
) -> WorldFrameEstimate:
    """Estimate camera-space world up from a broad consensus of surface normals.

    Normals are sign-oriented toward image-up before clustering.  The winning
    cluster must have both angular support and image-space coverage so one wall
    or small roof patch cannot define the whole world frame.
    """

    normals = np.asarray(normal_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ContractError("normal_map must have shape HxWx3")
    if valid.shape != normals.shape[:2]:
        raise ContractError("valid_mask shape must match normal_map")
    if not 0.0 <= lower_start_fraction < 1.0:
        raise ContractError("lower_start_fraction must be in [0, 1)")
    if not 1.0 <= angular_tolerance_degrees <= 45.0:
        raise ContractError("angular tolerance must be between 1 and 45 degrees")
    if max_samples < 32 or max_candidates < 4 or spatial_bins < 2:
        raise ContractError("world-up sampling limits are too small")

    height, width = valid.shape
    row_start = int(math.floor(height * lower_start_fraction))
    finite = np.isfinite(normals).all(axis=-1)
    lengths = np.linalg.norm(normals, axis=-1)
    eligible = valid & finite & (lengths > 1e-8)
    eligible[:row_start] = False
    rows, cols = np.nonzero(eligible)
    if rows.size == 0:
        return _fallback_or_raise("no finite valid normals", allow_fallback)

    values = normals[rows, cols] / lengths[rows, cols, None]
    flip = values[:, 1] > 0.0
    values[flip] *= -1.0
    keep = -values[:, 1] >= minimum_image_up_alignment
    values, rows, cols = values[keep], rows[keep], cols[keep]
    if values.shape[0] < 8:
        return _fallback_or_raise("too few image-up-aligned normals", allow_fallback)

    sample_indices = _even_indices(values.shape[0], max_samples)
    samples = values[sample_indices]
    sample_rows = rows[sample_indices]
    sample_cols = cols[sample_indices]

    # Prefer candidate diversity rather than simply taking the first pixels.
    candidate_indices = _even_indices(samples.shape[0], max_candidates)
    candidates = samples[candidate_indices]
    cosine_tolerance = math.cos(math.radians(angular_tolerance_degrees))
    sample_bin_ids = _spatial_bin_ids(
        sample_rows, sample_cols, height=height, width=width, bins=spatial_bins
    )
    valid_bin_count = max(1, np.unique(sample_bin_ids).size)

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for candidate in candidates:
        dots = samples @ candidate
        supporters = dots >= cosine_tolerance
        support_fraction = float(supporters.mean())
        if not supporters.any():
            continue
        coverage = float(np.unique(sample_bin_ids[supporters]).size / valid_bin_count)
        image_up_preference = float(np.clip(-candidate[1], 0.0, 1.0))
        score = support_fraction * (0.35 + 0.65 * coverage) * (
            0.50 + 0.50 * image_up_preference
        )
        if best is None or score > best[0]:
            best = (score, candidate, supporters)

    if best is None:
        return _fallback_or_raise("no normal consensus candidate", allow_fallback)

    _, initial, initial_support = best
    refined = samples[initial_support].mean(axis=0)
    refined_norm = float(np.linalg.norm(refined))
    if refined_norm <= 1e-8:
        refined = initial
    else:
        refined /= refined_norm
    if refined[1] > 0.0:
        refined *= -1.0

    dots = np.clip(samples @ refined, -1.0, 1.0)
    supporters = dots >= cosine_tolerance
    support_fraction = float(supporters.mean())
    coverage = float(np.unique(sample_bin_ids[supporters]).size / valid_bin_count)
    angles = np.degrees(np.arccos(np.clip(dots[supporters], -1.0, 1.0)))
    median_angle = float(np.median(angles)) if angles.size else 180.0
    confidence = float(np.clip(math.sqrt(support_fraction * coverage), 0.0, 1.0))

    if (
        support_fraction < minimum_support_fraction
        or coverage < minimum_spatial_coverage
    ):
        reason = (
            f"world-up consensus too weak: support={support_fraction:.4f}, "
            f"coverage={coverage:.4f}"
        )
        return _fallback_or_raise(reason, allow_fallback)

    rotation = camera_to_world_rotation(refined)
    return WorldFrameEstimate(
        up_camera=tuple(float(value) for value in refined),
        rotation_camera_to_world=tuple(tuple(float(value) for value in row) for row in rotation),
        method="spatial_normal_consensus",
        confidence=confidence,
        support_fraction=support_fraction,
        spatial_coverage=coverage,
        median_angle_degrees=median_angle,
        sampled_normal_count=int(samples.shape[0]),
        fallback_used=False,
    )


def camera_to_world_rotation(up_camera: np.ndarray) -> np.ndarray:
    """Return a right-handed row-basis mapping camera XYZ to world XYZ."""

    up = np.asarray(up_camera, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ContractError("up_camera must be a finite 3-vector")
    norm = float(np.linalg.norm(up))
    if norm <= 1e-8:
        raise ContractError("up_camera must be non-zero")
    up /= norm

    camera_forward = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    forward = camera_forward - np.dot(camera_forward, up) * up
    if np.linalg.norm(forward) <= 1e-6:
        fallback_forward = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
        forward = fallback_forward - np.dot(fallback_forward, up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)
    rotation = np.stack((right, forward, up), axis=0)
    if np.linalg.det(rotation) < 0.999:
        raise ContractError("camera-to-world rotation is not right-handed")
    return rotation


def transform_camera_vectors(vectors: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if values.shape[-1] != 3 or matrix.shape != (3, 3):
        raise ContractError("vectors must end in XYZ and rotation must be 3x3")
    return np.einsum("ij,...j->...i", matrix, values)


def _even_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _spatial_bin_ids(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    height: int,
    width: int,
    bins: int,
) -> np.ndarray:
    row_bins = np.minimum((rows.astype(np.int64) * bins) // max(height, 1), bins - 1)
    col_bins = np.minimum((cols.astype(np.int64) * bins) // max(width, 1), bins - 1)
    return row_bins * bins + col_bins


def _fallback_or_raise(reason: str, allow_fallback: bool) -> WorldFrameEstimate:
    if not allow_fallback:
        raise ContractError(reason)
    up = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
    rotation = camera_to_world_rotation(up)
    return WorldFrameEstimate(
        up_camera=(0.0, -1.0, 0.0),
        rotation_camera_to_world=tuple(tuple(float(value) for value in row) for row in rotation),
        method="opencv_image_up_fallback",
        confidence=0.0,
        support_fraction=0.0,
        spatial_coverage=0.0,
        median_angle_degrees=180.0,
        sampled_normal_count=0,
        fallback_used=True,
    )
