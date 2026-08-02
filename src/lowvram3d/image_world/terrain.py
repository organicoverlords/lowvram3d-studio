"""Deterministic terrain reconstruction primitives.

These functions operate on model observations but contain no ML dependency.
They preserve observed samples and explicitly mark generated cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractError


@dataclass(frozen=True)
class HeightfieldObservation:
    height: np.ndarray
    observed_mask: np.ndarray
    sample_count: np.ndarray
    variance: np.ndarray
    confidence: np.ndarray
    xy_bounds: tuple[float, float, float, float]

    def validate(self) -> None:
        shape = self.height.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ContractError("heightfield must be a square 2D array")
        for name in ("observed_mask", "sample_count", "variance", "confidence"):
            if getattr(self, name).shape != shape:
                raise ContractError(f"{name} shape must match height")
        if not np.isfinite(self.height[self.observed_mask]).all():
            raise ContractError("observed heights must be finite")
        if (self.sample_count < 0).any():
            raise ContractError("sample counts cannot be negative")
        if ((self.confidence < 0.0) | (self.confidence > 1.0)).any():
            raise ContractError("confidence must be between zero and one")


@dataclass(frozen=True)
class CompletedHeightfield:
    height: np.ndarray
    observed_mask: np.ndarray
    generated_mask: np.ndarray

    def validate(self) -> None:
        if self.height.shape != self.observed_mask.shape or self.height.shape != self.generated_mask.shape:
            raise ContractError("completed heightfield arrays must share a shape")
        if not np.isfinite(self.height).all():
            raise ContractError("completed heightfield must be finite")
        if np.any(self.observed_mask & self.generated_mask):
            raise ContractError("observed and generated masks must not overlap")
        if not np.all(self.observed_mask | self.generated_mask):
            raise ContractError("every cell must be observed or generated")


def rasterize_point_map(
    point_map: np.ndarray,
    terrain_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    grid_size: int = 513,
    xy_bounds: tuple[float, float, float, float] | None = None,
    minimum_confidence: float = 0.05,
) -> HeightfieldObservation:
    """Rasterize valid XYZ terrain observations into a top-down square grid.

    Per-cell height uses the median so a tree or landmark point that escaped
    semantic masking cannot dominate a cell as a maximum-height splat would.
    """

    points = np.asarray(point_map, dtype=np.float64)
    mask = np.asarray(terrain_mask, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ContractError("point_map must have shape HxWx3")
    if mask.shape != points.shape[:2]:
        raise ContractError("terrain_mask shape must match point_map")
    if grid_size < 2:
        raise ContractError("grid_size must be at least two")

    if confidence is None:
        conf = np.ones(mask.shape, dtype=np.float64)
    else:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape != mask.shape:
            raise ContractError("confidence shape must match terrain_mask")
        if not np.isfinite(conf).all() or ((conf < 0.0) | (conf > 1.0)).any():
            raise ContractError("confidence must be finite and between zero and one")

    valid = mask & np.isfinite(points).all(axis=-1) & (conf >= minimum_confidence)
    if not valid.any():
        raise ContractError("no valid terrain points remain after filtering")

    xyz = points[valid]
    weights = conf[valid]
    if xy_bounds is None:
        xmin, ymin = np.min(xyz[:, :2], axis=0)
        xmax, ymax = np.max(xyz[:, :2], axis=0)
        xspan = max(float(xmax - xmin), 1e-9)
        yspan = max(float(ymax - ymin), 1e-9)
        pad = 1e-6 * max(xspan, yspan)
        bounds = (float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad))
    else:
        bounds = tuple(float(value) for value in xy_bounds)
        if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ContractError("xy_bounds must be (xmin, xmax, ymin, ymax) with positive spans")

    xmin, xmax, ymin, ymax = bounds
    gx = np.floor((xyz[:, 0] - xmin) / (xmax - xmin) * grid_size).astype(np.int64)
    gy = np.floor((xyz[:, 1] - ymin) / (ymax - ymin) * grid_size).astype(np.int64)
    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)
    linear = gy * grid_size + gx

    height = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    sample_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    variance = np.zeros((grid_size, grid_size), dtype=np.float32)
    cell_confidence = np.zeros((grid_size, grid_size), dtype=np.float32)

    order = np.argsort(linear, kind="stable")
    linear = linear[order]
    z = xyz[:, 2][order]
    weights = weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear)) + 1]
    ends = np.r_[starts[1:], linear.size]
    for start, end in zip(starts, ends):
        cell = int(linear[start])
        row, col = divmod(cell, grid_size)
        values = z[start:end]
        height[row, col] = np.float32(np.median(values))
        sample_count[row, col] = end - start
        variance[row, col] = np.float32(np.var(values))
        cell_confidence[row, col] = np.float32(np.mean(weights[start:end]))

    observed = sample_count > 0
    result = HeightfieldObservation(
        height=height,
        observed_mask=observed,
        sample_count=sample_count,
        variance=variance,
        confidence=cell_confidence,
        xy_bounds=bounds,
    )
    result.validate()
    return result


def complete_heightfield(
    observed_height: np.ndarray,
    observed_mask: np.ndarray,
    *,
    smoothing_iterations: int = 64,
) -> CompletedHeightfield:
    """Fill unobserved cells while preserving all observed samples exactly."""

    source = np.asarray(observed_height, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    if source.ndim != 2 or source.shape != observed.shape:
        raise ContractError("observed height and mask must be matching 2D arrays")
    if not observed.any():
        raise ContractError("at least one observed cell is required")
    if not np.isfinite(source[observed]).all():
        raise ContractError("observed cells must be finite")
    if smoothing_iterations < 0:
        raise ContractError("smoothing_iterations cannot be negative")

    height = np.where(observed, source, np.nan)
    known = observed.copy()
    while not known.all():
        neighbor_sum, neighbor_count = _neighbor_sum_count(height, known)
        frontier = (~known) & (neighbor_count > 0)
        if not frontier.any():
            raise ContractError("heightfield completion could not reach all cells")
        height[frontier] = neighbor_sum[frontier] / neighbor_count[frontier]
        known[frontier] = True

    for _ in range(smoothing_iterations):
        neighbor_sum, neighbor_count = _neighbor_sum_count(height, np.ones_like(observed))
        candidate = neighbor_sum / np.maximum(neighbor_count, 1)
        height[~observed] = candidate[~observed]
        height[observed] = source[observed]

    result = CompletedHeightfield(
        height=height.astype(np.float32),
        observed_mask=observed.copy(),
        generated_mask=~observed,
    )
    result.validate()
    return result


def slope_degrees(height: np.ndarray, *, cell_size: float) -> np.ndarray:
    values = np.asarray(height, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("height must be a finite 2D array")
    if cell_size <= 0:
        raise ContractError("cell_size must be positive")
    dy, dx = np.gradient(values, cell_size, cell_size)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


def _neighbor_sum_count(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape, dtype=np.float64)
    count = np.zeros(values.shape, dtype=np.int16)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        source_rows = slice(max(0, -dr), values.shape[0] - max(0, dr))
        source_cols = slice(max(0, -dc), values.shape[1] - max(0, dc))
        target_rows = slice(max(0, dr), values.shape[0] - max(0, -dr))
        target_cols = slice(max(0, dc), values.shape[1] - max(0, -dc))
        source_valid = valid[source_rows, source_cols]
        total[target_rows, target_cols] += np.where(
            source_valid, values[source_rows, source_cols], 0.0
        )
        count[target_rows, target_cols] += source_valid
    return total, count
