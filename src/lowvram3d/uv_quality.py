"""UV candidate metrics and lexicographic preset selection.

Selection is deliberately lexicographic rather than a weighted score: a weighted score hides which
property actually decided the outcome, and these properties are not commensurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class UvPreset:
    name: str
    max_cost: float
    max_iterations: int


@dataclass
class UvCandidateMetrics:
    preset: str
    chart_count: int
    atlas_utilization: float
    atlas_count: int
    atlas_width: int
    atlas_height: int
    overlap_pair_count: int
    overlap_texel_area: float
    degenerate_triangle_count: int
    out_of_bounds_triangle_count: int
    stretch_p95: float
    tiny_chart_surface_percent: float
    runtime_seconds: float
    valid: bool
    errors: list[str] = field(default_factory=list)
    max_cost: float = 0.0

    def as_dict(self) -> dict:
        return {
            "preset": self.preset,
            "max_cost": self.max_cost,
            "chart_count": self.chart_count,
            "atlas_utilization": self.atlas_utilization,
            "atlas_count": self.atlas_count,
            "atlas_width": self.atlas_width,
            "atlas_height": self.atlas_height,
            "overlap_pair_count": self.overlap_pair_count,
            "overlap_texel_area": self.overlap_texel_area,
            "degenerate_triangle_count": self.degenerate_triangle_count,
            "out_of_bounds_triangle_count": self.out_of_bounds_triangle_count,
            "stretch_p95": self.stretch_p95,
            "tiny_chart_surface_percent": self.tiny_chart_surface_percent,
            "runtime_seconds": self.runtime_seconds,
            "valid": self.valid,
            "errors": list(self.errors),
        }


PRESETS = (
    UvPreset("A", 2.0, 2),
    UvPreset("B", 4.0, 2),
    UvPreset("C", 8.0, 2),
)

MAX_CHART_COUNT = 1_250
MIN_ATLAS_UTILIZATION = 0.55
MAX_STRETCH_P95 = 8.0
MAX_TINY_CHART_SURFACE_PERCENT = 1.0


def select_candidate(candidates: list[UvCandidateMetrics]) -> UvCandidateMetrics | None:
    """Lexicographic: chart count, then utilization, then stretch, then tiny-chart area, then
    max_cost as the final tie-breaker so the least aggressive configuration wins a true tie."""
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        return None
    return sorted(
        valid,
        key=lambda c: (
            c.chart_count,
            -c.atlas_utilization,
            c.stretch_p95,
            c.tiny_chart_surface_percent,
            c.max_cost,
        ),
    )[0]


def conformal_stretch(
    positions: np.ndarray,
    uv_triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Singular-value stretch per triangle, with 3D areas for area-weighted percentiles.

    Builds a 2D orthonormal basis in the triangle's own plane, forms the Jacobian of the map from
    that basis to UV, and returns sigma_max / sigma_min.
    """
    edge1 = positions[:, 1] - positions[:, 0]
    edge2 = positions[:, 2] - positions[:, 0]
    normal = np.cross(edge1, edge2)
    area3d = np.linalg.norm(normal, axis=1) * 0.5

    length1 = np.linalg.norm(edge1, axis=1)
    safe = np.maximum(length1, 1e-20)
    basis_x = edge1 / safe[:, None]
    projection = np.einsum("ij,ij->i", edge2, basis_x)
    perpendicular = edge2 - projection[:, None] * basis_x
    perpendicular_length = np.linalg.norm(perpendicular, axis=1)
    basis_y = perpendicular / np.maximum(perpendicular_length, 1e-20)[:, None]

    local = np.zeros((len(positions), 2, 2), np.float64)
    local[:, 0, 0] = np.einsum("ij,ij->i", edge1, basis_x)
    local[:, 1, 0] = np.einsum("ij,ij->i", edge1, basis_y)
    local[:, 0, 1] = np.einsum("ij,ij->i", edge2, basis_x)
    local[:, 1, 1] = np.einsum("ij,ij->i", edge2, basis_y)

    uv = np.zeros((len(uv_triangles), 2, 2), np.float64)
    uv[:, :, 0] = uv_triangles[:, 1] - uv_triangles[:, 0]
    uv[:, :, 1] = uv_triangles[:, 2] - uv_triangles[:, 0]

    stretch = np.full(len(positions), np.nan)
    determinant = local[:, 0, 0] * local[:, 1, 1] - local[:, 0, 1] * local[:, 1, 0]
    usable = np.abs(determinant) > 1e-20
    if usable.any():
        inverse = np.zeros_like(local[usable])
        d = determinant[usable]
        inverse[:, 0, 0] = local[usable][:, 1, 1] / d
        inverse[:, 1, 1] = local[usable][:, 0, 0] / d
        inverse[:, 0, 1] = -local[usable][:, 0, 1] / d
        inverse[:, 1, 0] = -local[usable][:, 1, 0] / d
        jacobian = np.einsum("nij,njk->nik", uv[usable], inverse)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        stretch[usable] = singular[:, 0] / np.maximum(singular[:, 1], 1e-12)
    return stretch, area3d


def area_weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    finite = np.isfinite(values) & (weights > 0)
    if not finite.any():
        return float("nan")
    order = np.argsort(values[finite])
    sorted_values = values[finite][order]
    sorted_weights = weights[finite][order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = cumulative[-1] * percentile / 100.0
    index = int(np.searchsorted(cumulative, cutoff))
    return float(sorted_values[min(index, len(sorted_values) - 1)])
