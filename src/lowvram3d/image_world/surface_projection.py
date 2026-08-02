"""Project MoGe camera-space observations into an auditable Z-up baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractError
from .hydrology import d8_flow_direction, flow_accumulation, priority_flood_fill, stream_mask
from .semantic_masks import SemanticMaskSet
from .terrain import (
    CompletedHeightfield,
    HeightfieldObservation,
    complete_heightfield,
    rasterize_point_map,
    slope_degrees,
)
from .world_frame import (
    WorldFrameEstimate,
    estimate_world_up_from_normals,
    transform_camera_vectors,
)


@dataclass(frozen=True)
class SurfaceProjectionResult:
    frame: WorldFrameEstimate
    observation: HeightfieldObservation
    completed: CompletedHeightfield
    hydrology_height: np.ndarray
    slope: np.ndarray
    flow_direction: np.ndarray
    flow_accumulation: np.ndarray
    stream_mask: np.ndarray
    candidate_mask: np.ndarray
    alignment: np.ndarray
    semantic_mask_applied: bool
    semantic_terrain_fraction: float | None
    cell_size: float
    stream_minimum_cells: float
    classification: str

    def validate(self) -> None:
        self.observation.validate()
        self.completed.validate()
        shape = self.completed.height.shape
        for name in (
            "hydrology_height",
            "slope",
            "flow_direction",
            "flow_accumulation",
            "stream_mask",
        ):
            if getattr(self, name).shape != shape:
                raise ContractError(f"{name} shape must match completed heightfield")
        if self.candidate_mask.shape != self.alignment.shape:
            raise ContractError("candidate mask and alignment must share image-space shape")
        if not np.isfinite(self.hydrology_height).all():
            raise ContractError("hydrology height must be finite")
        if self.cell_size <= 0.0:
            raise ContractError("cell_size must be positive")
        if self.semantic_mask_applied and self.semantic_terrain_fraction is None:
            raise ContractError("semantic terrain fraction is required when a semantic mask is applied")
        if not self.semantic_mask_applied and self.classification != "UNCLASSIFIED_SURFACE_BASELINE_NOT_TERRAIN_PROOF":
            raise ContractError("unclassified projection cannot be labelled terrain-safe")


def project_moge_surface(
    point_map: np.ndarray,
    normal_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    semantic_masks: SemanticMaskSet | None = None,
    grid_size: int = 513,
    minimum_surface_alignment: float = 0.15,
    smoothing_iterations: int = 32,
    stream_minimum_cells: float | None = None,
    allow_up_fallback: bool = False,
) -> SurfaceProjectionResult:
    """Create a deterministic top-down baseline from MoGe arrays.

    When ``semantic_masks`` is omitted, the result is deliberately classified
    as unpromotable.  When supplied, only pixels proven as terrain candidates
    may enter rasterization; unresolved and non-terrain pixels are excluded.
    """

    points = np.asarray(point_map, dtype=np.float64)
    normals = np.asarray(normal_map, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ContractError("point_map must have shape HxWx3")
    if normals.shape != points.shape or valid.shape != points.shape[:2]:
        raise ContractError("normal_map and valid_mask must match point_map")
    if not 0.0 <= minimum_surface_alignment <= 1.0:
        raise ContractError("minimum_surface_alignment must be in [0, 1]")

    semantic_terrain_fraction: float | None = None
    if semantic_masks is not None:
        semantic_masks.validate()
        if semantic_masks.terrain_candidate.shape != valid.shape:
            raise ContractError("semantic mask shape must match MoGe image-space maps")
        semantic_gate = semantic_masks.terrain_candidate
        semantic_terrain_fraction = float(semantic_gate.mean())
    else:
        semantic_gate = np.ones(valid.shape, dtype=bool)

    frame = estimate_world_up_from_normals(
        normals,
        valid & semantic_gate,
        allow_fallback=allow_up_fallback,
    )
    rotation = np.asarray(frame.rotation_camera_to_world, dtype=np.float64)
    world_points = transform_camera_vectors(points, rotation)
    world_normals = transform_camera_vectors(normals, rotation)

    normal_lengths = np.linalg.norm(world_normals, axis=-1)
    finite = np.isfinite(world_points).all(axis=-1) & np.isfinite(world_normals).all(axis=-1)
    normalized = np.zeros_like(world_normals)
    good_normals = finite & (normal_lengths > 1e-8)
    normalized[good_normals] = world_normals[good_normals] / normal_lengths[good_normals, None]
    alignment = np.zeros(valid.shape, dtype=np.float32)
    alignment[good_normals] = np.abs(normalized[good_normals, 2]).astype(np.float32)
    candidate = valid & semantic_gate & finite & (alignment >= minimum_surface_alignment)
    if not candidate.any():
        raise ContractError("no terrain surface candidates remain after semantic and alignment filtering")

    bounds = robust_xy_bounds(world_points, candidate)
    confidence = np.where(candidate, alignment, 0.0).astype(np.float32)
    observation = rasterize_point_map(
        world_points,
        candidate,
        confidence=confidence,
        grid_size=grid_size,
        xy_bounds=bounds,
        minimum_confidence=max(0.001, minimum_surface_alignment),
    )
    completed = complete_heightfield(
        observation.height,
        observation.observed_mask,
        smoothing_iterations=smoothing_iterations,
    )

    xmin, xmax, ymin, ymax = observation.xy_bounds
    cell_size = max(xmax - xmin, ymax - ymin) / max(grid_size - 1, 1)
    slope = slope_degrees(completed.height, cell_size=cell_size)
    hydrology_height = priority_flood_fill(completed.height)
    direction = d8_flow_direction(hydrology_height, cell_size=cell_size)
    accumulation = flow_accumulation(direction)
    threshold = (
        float(stream_minimum_cells)
        if stream_minimum_cells is not None
        else max(32.0, float(grid_size * grid_size) * 0.002)
    )
    streams = stream_mask(accumulation, minimum_cells=threshold)

    result = SurfaceProjectionResult(
        frame=frame,
        observation=observation,
        completed=completed,
        hydrology_height=hydrology_height,
        slope=slope,
        flow_direction=direction,
        flow_accumulation=accumulation,
        stream_mask=streams,
        candidate_mask=candidate,
        alignment=alignment,
        semantic_mask_applied=semantic_masks is not None,
        semantic_terrain_fraction=semantic_terrain_fraction,
        cell_size=float(cell_size),
        stream_minimum_cells=threshold,
        classification=(
            "SEMANTICALLY_FILTERED_TERRAIN_BASELINE_NOT_SOURCE_QUALITY_PROOF"
            if semantic_masks is not None
            else "UNCLASSIFIED_SURFACE_BASELINE_NOT_TERRAIN_PROOF"
        ),
    )
    result.validate()
    return result


def robust_xy_bounds(
    world_points: np.ndarray,
    mask: np.ndarray,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    padding_fraction: float = 0.01,
) -> tuple[float, float, float, float]:
    points = np.asarray(world_points, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:2]:
        raise ContractError("world_points must be HxWx3 and mask must be HxW")
    selected = points[valid]
    selected = selected[np.isfinite(selected).all(axis=-1)]
    if selected.shape[0] < 4:
        raise ContractError("at least four finite points are required for XY bounds")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ContractError("invalid robust-bound percentiles")
    if padding_fraction < 0.0:
        raise ContractError("padding_fraction cannot be negative")

    xmin, ymin = np.percentile(selected[:, :2], lower_percentile, axis=0)
    xmax, ymax = np.percentile(selected[:, :2], upper_percentile, axis=0)
    xspan = max(float(xmax - xmin), 1e-6)
    yspan = max(float(ymax - ymin), 1e-6)
    pad = padding_fraction * max(xspan, yspan)
    return (
        float(xmin - pad),
        float(xmax + pad),
        float(ymin - pad),
        float(ymax + pad),
    )
