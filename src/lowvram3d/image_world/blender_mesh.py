"""Pure terrain-mesh preparation used by the Blender proof builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractError


@dataclass(frozen=True)
class TerrainMeshData:
    vertices: np.ndarray
    faces: np.ndarray
    observed: np.ndarray
    generated: np.ndarray
    confidence: np.ndarray
    source_rows: int
    source_cols: int
    mesh_rows: int
    mesh_cols: int
    horizontal_size: float
    vertical_scale: float
    minimum_height: float
    maximum_height: float

    def validate(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ContractError("vertices must have shape Nx3")
        if self.faces.ndim != 2 or self.faces.shape[1] != 4:
            raise ContractError("faces must have shape Mx4")
        count = self.vertices.shape[0]
        for name in ("observed", "generated", "confidence"):
            if getattr(self, name).shape != (count,):
                raise ContractError(f"{name} must have one value per vertex")
        if not np.isfinite(self.vertices).all():
            raise ContractError("vertices must be finite")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= count):
            raise ContractError("faces reference invalid vertices")
        if np.any(self.observed & self.generated):
            raise ContractError("observed and generated flags cannot overlap")
        if not np.all(self.observed | self.generated):
            raise ContractError("every terrain vertex must be observed or generated")
        if ((self.confidence < 0.0) | (self.confidence > 1.0)).any():
            raise ContractError("confidence must be in [0, 1]")


def build_terrain_mesh_data(
    height: np.ndarray,
    observed_mask: np.ndarray,
    generated_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    horizontal_size: float = 1000.0,
    vertical_scale: float = 250.0,
    maximum_resolution: int = 257,
) -> TerrainMeshData:
    """Create a centered quad grid while preserving proof attributes.

    Height is normalized only for diagnostic visualization. The original arrays
    remain the source of truth and are referenced in the Blender report.
    """

    values = np.asarray(height, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    generated = np.asarray(generated_mask, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ContractError("height must be a 2D array of at least 2x2")
    if observed.shape != values.shape or generated.shape != values.shape or conf.shape != values.shape:
        raise ContractError("terrain arrays must share a shape")
    if not np.isfinite(values).all() or not np.isfinite(conf).all():
        raise ContractError("height and confidence must be finite")
    if horizontal_size <= 0.0 or vertical_scale <= 0.0:
        raise ContractError("terrain scales must be positive")
    if maximum_resolution < 2:
        raise ContractError("maximum_resolution must be at least two")
    if np.any(observed & generated) or not np.all(observed | generated):
        raise ContractError("observed/generated masks must partition the terrain")
    if ((conf < 0.0) | (conf > 1.0)).any():
        raise ContractError("confidence must be in [0, 1]")

    row_index = _sample_indices(values.shape[0], maximum_resolution)
    col_index = _sample_indices(values.shape[1], maximum_resolution)
    sampled_height = values[np.ix_(row_index, col_index)]
    sampled_observed = observed[np.ix_(row_index, col_index)]
    sampled_generated = generated[np.ix_(row_index, col_index)]
    sampled_confidence = conf[np.ix_(row_index, col_index)]

    minimum = float(sampled_height.min())
    maximum = float(sampled_height.max())
    span = maximum - minimum
    normalized = np.zeros_like(sampled_height) if span <= 1e-12 else (sampled_height - minimum) / span

    rows, cols = sampled_height.shape
    x = np.linspace(-horizontal_size / 2.0, horizontal_size / 2.0, cols)
    y = np.linspace(-horizontal_size / 2.0, horizontal_size / 2.0, rows)
    grid_x, grid_y = np.meshgrid(x, y)
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), (normalized * vertical_scale).ravel()))

    faces = np.empty(((rows - 1) * (cols - 1), 4), dtype=np.int32)
    cursor = 0
    for row in range(rows - 1):
        base = row * cols
        for col in range(cols - 1):
            a = base + col
            faces[cursor] = (a, a + 1, a + cols + 1, a + cols)
            cursor += 1

    result = TerrainMeshData(
        vertices=vertices.astype(np.float32),
        faces=faces,
        observed=sampled_observed.ravel(),
        generated=sampled_generated.ravel(),
        confidence=sampled_confidence.astype(np.float32).ravel(),
        source_rows=int(values.shape[0]),
        source_cols=int(values.shape[1]),
        mesh_rows=int(rows),
        mesh_cols=int(cols),
        horizontal_size=float(horizontal_size),
        vertical_scale=float(vertical_scale),
        minimum_height=minimum,
        maximum_height=maximum,
    )
    result.validate()
    return result


def _sample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, size - 1, maximum)).astype(np.int64))
