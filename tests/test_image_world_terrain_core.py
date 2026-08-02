import numpy as np

from lowvram3d.image_world.hydrology import (
    d8_flow_direction,
    flow_accumulation,
    priority_flood_fill,
    stream_mask,
)
from lowvram3d.image_world.terrain import (
    complete_heightfield,
    rasterize_point_map,
    slope_degrees,
)


def test_rasterization_uses_median_and_records_observation_evidence():
    points = np.array([
        [[0.0, 0.0, 1.0], [0.01, 0.01, 2.0]],
        [[0.02, 0.02, 100.0], [1.0, 1.0, 4.0]],
    ])
    mask = np.ones((2, 2), dtype=bool)
    result = rasterize_point_map(
        points,
        mask,
        grid_size=2,
        xy_bounds=(0, 1.1, 0, 1.1),
    )
    assert result.sample_count[0, 0] == 3
    assert result.height[0, 0] == 2.0
    assert result.variance[0, 0] > 0
    assert result.observed_mask.sum() == 2


def test_completion_preserves_observed_cells_and_fills_everything():
    height = np.full((5, 5), np.nan)
    mask = np.zeros((5, 5), dtype=bool)
    height[0, 0] = 0.0
    height[4, 4] = 10.0
    mask[0, 0] = True
    mask[4, 4] = True
    result = complete_heightfield(height, mask, smoothing_iterations=20)
    assert np.isfinite(result.height).all()
    assert result.height[0, 0] == 0.0
    assert result.height[4, 4] == 10.0
    assert result.generated_mask.sum() == 23


def test_priority_flood_removes_closed_pit_without_changing_boundary():
    height = np.array([
        [5, 5, 5, 5, 5],
        [5, 4, 4, 4, 5],
        [5, 4, 0, 4, 5],
        [5, 4, 4, 4, 5],
        [5, 5, 5, 5, 5],
    ], dtype=float)
    filled = priority_flood_fill(height)
    assert filled[2, 2] == 5.0
    assert np.array_equal(filled[[0, -1]], height[[0, -1]])


def test_d8_accumulation_reaches_single_low_outlet():
    height = np.add.outer(
        np.arange(5, 0, -1),
        np.arange(5, 0, -1),
    ).astype(float)
    direction = d8_flow_direction(height)
    accumulation = flow_accumulation(direction)
    assert accumulation[-1, -1] == 25
    assert stream_mask(accumulation, minimum_cells=10)[-1, -1]


def test_slope_degrees_reports_flat_and_sloped_surfaces():
    flat = np.zeros((3, 3), dtype=float)
    assert np.allclose(slope_degrees(flat, cell_size=1.0), 0.0)
    ramp = np.tile(np.arange(3, dtype=float), (3, 1))
    assert np.all(slope_degrees(ramp, cell_size=1.0) > 0.0)
