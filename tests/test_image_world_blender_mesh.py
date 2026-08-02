import numpy as np
import pytest

from lowvram3d.image_world.blender_mesh import build_terrain_mesh_data
from lowvram3d.image_world.contracts import ContractError


def test_mesh_builder_preserves_partition_and_dimensions():
    height = np.arange(25, dtype=np.float32).reshape(5, 5)
    observed = np.zeros((5, 5), dtype=bool)
    observed[:, :3] = True
    generated = ~observed
    confidence = np.where(observed, 0.8, 0.0).astype(np.float32)

    mesh = build_terrain_mesh_data(
        height,
        observed,
        generated,
        confidence,
        horizontal_size=100.0,
        vertical_scale=20.0,
        maximum_resolution=5,
    )
    assert mesh.vertices.shape == (25, 3)
    assert mesh.faces.shape == (16, 4)
    assert mesh.observed.sum() == 15
    assert mesh.generated.sum() == 10
    assert mesh.vertices[:, 0].min() == pytest.approx(-50.0)
    assert mesh.vertices[:, 0].max() == pytest.approx(50.0)
    assert mesh.vertices[:, 2].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 2].max() == pytest.approx(20.0)


def test_mesh_builder_downsamples_deterministically():
    height = np.arange(81, dtype=np.float32).reshape(9, 9)
    observed = np.ones((9, 9), dtype=bool)
    generated = np.zeros((9, 9), dtype=bool)
    confidence = np.ones((9, 9), dtype=np.float32)
    first = build_terrain_mesh_data(height, observed, generated, confidence, maximum_resolution=5)
    second = build_terrain_mesh_data(height, observed, generated, confidence, maximum_resolution=5)
    assert first.mesh_rows == 5
    assert first.mesh_cols == 5
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.faces, second.faces)


def test_mesh_builder_rejects_invalid_partition():
    height = np.zeros((4, 4), dtype=np.float32)
    observed = np.ones((4, 4), dtype=bool)
    generated = np.ones((4, 4), dtype=bool)
    confidence = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ContractError):
        build_terrain_mesh_data(height, observed, generated, confidence)


def test_mesh_builder_rejects_nonfinite_height():
    height = np.zeros((4, 4), dtype=np.float32)
    height[1, 1] = np.nan
    observed = np.ones((4, 4), dtype=bool)
    generated = np.zeros((4, 4), dtype=bool)
    confidence = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ContractError):
        build_terrain_mesh_data(height, observed, generated, confidence)
