import numpy as np
import pytest

from lowvram3d.image_world.contracts import ContractError
from lowvram3d.image_world.unreal_landscape import (
    decode_unreal_heightmap,
    encode_unreal_heightmap,
    encode_weightmap,
    landscape_xy_scale_cm,
)


def test_unreal_height_round_trip_within_quantization():
    axis = np.linspace(-20.0, 80.0, 513, dtype=np.float64)
    height = np.tile(axis[:, None], (1, 513))
    encoded = encode_unreal_heightmap(height)
    decoded = decode_unreal_heightmap(
        encoded.encoded,
        landscape_z_scale=encoded.landscape_z_scale,
        actor_z_cm=encoded.actor_z_cm,
    )
    error = np.abs(decoded - height)
    assert error.max() <= encoded.quantization_step_m * 0.51
    assert encoded.encoded.dtype == np.uint16


def test_flat_heightfield_uses_minimum_span_and_finite_scale():
    height = np.full((513, 513), 12.0, dtype=np.float32)
    encoded = encode_unreal_heightmap(height)
    assert encoded.landscape_z_scale > 0.0
    assert np.unique(encoded.encoded).tolist() == [32768]
    assert encoded.actor_z_cm == pytest.approx(1200.0)


def test_unsupported_landscape_size_fails_closed():
    with pytest.raises(ContractError):
        encode_unreal_heightmap(np.zeros((512, 512), dtype=np.float32))


def test_weightmap_encoding_bounds():
    values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    encoded = encode_weightmap(values)
    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [[0, 128, 255]]
    with pytest.raises(ContractError):
        encode_weightmap(np.array([[1.1]], dtype=np.float32))


def test_xy_scale_matches_world_width():
    scale = landscape_xy_scale_cm(4096.0, 513)
    assert scale == pytest.approx(800.0)
