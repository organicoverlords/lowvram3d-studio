"""Pure conversion contracts for Unreal Landscape packages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractError


SUPPORTED_LANDSCAPE_SIZES = (513, 1009, 2017, 4033, 8129)


@dataclass(frozen=True)
class UnrealHeightEncoding:
    encoded: np.ndarray
    source_minimum_m: float
    source_maximum_m: float
    center_height_m: float
    actor_z_cm: float
    landscape_z_scale: float
    quantization_step_m: float

    def validate(self) -> None:
        if self.encoded.dtype != np.uint16 or self.encoded.ndim != 2:
            raise ContractError("encoded Unreal heightmap must be a uint16 2D array")
        if self.encoded.shape[0] != self.encoded.shape[1]:
            raise ContractError("encoded Unreal heightmap must be square")
        if self.encoded.shape[0] not in SUPPORTED_LANDSCAPE_SIZES:
            raise ContractError("encoded heightmap size is not a supported Landscape size")
        for value in (
            self.source_minimum_m,
            self.source_maximum_m,
            self.center_height_m,
            self.actor_z_cm,
            self.landscape_z_scale,
            self.quantization_step_m,
        ):
            if not np.isfinite(value):
                raise ContractError("height encoding metadata must be finite")
        if self.source_maximum_m < self.source_minimum_m:
            raise ContractError("height encoding range is invalid")
        if self.landscape_z_scale <= 0.0 or self.quantization_step_m <= 0.0:
            raise ContractError("Unreal scale and quantization must be positive")


def encode_unreal_heightmap(
    height_m: np.ndarray,
    *,
    minimum_vertical_span_m: float = 1.0,
    vertical_padding_fraction: float = 0.02,
) -> UnrealHeightEncoding:
    """Encode metre heights around the uint16 midpoint used by Landscape.

    Unreal interprets the signed height range as approximately -256..+256
    multiplied by Landscape Z Scale in centimetres. The actor Z location stores
    the centre elevation, avoiding loss of absolute height provenance.
    """

    height = np.asarray(height_m, dtype=np.float64)
    if height.ndim != 2 or height.shape[0] != height.shape[1]:
        raise ContractError("height_m must be a square 2D array")
    if height.shape[0] not in SUPPORTED_LANDSCAPE_SIZES:
        raise ContractError(
            f"heightmap size {height.shape[0]} is unsupported; expected one of {SUPPORTED_LANDSCAPE_SIZES}"
        )
    if not np.isfinite(height).all():
        raise ContractError("height_m must be finite")
    if minimum_vertical_span_m <= 0.0:
        raise ContractError("minimum_vertical_span_m must be positive")
    if not 0.0 <= vertical_padding_fraction <= 0.5:
        raise ContractError("vertical_padding_fraction must be in [0, 0.5]")

    source_minimum = float(height.min())
    source_maximum = float(height.max())
    center = 0.5 * (source_minimum + source_maximum)
    source_span = max(source_maximum - source_minimum, minimum_vertical_span_m)
    padded_span = source_span * (1.0 + 2.0 * vertical_padding_fraction)
    half_span = 0.5 * padded_span

    # Landscape local signed range is 512 units total. Z Scale is centimetres
    # per local unit, while actor Z stores the world-space centre elevation.
    z_scale = half_span * 100.0 / 256.0
    local_units = (height - center) * 100.0 / z_scale
    encoded_float = 32768.0 + local_units * 128.0
    encoded = np.rint(np.clip(encoded_float, 0.0, 65535.0)).astype(np.uint16)
    quantization_step_m = z_scale / (128.0 * 100.0)

    result = UnrealHeightEncoding(
        encoded=encoded,
        source_minimum_m=source_minimum,
        source_maximum_m=source_maximum,
        center_height_m=center,
        actor_z_cm=center * 100.0,
        landscape_z_scale=float(z_scale),
        quantization_step_m=float(quantization_step_m),
    )
    result.validate()
    return result


def decode_unreal_heightmap(
    encoded: np.ndarray,
    *,
    landscape_z_scale: float,
    actor_z_cm: float,
) -> np.ndarray:
    values = np.asarray(encoded)
    if values.dtype != np.uint16 or values.ndim != 2:
        raise ContractError("encoded must be a uint16 2D array")
    if landscape_z_scale <= 0.0 or not np.isfinite(actor_z_cm):
        raise ContractError("Unreal scale metadata is invalid")
    local_units = (values.astype(np.float64) - 32768.0) / 128.0
    return (local_units * landscape_z_scale + actor_z_cm) / 100.0


def encode_weightmap(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ContractError("weightmap values must be a finite 2D array")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ContractError("weightmap values must be in [0, 1]")
    return np.rint(array * 255.0).astype(np.uint8)


def landscape_xy_scale_cm(world_width_m: float, vertex_count: int) -> float:
    if world_width_m <= 0.0:
        raise ContractError("world_width_m must be positive")
    if vertex_count not in SUPPORTED_LANDSCAPE_SIZES:
        raise ContractError("vertex_count is not a supported Landscape size")
    return world_width_m * 100.0 / float(vertex_count - 1)
