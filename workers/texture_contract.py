"""Small, fail-closed contracts shared by the CPU texture stages."""
from __future__ import annotations

import struct
from pathlib import Path


class AtlasResolutionContractError(RuntimeError):
    """Raised when one texture stage silently changes the negotiated atlas size."""


def validate_requested_atlas_size(size: int) -> int:
    value = int(size)
    if value < 512 or value > 4096 or value & (value - 1):
        raise AtlasResolutionContractError(
            f"ATLAS_RESOLUTION_CONTRACT_MISMATCH: invalid requested atlas size {value}"
        )
    return value


def assert_atlas_dimensions(actual: tuple[int, int] | list[int], expected: int, stage: str) -> None:
    expected = validate_requested_atlas_size(expected)
    dimensions = tuple(int(value) for value in actual)
    if dimensions != (expected, expected):
        raise AtlasResolutionContractError(
            "ATLAS_RESOLUTION_CONTRACT_MISMATCH: "
            f"{stage} produced {dimensions}, expected {(expected, expected)}"
        )


def png_dimensions(path: str | Path) -> tuple[int, int]:
    """Read PNG dimensions without decoding pixels or accepting another format."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AtlasResolutionContractError(f"{path} is not a valid PNG atlas")
    return struct.unpack(">II", data[16:24])
