"""Pure contracts and validation for the isolated MoGe hardware probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import ContractError


DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vits-normal"


@dataclass(frozen=True)
class MogeProbeSettings:
    model: str = DEFAULT_MOGE_MODEL
    num_tokens: int = 1200
    input_long_edge: int = 768
    fp16: bool = True
    max_gpu_memory_mb: int = 5600
    allow_download: bool = False
    allow_cpu: bool = False

    def validate(self) -> None:
        if not self.model.strip():
            raise ContractError("MoGe model name is required")
        if not 1200 <= self.num_tokens <= 2500:
            raise ContractError("MoGe num_tokens must be in the supported 1200..2500 range")
        if not 256 <= self.input_long_edge <= 2048:
            raise ContractError("MoGe input_long_edge must be between 256 and 2048")
        if self.max_gpu_memory_mb <= 0:
            raise ContractError("GPU memory ceiling must be positive")


@dataclass(frozen=True)
class MogeOutputSummary:
    height: int
    width: int
    valid_fraction: float
    finite_point_fraction: float
    finite_depth_fraction: float
    finite_normal_fraction: float


@dataclass(frozen=True)
class MogeProbeReport:
    status: str
    source_sha256: str
    settings: MogeProbeSettings
    output: MogeOutputSummary | None
    wall_time_seconds: float
    peak_gpu_allocated_mb: float | None
    peak_gpu_reserved_mb: float | None
    versions: Mapping[str, str]
    errors: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_moge_output(output: Mapping[str, Any]) -> MogeOutputSummary:
    required = {"points", "depth", "mask", "intrinsics"}
    missing = required - set(output)
    if missing:
        raise ContractError(f"MoGe output missing keys: {sorted(missing)}")

    points = _as_numpy(output["points"])
    depth = _as_numpy(output["depth"])
    mask = _as_numpy(output["mask"]).astype(bool)
    intrinsics = _as_numpy(output["intrinsics"])
    normal = None if output.get("normal") is None else _as_numpy(output["normal"])

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ContractError("MoGe points must have shape HxWx3")
    height, width = points.shape[:2]
    if depth.shape != (height, width) or mask.shape != (height, width):
        raise ContractError("MoGe depth and mask must match point-map dimensions")
    if normal is not None and normal.shape != points.shape:
        raise ContractError("MoGe normal map must match point-map shape")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ContractError("MoGe intrinsics must be a finite 3x3 matrix")
    if not mask.any():
        raise ContractError("MoGe valid-pixel mask is empty")

    valid = mask
    finite_points = np.isfinite(points).all(axis=-1)
    finite_depth = np.isfinite(depth)
    finite_normals = (
        np.ones((height, width), dtype=bool)
        if normal is None
        else np.isfinite(normal).all(axis=-1)
    )
    if not finite_points[valid].all():
        raise ContractError("MoGe point map contains non-finite values in valid pixels")
    if not finite_depth[valid].all():
        raise ContractError("MoGe depth contains non-finite values in valid pixels")
    if normal is not None and not finite_normals[valid].all():
        raise ContractError("MoGe normals contain non-finite values in valid pixels")

    return MogeOutputSummary(
        height=height,
        width=width,
        valid_fraction=float(valid.mean()),
        finite_point_fraction=float(finite_points.mean()),
        finite_depth_fraction=float(finite_depth.mean()),
        finite_normal_fraction=float(finite_normals.mean()),
    )


def save_moge_maps(output: Mapping[str, Any], output_dir: str | Path) -> MogeOutputSummary:
    summary = validate_moge_output(output)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "points.npy", _as_numpy(output["points"]).astype(np.float32), allow_pickle=False)
    np.save(root / "depth.npy", _as_numpy(output["depth"]).astype(np.float32), allow_pickle=False)
    np.save(root / "mask.npy", _as_numpy(output["mask"]).astype(np.uint8), allow_pickle=False)
    np.save(
        root / "intrinsics.npy",
        _as_numpy(output["intrinsics"]).astype(np.float64),
        allow_pickle=False,
    )
    if output.get("normal") is not None:
        np.save(
            root / "normal.npy",
            _as_numpy(output["normal"]).astype(np.float32),
            allow_pickle=False,
        )
    return summary


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)
