"""Deterministic visual previews for MoGe geometry artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MogePreviewSummary:
    height: int
    width: int
    valid_fraction: float
    depth_low: float
    depth_high: float
    previews: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def depth_preview(depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return an 8-bit near-white/far-dark depth preview and robust limits."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth and mask must be matching two-dimensional arrays")
    values = depth[valid]
    if values.size == 0:
        raise ValueError("depth preview has no finite valid pixels")

    low, high = (float(value) for value in np.percentile(values, [2.0, 98.0]))
    if not high > low:
        low = float(values.min())
        high = float(values.max())
    if not high > low:
        high = low + 1.0

    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    preview[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return preview, low, high


def normal_preview(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 normal preview using the conventional [-1, 1] mapping."""
    normal = np.asarray(normal, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if normal.ndim != 3 or normal.shape[-1] != 3 or valid.shape != normal.shape[:2]:
        raise ValueError("normal must be HxWx3 and mask must be HxW")
    finite = np.isfinite(normal).all(axis=-1)
    valid = valid & finite
    preview = np.zeros(normal.shape, dtype=np.uint8)
    mapped = np.rint(np.clip((normal + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
    preview[valid] = mapped[valid]
    return preview


def save_moge_previews(
    geometry_dir: str | Path,
    output_dir: str | Path,
    *,
    cv2_module: Any,
) -> MogePreviewSummary:
    """Load non-pickle MoGe arrays and write compact PNG proof images."""
    geometry = Path(geometry_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    depth = np.load(geometry / "depth.npy", allow_pickle=False)
    mask = np.load(geometry / "mask.npy", allow_pickle=False).astype(bool)
    depth_image, low, high = depth_preview(depth, mask)

    previews = ["depth.png", "mask.png"]
    if not cv2_module.imwrite(str(output / "depth.png"), depth_image):
        raise RuntimeError("OpenCV failed to write depth preview")
    mask_image = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2_module.imwrite(str(output / "mask.png"), mask_image):
        raise RuntimeError("OpenCV failed to write mask preview")

    normal_path = geometry / "normal.npy"
    if normal_path.is_file():
        normal = np.load(normal_path, allow_pickle=False)
        normal_rgb = normal_preview(normal, mask)
        normal_bgr = cv2_module.cvtColor(normal_rgb, cv2_module.COLOR_RGB2BGR)
        if not cv2_module.imwrite(str(output / "normal.png"), normal_bgr):
            raise RuntimeError("OpenCV failed to write normal preview")
        previews.append("normal.png")

    summary = MogePreviewSummary(
        height=int(depth.shape[0]),
        width=int(depth.shape[1]),
        valid_fraction=float(mask.mean()),
        depth_low=low,
        depth_high=high,
        previews=tuple(previews),
    )
    (output / "preview-summary.json").write_text(summary.to_json(), encoding="utf-8")
    return summary
