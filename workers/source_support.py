"""Conservative source-silhouette support for detached mesh components.

This module is intentionally used only for post-LOD debris decisions. A tiny component is not debris
just because decimation reduced it to one triangle: cords, leaves and pendants can legitimately do
that. The component must also be high/outboard and absent from the original source silhouette.

The source/mesh relation may be mirrored, so support is measured in both horizontal orientations
and the stronger score wins. The foreground mask is slightly dilated to fail in favour of keeping
ambiguous details.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.asset_profiles import foreground_mask


@dataclass(frozen=True)
class SupportContext:
    manifest_path: Path
    source_path: Path
    source_mask: np.ndarray
    source_bbox: tuple[int, int, int, int]
    width_axis: int
    height_axis: int
    geometry_low: np.ndarray
    geometry_high: np.ndarray


def is_post_lod_path(path: Path) -> bool:
    return any(part.upper() == "LOD" for part in Path(path).parts)


def locate_manifest(path: Path) -> Path | None:
    current = Path(path).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / "asset_manifest.json"
        if candidate.exists():
            return candidate
    return None


def _subject_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("source foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _geometry_axes(positions: np.ndarray) -> tuple[int, int]:
    extent = positions.max(axis=0) - positions.min(axis=0)
    height_axis = int(np.argmax(extent))
    remaining = [axis for axis in range(3) if axis != height_axis]
    width_axis = remaining[int(np.argmax(extent[remaining]))]
    return width_axis, height_axis


def load_support_context(mesh_path: Path, positions: np.ndarray) -> SupportContext | None:
    """Load a conservative source/geometry frame for post-LOD classification.

    Returns None outside a Pipeline V2 asset tree or when its source image is unavailable. Callers
    then retain the legacy policy rather than inventing source evidence.
    """
    mesh_path = Path(mesh_path)
    if not is_post_lod_path(mesh_path):
        return None
    manifest_path = locate_manifest(mesh_path)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        source_path = Path(manifest["source"]["path"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not source_path.exists():
        return None

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    mask = foreground_mask(image).astype(np.uint8)
    radius = max(2, int(round(max(mask.shape) * 0.010)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.dilate(mask, kernel) > 0
    source_bbox = _subject_bbox(mask)

    width_axis, height_axis = _geometry_axes(positions)
    low = np.percentile(positions, 0.20, axis=0).astype(np.float64)
    high = np.percentile(positions, 99.80, axis=0).astype(np.float64)
    raw_low, raw_high = positions.min(axis=0), positions.max(axis=0)
    for axis in (width_axis, height_axis):
        span = float(raw_high[axis] - raw_low[axis])
        if float(high[axis] - low[axis]) < span * 0.70:
            low[axis], high[axis] = raw_low[axis], raw_high[axis]

    return SupportContext(
        manifest_path=manifest_path,
        source_path=source_path,
        source_mask=mask,
        source_bbox=source_bbox,
        width_axis=width_axis,
        height_axis=height_axis,
        geometry_low=low,
        geometry_high=high,
    )


def component_support(
    context: SupportContext,
    positions: np.ndarray,
    triangles: np.ndarray,
    members: np.ndarray,
) -> dict:
    """Measure source support for one triangle component.

    Vertices plus face centroids are sampled. Horizontal mirroring is treated as equivalent because
    Mini Turbo may mirror a model relative to the illustration; vertical mirroring is never allowed.
    """
    component_triangles = triangles[members]
    indices = np.unique(component_triangles)
    vertices = positions[indices].astype(np.float64)
    centroids = positions[component_triangles].mean(axis=1).astype(np.float64)
    samples = np.concatenate([vertices, centroids], axis=0)

    width_span = max(
        float(context.geometry_high[context.width_axis] - context.geometry_low[context.width_axis]),
        1e-9,
    )
    height_span = max(
        float(context.geometry_high[context.height_axis] - context.geometry_low[context.height_axis]),
        1e-9,
    )
    u = (
        samples[:, context.width_axis] - context.geometry_low[context.width_axis]
    ) / width_span
    v_up = (
        samples[:, context.height_axis] - context.geometry_low[context.height_axis]
    ) / height_span

    x0, y0, x1, y1 = context.source_bbox
    width = max(x1 - x0 - 1, 1)
    height = max(y1 - y0 - 1, 1)
    y = np.rint(y0 + (1.0 - v_up) * height).astype(np.int64)
    x_direct = np.rint(x0 + u * width).astype(np.int64)
    x_mirror = np.rint(x0 + (1.0 - u) * width).astype(np.int64)
    mask = context.source_mask
    valid_y = (y >= 0) & (y < mask.shape[0])

    def score(x: np.ndarray) -> tuple[float, float]:
        valid = valid_y & (x >= 0) & (x < mask.shape[1])
        if not valid.any():
            return 0.0, 0.0
        supported = np.zeros(len(x), bool)
        supported[valid] = mask[y[valid], x[valid]]
        return float(supported.mean()), float(valid.mean())

    direct, direct_in_frame = score(x_direct)
    mirrored, mirror_in_frame = score(x_mirror)
    return {
        "support": round(max(direct, mirrored), 6),
        "direct_support": round(direct, 6),
        "mirrored_support": round(mirrored, 6),
        "in_frame_fraction": round(max(direct_in_frame, mirror_in_frame), 6),
        "sample_count": int(len(samples)),
        "source_path": str(context.source_path),
        "manifest_path": str(context.manifest_path),
    }


def component_position(context: SupportContext, vertices: np.ndarray) -> dict:
    height_span = max(
        float(context.geometry_high[context.height_axis] - context.geometry_low[context.height_axis]),
        1e-9,
    )
    width_span = max(
        float(context.geometry_high[context.width_axis] - context.geometry_low[context.width_axis]),
        1e-9,
    )
    height = float(
        ((vertices[:, context.height_axis] - context.geometry_low[context.height_axis]) / height_span).mean()
    )
    centre = float(
        (context.geometry_low[context.width_axis] + context.geometry_high[context.width_axis]) * 0.5
    )
    lateral = float(np.abs(vertices[:, context.width_axis] - centre).mean() / width_span)
    return {"height_mean": height, "lateral_mean": lateral}
