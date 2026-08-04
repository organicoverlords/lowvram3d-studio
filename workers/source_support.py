"""Conservative, explicit source-silhouette support for mesh components.

Cleanup and LOD stages must use the same source-facing coordinate contract.  Explicit callers pass
the source image/matte, reference mesh, and axes; the old manifest lookup remains only as a
backward-compatible fallback for post-LOD paths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.asset_profiles import foreground_mask
from mesh_io import read_glb


@dataclass(frozen=True)
class SupportContext:
    manifest_path: Path | None
    source_path: Path
    source_mask: np.ndarray
    source_bbox: tuple[int, int, int, int]
    right_axis: int
    up_axis: int
    front_axis: int | None
    allow_source_mirror: bool
    geometry_low: np.ndarray
    geometry_high: np.ndarray


def component_surface_separation(body_vertices: np.ndarray, component_vertices: np.ndarray) -> float:
    """Return nearest-surface distance using a bounded body-vertex index."""
    body = np.asarray(body_vertices, dtype=np.float64)
    component = np.asarray(component_vertices, dtype=np.float64)
    if not len(body) or not len(component):
        return float("inf")
    # A decimated nearest-neighbour index is sufficient for the conservative remove/keep gate and
    # avoids an O(body_vertices * component_vertices) distance matrix on million-face meshes.
    if len(body) > 250_000:
        step = int(np.ceil(len(body) / 250_000))
        body = body[::step]
    try:
        from scipy.spatial import cKDTree
        return float(cKDTree(body).query(component, workers=1)[0].min())
    except Exception:
        distances = np.linalg.norm(component[:, None, :] - body[None, :, :], axis=2)
        return float(distances.min())


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


def _axis(value: int | str | None, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"x", "y", "z"}:
            return "xyz".index(value)
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name} axis: {value!r}") from exc
    if value not in (0, 1, 2):
        raise ValueError(f"invalid {name} axis: {value!r}")
    return value


def build_support_context(
    source_path: Path,
    reference_positions: np.ndarray,
    *,
    manifest_path: Path | None = None,
    up_axis: int | str | None = None,
    front_axis: int | str | None = None,
    right_axis: int | str | None = None,
    allow_source_mirror: bool = False,
) -> SupportContext:
    """Build a source/geometry registration frame from explicit inputs."""
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    positions = np.asarray(reference_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("support reference mesh has no usable positions")

    fallback_right, fallback_up = _geometry_axes(positions)
    resolved_up = fallback_up if _axis(up_axis, name="up") is None else _axis(up_axis, name="up")
    resolved_right = (fallback_right if _axis(right_axis, name="right") is None
                      else _axis(right_axis, name="right"))
    resolved_front = _axis(front_axis, name="front")
    if resolved_up == resolved_right:
        raise ValueError("support up and right axes must differ")
    if resolved_front is not None and resolved_front in (resolved_up, resolved_right):
        raise ValueError("support front axis must differ from up and right axes")

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read support source image: {source_path}")
    mask = foreground_mask(image).astype(np.uint8)
    radius = max(2, int(round(max(mask.shape) * 0.010)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.dilate(mask, kernel) > 0
    source_bbox = _subject_bbox(mask)

    low = np.percentile(positions, 0.20, axis=0).astype(np.float64)
    high = np.percentile(positions, 99.80, axis=0).astype(np.float64)
    raw_low, raw_high = positions.min(axis=0), positions.max(axis=0)
    for axis in (resolved_right, resolved_up):
        span = float(raw_high[axis] - raw_low[axis])
        if float(high[axis] - low[axis]) < span * 0.70:
            low[axis], high[axis] = raw_low[axis], raw_high[axis]

    return SupportContext(
        manifest_path=manifest_path,
        source_path=source_path,
        source_mask=mask,
        source_bbox=source_bbox,
        right_axis=int(resolved_right),
        up_axis=int(resolved_up),
        front_axis=resolved_front,
        allow_source_mirror=bool(allow_source_mirror),
        geometry_low=low,
        geometry_high=high,
    )


def load_support_context(
    mesh_path: Path,
    positions: np.ndarray,
    *,
    source_image: Path | None = None,
    support_reference_mesh: Path | None = None,
    up_axis: int | str | None = None,
    front_axis: int | str | None = None,
    right_axis: int | str | None = None,
    allow_source_mirror: bool | None = None,
) -> SupportContext | None:
    """Load explicit support evidence, with the old lookup as a safe fallback."""
    mesh_path = Path(mesh_path)
    manifest_path = locate_manifest(mesh_path)
    explicit = any(value is not None for value in (
        source_image, support_reference_mesh, up_axis, front_axis, right_axis,
        allow_source_mirror,
    ))
    if not explicit and not is_post_lod_path(mesh_path):
        return None

    manifest: dict = {}
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    orientation = manifest.get("orientation") or {}
    if source_image is None:
        source_value = (manifest.get("source") or {}).get("path")
        source_image = Path(source_value) if source_value else None
    if source_image is None or not Path(source_image).is_file():
        return None

    reference_positions = np.asarray(positions, dtype=np.float64)
    if support_reference_mesh:
        reference_path = Path(support_reference_mesh)
        if not reference_path.is_file():
            return None
        reference_positions = read_glb(reference_path)[0].astype(np.float64)

    return build_support_context(
        Path(source_image), reference_positions,
        manifest_path=manifest_path,
        up_axis=up_axis if up_axis is not None else orientation.get("up_axis"),
        front_axis=front_axis if front_axis is not None else orientation.get("front_axis"),
        right_axis=right_axis if right_axis is not None else (
            orientation.get("right_axis", orientation.get("lateral_axis"))),
        allow_source_mirror=(bool(allow_source_mirror) if allow_source_mirror is not None else
                             bool(orientation.get("allow_source_mirror", False))),
    )


def component_support(
    context: SupportContext,
    positions: np.ndarray,
    triangles: np.ndarray,
    members: np.ndarray,
) -> dict:
    """Measure direct and optional mirrored source support for one component."""
    component_triangles = triangles[members]
    indices = np.unique(component_triangles)
    vertices = positions[indices].astype(np.float64)
    centroids = positions[component_triangles].mean(axis=1).astype(np.float64)
    samples = np.concatenate([vertices, centroids], axis=0)

    width_span = max(float(context.geometry_high[context.right_axis]
                           - context.geometry_low[context.right_axis]), 1e-9)
    height_span = max(float(context.geometry_high[context.up_axis]
                            - context.geometry_low[context.up_axis]), 1e-9)
    u = ((samples[:, context.right_axis] - context.geometry_low[context.right_axis])
         / width_span)
    v_up = ((samples[:, context.up_axis] - context.geometry_low[context.up_axis]) / height_span)

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
    mirrored, mirror_in_frame = score(x_mirror) if context.allow_source_mirror else (0.0, 0.0)
    return {
        "support": round(max(direct, mirrored), 6),
        "direct_support": round(direct, 6),
        "mirrored_support": round(mirrored, 6),
        "in_frame_fraction": round(max(direct_in_frame, mirror_in_frame), 6),
        "registration": "mirrored" if mirrored > direct else "direct",
        "allow_source_mirror": context.allow_source_mirror,
        "sample_count": int(len(samples)),
        "source_path": str(context.source_path),
        "manifest_path": str(context.manifest_path) if context.manifest_path else None,
    }


def component_position(context: SupportContext, vertices: np.ndarray) -> dict:
    height_span = max(float(context.geometry_high[context.up_axis]
                            - context.geometry_low[context.up_axis]), 1e-9)
    width_span = max(float(context.geometry_high[context.right_axis]
                           - context.geometry_low[context.right_axis]), 1e-9)
    height = float(((vertices[:, context.up_axis] - context.geometry_low[context.up_axis])
                    / height_span).mean())
    centre = float((context.geometry_low[context.right_axis]
                    + context.geometry_high[context.right_axis]) * 0.5)
    lateral = float(np.abs(vertices[:, context.right_axis] - centre).mean() / width_span)
    return {"height_mean": height, "lateral_mean": lateral}
