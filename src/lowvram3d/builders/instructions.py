"""Shared deterministic primitives for scene builder instruction manifests.

The functions in this module deliberately know nothing about a particular
scene.  Scene-specific corrections are supplied as validated data by the
composition stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SEMANTIC_ALIASES = {
    "castle": "architecture",
    "building": "architecture",
    "wall": "architecture",
    "roof": "architecture",
    "tower": "architecture",
    "bridge": "crossing",
    "bridge_module": "crossing",
    "road": "road_or_path",
    "path": "road_or_path",
    "grass": "vegetation",
    "tree": "vegetation",
    "forest": "vegetation",
    "cliff": "terrain",
    "ground": "terrain",
    "ground_surface": "terrain",
    "sky": "sky_or_ceiling",
}


def semantic_class(region: Mapping[str, Any]) -> str:
    raw = str(region.get("semantic_class") or region.get("layer_type") or region.get("layer") or "unknown").lower()
    if raw == "unknown":
        text = " ".join(
            [str(region.get("id", "")), str(region.get("label", "")), str(region.get("representation", ""))]
            + [str(tag) for tag in region.get("tags", [])]
        ).lower()
        for candidate, needles in (("water", ("water", "river", "stream")), ("vegetation", ("vegetation", "tree", "grass", "forest")), ("crossing", ("bridge", "crossing")), ("architecture", ("building", "castle", "architecture", "tower")), ("sky_or_ceiling", ("sky", "cloud", "ceiling")), ("terrain", ("terrain", "cliff", "island", "ground"))):
            if any(needle in text for needle in needles):
                raw = candidate
                break
    return SEMANTIC_ALIASES.get(raw, raw)


def deterministic_seed(spec: Mapping[str, Any]) -> int:
    intent = spec.get("intent", {})
    try:
        return int(intent.get("deterministic_seed", 0))
    except (TypeError, ValueError):
        return 0


def _finite_vector(value: Any, name: str, length: int = 3) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _bounds_from_bbox(bbox: Sequence[Any], extent: Sequence[Any]) -> tuple[list[float], list[float]]:
    if len(bbox) != 4 or len(extent) != 4:
        raise ValueError("bbox and world extent must contain four values")
    x0, y0, x1, y1 = [float(item) for item in bbox]
    ex0, ey0, ex1, ey1 = [float(item) for item in extent]
    center = [ex0 + (ex1 - ex0) * ((x0 + x1) / 2.0), ey0 + (ey1 - ey0) * ((y0 + y1) / 2.0)]
    size = [max(0.1, abs(ex1 - ex0) * abs(x1 - x0)), max(0.1, abs(ey1 - ey0) * abs(y1 - y0))]
    return center, size


def region_parts(spec: Mapping[str, Any], region: Mapping[str, Any], overrides: Mapping[str, Any] | None = None, *, part_key: str = "parts") -> list[dict[str, Any]]:
    """Return geometry parts from explicit scene data or generic bounds."""

    region_id = str(region.get("id", "region"))
    override = (overrides or {}).get(region_id, {})
    parts = override.get(part_key) if isinstance(override, Mapping) else None
    if parts is None and part_key == "parts":
        parts = override.get("parts") if isinstance(override, Mapping) else None
    if parts is None:
        parts = region.get(part_key) or region.get("parts") or region.get("geometry_parts")
    if parts is not None:
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            raise ValueError(f"parts for {region_id} must be an array")
        return [dict(part) for part in parts]

    if "center_m" in region and "size_m" in region:
        return [{"id": region_id, "primitive": "box", "center_m": _finite_vector(region["center_m"], f"{region_id}.center_m"), "size_m": _finite_vector(region["size_m"], f"{region_id}.size_m")}]

    bbox = region.get("bbox_norm_xyxy")
    extent = spec.get("world_extent_m", [-50.0, -50.0, 50.0, 50.0])
    if bbox is not None:
        center_xy, size_xy = _bounds_from_bbox(bbox, extent)
        center_z = float(region.get("center_z_m", 0.0))
        height = max(0.5, float(region.get("height_m", max(1.0, size_xy[0] * 0.15))))
        return [{"id": region_id, "primitive": "box", "center_m": [center_xy[0], center_xy[1], center_z], "size_m": [size_xy[0], size_xy[1], height]}]

    # A region without spatial evidence gets a bounded placeholder at the
    # origin.  It remains explicit and auditable, never silently hand-placed.
    return [{"id": region_id, "primitive": "box", "center_m": [0.0, 0.0, 0.0], "size_m": [1.0, 1.0, 1.0], "spatial_evidence": "missing_region_bounds"}]


def actor_record(*, actor_id: str, region_id: str, builder_id: str, spec: Mapping[str, Any], part: Mapping[str, Any], material_class: str, collision_policy: str, navigation_policy: str, source_evidence: Any, transform_derivation: Any, geometry_parameters: Mapping[str, Any] | None = None, asset_path: str | None = None, semantic_class: str | None = None) -> dict[str, Any]:
    center = _finite_vector(part.get("center_m", [0.0, 0.0, 0.0]), f"{actor_id}.center_m")
    size = _finite_vector(part.get("size_m", [1.0, 1.0, 1.0]), f"{actor_id}.size_m")
    rotation = _finite_vector(part.get("rotation_deg", [0.0, 0.0, 0.0]), f"{actor_id}.rotation_deg")
    if any(value <= 0.0 for value in size):
        raise ValueError(f"{actor_id}.size_m must be positive")
    return {
        "actor_id": str(actor_id),
        "semantic_region_id": str(region_id),
        "semantic_class": semantic_class or str(part.get("semantic_class", "unknown")),
        "builder_id": str(builder_id),
        "builder_version": "1.0.0",
        "source_evidence": source_evidence if isinstance(source_evidence, list) else [str(source_evidence)],
        "transform_derivation": transform_derivation if isinstance(transform_derivation, list) else [str(transform_derivation)],
        "world_transform": {"location_m": center, "rotation_deg": rotation, "scale_m": size},
        "geometry_parameters": {"primitive": str(part.get("primitive", "box")), **dict(geometry_parameters or {})},
        "material_class": str(material_class),
        "collision_policy": str(collision_policy),
        "navigation_policy": str(navigation_policy),
        "deterministic_seed": deterministic_seed(spec),
        "asset_path": asset_path or "/Engine/BasicShapes/Cube.Cube",
        "manual_only": False,
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
