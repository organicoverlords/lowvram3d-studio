"""Generic data-driven ground-surface planner."""

from __future__ import annotations

from typing import Any, Mapping


def build_terrain_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    features = []
    for region in spec.get("regions", []):
        layer_type = region.get("layer_type") or region.get("layer")
        semantic = str(region.get("semantic_class", "")).lower()
        if layer_type not in {"terrain", "ground_surface"} and not any(token in semantic for token in ("ground", "terrain", "landscape", "cliff")):
            continue
        features.append({
            "id": str(region.get("id", f"ground_{len(features):03d}")),
            "region_id": str(region.get("id", "unknown")),
            "primitive": "bounded_ground_module",
            "bbox_norm_xyxy": list(region.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
            "collision": "blocking" if region.get("walkable", True) else "none",
            "navigation": "walkable" if region.get("walkable", True) else "ignored",
        })
    return {"schema_version": "ground_surface_plan_v2", "classification": "PROVEN", "features": features, "source": "SceneSpec semantics"}
