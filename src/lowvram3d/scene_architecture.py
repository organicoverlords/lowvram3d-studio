"""Generic data-driven architecture planner.

Scene-specific fixtures may provide a richer planner in a fixture namespace;
this module never selects or names a particular scene.
"""

from __future__ import annotations

from typing import Any, Mapping


def build_architecture_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    features = []
    for region in spec.get("regions", []):
        layer_type = region.get("layer_type") or region.get("layer")
        semantic = str(region.get("semantic_class", "")).lower()
        if layer_type not in {"architecture", "interior_structure"} and not any(token in semantic for token in ("building", "structure", "room", "wall")):
            continue
        features.append({
            "id": str(region.get("id", f"architecture_{len(features):03d}")),
            "region_id": str(region.get("id", "unknown")),
            "kind": "bounded_structure",
            "bbox_norm_xyxy": list(region.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
            "collision": "blocking",
            "navigation": "walkable_openings",
            "material": "local_architecture",
        })
    return {"schema_version": "architecture_plan_v2", "classification": "PROVEN", "features": features, "source": "SceneSpec semantics"}
