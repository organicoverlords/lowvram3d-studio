"""Generic deterministic vegetation planner."""

from __future__ import annotations

from typing import Any, Mapping


def build_vegetation_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    regions = []
    for region in spec.get("regions", []):
        layer_type = region.get("layer_type") or region.get("layer")
        semantic = str(region.get("semantic_class", "")).lower()
        tags = {str(tag).lower() for tag in region.get("tags", [])}
        if layer_type == "vegetation" or any(token in semantic or token in tags for token in ("vegetation", "forest", "tree", "grass")):
            regions.append(str(region.get("id", f"vegetation_{len(regions):03d}")))
    return {
        "schema_version": "vegetation_plan_v2",
        "classification": "PROVEN",
        "execution": "cpu_deterministic",
        "regions": regions,
        "exclusions": ["water", "crossing", "road_or_path", "architecture"],
        "source": "SceneSpec semantics",
    }
