"""Generic material-region analysis without image-specific assumptions."""

from __future__ import annotations

from typing import Any, Mapping


def build_material_regions(spec: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for index, region in enumerate(spec.get("regions", [])):
        semantic = str(region.get("semantic_class") or region.get("layer_type") or "unknown").lower()
        material = str(region.get("material") or region.get("material_policy") or "local_material")
        if material == "local_material":
            material = {"water": "water", "vegetation": "biome_local", "architecture": "local_architecture", "terrain": "rock_or_ground", "ground": "rock_or_ground", "crossing": "wood_or_local", "road_or_path": "path", "sky_or_ceiling": "environment"}.get(semantic, material)
        records.append({"region_id": str(region.get("id", f"region_{index + 1:03d}")), "semantic_class": semantic, "material_family": material, "observed": bool(region.get("observed_material", True)), "fallback": "material_local_neighbour_fill" if not region.get("observed_material", True) else "local_material"})
    return {"schema_version": "scene_material_analysis_v1", "classification": "PROVEN", "regions": records, "global_fallback": "neutral_local_material", "default_grey_allowed": False}
