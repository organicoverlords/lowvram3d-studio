"""Bounded asset strategy selection before geometry construction."""

from __future__ import annotations

from typing import Any, Mapping


STRATEGIES = ("reuse_project_asset", "reuse_engine_asset", "procedural_unreal_primitives", "procedural_unreal_mesh", "procedural_blender_mesh", "image_conditioned_generation", "approved_asset_library", "source_projection", "visual_shell_fallback")


def choose_strategy(region: Mapping[str, Any], representation: str) -> str:
    explicit = str(region.get("asset_strategy", ""))
    if explicit in STRATEGIES:
        return explicit
    if representation == "visual_shell":
        return "source_projection"
    semantic = str(region.get("semantic_class") or region.get("layer_type") or "unknown").lower()
    if semantic in {"water", "vegetation", "terrain", "ground", "ground_surface", "crossing", "road_or_path"}:
        return "procedural_unreal_mesh"
    if semantic in {"architecture", "interior_structure", "wall", "roof", "tower", "opening"}:
        return "procedural_unreal_mesh"
    if semantic == "hero_object" and float(region.get("confidence", 0.5)) >= 0.7:
        return "approved_asset_library"
    return "visual_shell_fallback"


def build_asset_strategy(spec: Mapping[str, Any], representation_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    representation_by_region = {str(item.get("region_id")): item for item in (representation_manifest or {}).get("records", [])}
    records = []
    for index, region in enumerate(spec.get("regions", [])):
        region_id = str(region.get("id", f"region_{index + 1:03d}"))
        representation = representation_by_region.get(region_id, {}).get("selected", region.get("representation", "unresolved"))
        strategy = choose_strategy(region, str(representation))
        records.append({
            "region_id": region_id,
            "candidate_assets": list(region.get("candidate_assets", [])),
            "chosen_strategy": strategy,
            "silhouette_suitability": "source_supported" if strategy in {"source_projection", "approved_asset_library"} else "bounded_low_detail",
            "scale_suitability": "scene_contract" if region.get("world_anchor") or region.get("bbox_norm_xyxy") else "uncertain",
            "material_compatibility": "local_material",
            "collision_suitability": "walkable" if region.get("walkable") else "noninteractive",
            "triangle_estimate": int(region.get("triangle_estimate", 10000)),
            "texture_memory_estimate_mb": int(region.get("texture_memory_estimate_mb", 8)),
            "reason": "semantic class, confidence, representation, and bounded fallback policy",
            "fallback": "visual_shell_fallback",
        })
    return {"schema_version": "asset_strategy_v1", "classification": "PROVEN", "records": records, "strategies": list(STRATEGIES), "asset_search_may_not_block_smoke": True}
