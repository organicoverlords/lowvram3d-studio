"""Data-driven builder registry and selection."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BuilderContract, estimate_budget


BUILDER_REGISTRY: dict[str, BuilderContract] = {
    "terrain": BuilderContract("terrain", ("ground", "terrain", "cliff"), ("terrain", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "blocking", "walkable", "rock_or_ground", ("geometry", "collision"), "visual_shell", {"triangles": 100000}),
    "architecture": BuilderContract("architecture", ("architecture", "interior_structure", "wall", "roof", "tower", "opening"), ("modular_architecture", "editable_mesh"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking_by_region", "walkable_openings", "local_architecture", ("geometry", "collision"), "conservative_blocker", {"triangles": 200000}),
    "water": BuilderContract("water", ("water",), ("water_surface", "spline_structure"), ("regions", "visibility"), ("StaticMeshActor",), "none", "excluded", "water", ("visibility", "exclusion"), "visual_shell", {"triangles": 20000}),
    "crossing": BuilderContract("crossing", ("crossing", "road_or_path"), ("spline_structure", "procedural_mesh", "gameplay_proxy"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking", "walkable", "wood_or_local", ("geometry", "collision", "navigation"), "unresolved", {"triangles": 50000}),
    "vegetation": BuilderContract("vegetation", ("vegetation",), ("procedural_population", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "none", "ignored", "biome_local", ("population", "exclusion"), "visual_shell", {"instances": 1000}),
    "hero_object": BuilderContract("hero_object", ("hero_object",), ("editable_mesh", "visual_shell"), ("regions", "scene_graph"), ("StaticMeshActor",), "local", "local", "object_local", ("geometry", "material"), "visual_shell", {"triangles": 100000}),
    "background": BuilderContract("background", ("background_geometry",), ("background_card", "visual_shell"), ("regions", "visibility"), ("StaticMeshActor",), "none", "ignored", "background", ("coverage",), "visual_shell", {"triangles": 100000}),
    "environment": BuilderContract("environment", ("sky_or_ceiling", "lighting_source"), ("sky_environment", "visual_shell"), ("regions", "camera"), ("SkyAtmosphere", "DirectionalLight", "ExponentialHeightFog"), "none", "ignored", "environment", ("environment", "lighting"), "flat_environment", {"draw_calls": 10}),
    "gameplay_boundary": BuilderContract("gameplay_boundary", ("gameplay_boundary",), ("gameplay_proxy", "collision_only", "navigation_only"), ("regions", "scene_graph"), ("BlockingVolume",), "blocking", "blocked", "debug", ("collision",), "bounded_box", {"actors": 32}),
}


def _region_class(region: Mapping[str, Any]) -> str:
    raw = str(region.get("semantic_class") or region.get("layer_type") or region.get("layer") or "unknown").lower()
    if raw == "unknown":
        tags = {str(tag).lower() for tag in region.get("tags", [])}
        text = " ".join((str(region.get("id", "")), str(region.get("label", "")), str(region.get("depth_band", "")), str(region.get("representation", "")), *tags)).lower()
        if "water" in text:
            raw = "water"
        elif "vegetation" in text or "grass" in text or "tree" in text or "decorative" in text:
            raw = "vegetation"
        elif "castle" in text or "architecture" in text or "hero_structure" in text:
            raw = "architecture"
        elif "sky" in text or "cloud" in text or "background" in text:
            raw = "sky_or_ceiling"
        elif "shell" in text:
            raw = "background_geometry"
    aliases = {
        "castle": "architecture",
        "bridge_module": "crossing",
        "bridge": "crossing",
        "grass": "vegetation",
        "tree": "vegetation",
        "forest": "vegetation",
        "source_visible_shell": "background_geometry",
        "sky": "sky_or_ceiling",
    }
    return aliases.get(raw, raw)


def select_builders(spec: Mapping[str, Any], representation_manifest: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for index, region in enumerate(spec.get("regions", [])):
        region_id = str(region.get("id", f"region_{index + 1:03d}"))
        semantic = _region_class(region)
        for builder_id, contract in BUILDER_REGISTRY.items():
            if semantic not in contract.semantic_classes:
                continue
            selected.setdefault(builder_id, {"region_ids": [], "reason": "SceneSpec semantics and representation"})["region_ids"].append(region_id)
    return selected


def builder_manifest(spec: Mapping[str, Any], representation_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = select_builders(spec, representation_manifest)
    records = {}
    for builder_id, details in sorted(selected.items()):
        contract = BUILDER_REGISTRY[builder_id]
        records[builder_id] = {"contract": contract.to_dict(), **details, "estimated_budget": estimate_budget(contract, len(details["region_ids"]))}
    return {"schema_version": "builder_manifest_v2", "classification": "PROVEN", "selected": records, "selection_source": "SceneSpec semantic classes and representation manifest", "filename_not_used_for_selection": True, "available_builder_count": len(BUILDER_REGISTRY)}
