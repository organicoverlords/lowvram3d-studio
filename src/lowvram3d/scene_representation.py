"""Generic representation selection from scene semantics and budgets."""

from __future__ import annotations

from typing import Any, Mapping


REPRESENTATIONS = ("visual_shell", "editable_mesh", "procedural_mesh", "modular_architecture", "terrain", "gameplay_proxy", "water_surface", "spline_structure", "procedural_population", "background_card", "sky_environment", "collision_only", "navigation_only", "unresolved")


def choose_representation(region: Mapping[str, Any], visibility: Mapping[str, Any] | None = None, budgets: Mapping[str, Any] | None = None) -> list[str]:
    visibility = visibility or {}
    semantic = str(region.get("semantic_class") or region.get("layer_type") or "unknown").lower()
    confidence = float(region.get("confidence", 0.5))
    interactive = bool(region.get("interactive") or region.get("walkable"))
    source_important = bool(region.get("source_camera_important", True))
    representations: list[str] = []
    if region.get("representation") in REPRESENTATIONS:
        representations.append(str(region["representation"]))
    if semantic in {"water"}:
        representations.append("water_surface")
        if interactive:
            representations.append("collision_only")
    elif semantic in {"crossing", "road_or_path"}:
        representations.extend(("spline_structure", "gameplay_proxy") if interactive else ("procedural_mesh",))
    elif semantic in {"vegetation"}:
        representations.append("procedural_population")
    elif semantic in {"terrain", "ground", "cliff", "ground_surface"}:
        representations.append("terrain")
    elif semantic in {"architecture", "interior_structure", "wall", "roof", "tower", "opening"}:
        representations.append("modular_architecture")
        if interactive:
            representations.append("gameplay_proxy")
    elif semantic in {"sky_or_ceiling", "lighting_source"}:
        representations.append("sky_environment")
    elif semantic in {"hero_object"}:
        representations.append("editable_mesh" if confidence >= 0.65 else "visual_shell")
    elif semantic in {"background_geometry"}:
        representations.append("background_card")
    if source_important and confidence < 0.4:
        representations.append("visual_shell")
    if not representations:
        representations.append("unresolved")
    result = []
    for item in representations:
        if item not in result:
            result.append(item)
    return result


def build_representation_manifest(spec: Mapping[str, Any], visibility_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    visibility = {str(record.get("region_id")): record for record in (visibility_manifest or {}).get("records", [])}
    records = []
    for index, region in enumerate(spec.get("regions", [])):
        region_id = str(region.get("id", f"region_{index + 1:03d}"))
        choices = choose_representation(region, visibility.get(region_id), spec.get("budgets", {}))
        records.append({"region_id": region_id, "semantic_class": str(region.get("semantic_class") or region.get("layer_type") or "unknown"), "representations": choices, "selected": choices[0], "confidence": float(region.get("confidence", 0.5)), "source_camera_important": bool(region.get("source_camera_important", True)), "materially_visible": choices[0] != "unresolved", "fallback": "visual_shell" if "visual_shell" in choices else "unresolved"})
    return {"schema_version": "representation_manifest_v1", "classification": "PROVEN", "records": records, "selection_policy": "semantic class, confidence, visibility, gameplay role, camera importance, and budgets"}
