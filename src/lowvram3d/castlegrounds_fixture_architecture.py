"""Compatibility projection of the generic architecture instruction builder."""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path

from .builders.architecture import build_instructions
from .scene_composition import load_scene_overrides


def build_architecture_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2] / "scenes"
    overrides = load_scene_overrides(str(spec.get("scene_id", "")), root)
    override_map = {str(item["target_region_id"]): dict(item.get("corrected_value", {})) for item in overrides.get("overrides", [])}
    instructions = build_instructions(spec, overrides=override_map)
    components = []
    for actor in instructions["actors"]:
        transform = actor["world_transform"]
        components.append({"id": actor["actor_id"].removeprefix("architecture_"), "kind": actor["geometry_parameters"]["primitive"], "region_id": actor["semantic_region_id"], "center_m": transform["location_m"], "size_m": transform["scale_m"], "material": actor["material_class"], "collision": actor["collision_policy"]})
    return {"schema_version": "scene_architecture_plan_v1", "classification": "PROVEN", "scene_id": spec["scene_id"], "components": components, "unseen_surface_policy": "conservative proxy outside source camera; manifest-owned"}
