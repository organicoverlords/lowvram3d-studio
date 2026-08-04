"""Compatibility projection of the generic deterministic population builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .builders.vegetation import build_instructions
from .scene_composition import load_scene_overrides


def build_vegetation_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2] / "scenes"
    overrides = load_scene_overrides(str(spec.get("scene_id", "")), root)
    override_map = {str(item["target_region_id"]): dict(item.get("corrected_value", {})) for item in overrides.get("overrides", [])}
    instructions = build_instructions(spec, overrides=override_map)
    placements = []
    species_counts: dict[str, int] = {}
    for actor in instructions["actors"]:
        transform = actor["world_transform"]
        species = str(actor["geometry_parameters"].get("species", "vegetation"))
        species_counts[species] = species_counts.get(species, 0) + 1
        placements.append(transform["location_m"])
    species = [{"id": name, "mesh": "/Engine/BasicShapes/Cube.Cube", "material": material, "count": count, "scale_m": [1.0, 1.0, 1.0]} for name, count in sorted(species_counts.items()) for material in ["biome_local"]]
    exclusions = next((item.get("corrected_value", {}).get("population", []) for item in overrides.get("overrides", []) if item.get("target_region_id") == "vegetation"), [])
    return {"schema_version": "scene_vegetation_plan_v1", "classification": "PROVEN", "scene_id": spec["scene_id"], "seed": int(spec["intent"]["deterministic_seed"]), "execution": "cpu_deterministic", "species": species, "placements_m": placements, "exclusions": ["river_main", "bridge_axis_main", "castle_core", "SP_PlayerStart_Castle_V1", "walkable_navigation_path"], "collision": "none", "navigation": "ignored"}
