"""Compatibility projection of the scene-data terrain builder.

The production source of truth is the generic builder plus the explicit scene
override file.  This module preserves the older evidence-plan shape for
existing callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .builders.terrain import build_instructions
from .scene_composition import load_scene_overrides


def _override_map(spec: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2] / "scenes"
    value = load_scene_overrides(str(spec.get("scene_id", "")), root)
    return {str(item["target_region_id"]): dict(item.get("corrected_value", {})) for item in value.get("overrides", [])}


def build_terrain_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    instructions = build_instructions(spec, overrides=_override_map(spec))
    features = []
    landmarks = {item["id"]: item for item in spec.get("landmarks", [])}
    for actor in instructions["actors"]:
        transform = actor["world_transform"]
        prefix = f"terrain_{actor['semantic_region_id']}_"
        feature = {"id": actor["actor_id"][len(prefix):] if actor["actor_id"].startswith(prefix) else actor["actor_id"].removeprefix("terrain_"), "region_id": actor["semantic_region_id"], "primitive": actor["geometry_parameters"]["primitive"], "center_m": transform["location_m"], "size_m": transform["scale_m"], "collision": actor["collision_policy"], "navigation": actor["navigation_policy"]}
        features.append(feature)
    anchor = landmarks.get("castle_base", {}).get("world_m")
    if anchor:
        for feature in features:
            if feature["id"] == "playable_castle_island":
                feature["center_m"][:2] = [float(anchor[0]), float(anchor[1])]
    return {"schema_version": "scene_terrain_plan_v1", "classification": "PROVEN", "scene_id": spec["scene_id"], "units": "meters", "seed": int(spec["intent"]["deterministic_seed"]), "features": features, "source_camera_silhouette_policy": "bounded_modular_landforms; no source-shell edits", "water_exclusion": "explicit water spline exclusion"}


def write_terrain_plan(spec_path: str | Path, output: str | Path) -> Path:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_terrain_plan(spec), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
