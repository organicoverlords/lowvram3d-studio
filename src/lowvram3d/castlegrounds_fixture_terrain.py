"""Deterministic modular terrain and landform plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def build_terrain_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    landmarks = {item["id"]: item for item in spec.get("landmarks", [])}
    castle = landmarks.get("castle_base", {}).get("world_m", [10.0, 28.0, 5.0])
    cliff = landmarks.get("foreground_cliff", {}).get("world_m", [-22.0, 31.0, 16.0])
    return {
        "schema_version": "scene_terrain_plan_v1",
        "classification": "PROVEN",
        "scene_id": spec["scene_id"],
        "units": "meters",
        "seed": int(spec["intent"]["deterministic_seed"]),
        "features": [
            {"id": "playable_castle_island", "region_id": "castle_core", "primitive": "modular_box_cluster", "center_m": [float(castle[0]), float(castle[1]), 2.5], "size_m": [24.0, 20.0, 5.0], "collision": "blocking", "navigation": "walkable"},
            {"id": "foreground_cliff_mass", "region_id": "foreground_cliff", "primitive": "modular_rock_cluster", "center_m": [float(cliff[0]), float(cliff[1]), 8.0], "size_m": [18.0, 14.0, 16.0], "collision": "blocking", "navigation": "walkable_boundary"},
            {"id": "secondary_island_west", "region_id": "foreground_cliff", "primitive": "modular_rock_cluster", "center_m": [-34.0, 49.0, 3.0], "size_m": [12.0, 10.0, 6.0], "collision": "blocking", "navigation": "ignored"},
            {"id": "secondary_island_east", "region_id": "foreground_cliff", "primitive": "modular_rock_cluster", "center_m": [34.0, 58.0, 3.0], "size_m": [14.0, 11.0, 6.0], "collision": "blocking", "navigation": "ignored"},
            {"id": "playable_boundary_north", "region_id": "castle_core", "primitive": "boundary_box", "center_m": [10.0, 72.0, 3.0], "size_m": [70.0, 2.0, 6.0], "collision": "blocking", "navigation": "blocked"},
        ],
        "source_camera_silhouette_policy": "bounded_modular_landforms; no source-shell edits",
        "water_exclusion": "river_main plus 3m radius",
    }


def write_terrain_plan(spec_path: str | Path, output: str | Path) -> Path:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    plan = build_terrain_plan(spec)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
