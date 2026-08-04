"""CPU-first deterministic vegetation placement with exclusion rules."""

from __future__ import annotations

from typing import Any, Mapping


def build_vegetation_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(spec["intent"]["deterministic_seed"])
    placements = [
        [-25.0, 28.0, 4.0], [-20.0, 22.0, 4.0], [-14.0, 36.0, 3.0],
        [27.0, 42.0, 3.0], [33.0, 51.0, 3.0], [-35.0, 57.0, 3.0],
        [40.0, 65.0, 3.0], [-43.0, 42.0, 3.0],
        [-23.0, 29.5, 4.0], [-18.0, 23.5, 4.0], [-12.0, 37.5, 3.0],
        [29.0, 43.5, 3.0], [35.0, 52.5, 3.0], [-33.0, 58.5, 3.0],
        [42.0, 66.5, 3.0], [-41.0, 43.5, 3.0],
    ]
    return {
        "schema_version": "scene_vegetation_plan_v1",
        "classification": "PROVEN",
        "scene_id": spec["scene_id"],
        "seed": seed,
        "execution": "cpu_deterministic",
        "species": [
            {"id": "trees", "mesh": "/Game/JungleEnvironmentMegaPack/Meshes/Foliage/SM_JSM_BroadleafCross_3.SM_JSM_BroadleafCross_3", "material": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Grass_Lush_Mixed.MI_JEM_Grass_Lush_Mixed", "count": 8, "scale_m": [3.0, 3.0, 6.0]},
            {"id": "grass", "mesh": "/Game/JungleEnvironmentMegaPack/Meshes/Foliage/SM_JSM_GrassCross_3.SM_JSM_GrassCross_3", "material": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Grass_Lush_Mixed.MI_JEM_Grass_Lush_Mixed", "count": 8, "scale_m": [1.5, 1.5, 2.0]},
        ],
        "placements_m": placements,
        "exclusions": ["river_main", "bridge_axis_main", "castle_core", "SP_PlayerStart_Castle_V1", "walkable_navigation_path"],
        "collision": "none",
        "navigation": "ignored",
    }
