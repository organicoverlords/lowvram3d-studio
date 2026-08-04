"""Low-detail replaceable castle and lighthouse architecture plan."""

from __future__ import annotations

from typing import Any, Mapping


def build_architecture_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    landmarks = {item["id"]: item for item in spec.get("landmarks", [])}
    base = landmarks.get("castle_base", {}).get("world_m", [10.0, 28.0, 5.0])
    light = landmarks.get("lighthouse_top", {}).get("world_m", [12.0, 25.0, 49.0])
    return {
        "schema_version": "scene_architecture_plan_v1",
        "classification": "PROVEN",
        "scene_id": spec["scene_id"],
        "components": [
            {"id": "castle_main_body", "kind": "box", "region_id": "castle_core", "center_m": [base[0], base[1], 12.0], "size_m": [16.0, 11.0, 14.0], "material": "stone"},
            {"id": "castle_west_wing", "kind": "box", "region_id": "castle_core", "center_m": [base[0] - 9.0, base[1] + 1.0, 8.0], "size_m": [5.0, 9.0, 8.0], "material": "stone"},
            {"id": "castle_east_wing", "kind": "box", "region_id": "castle_core", "center_m": [base[0] + 9.0, base[1] + 1.0, 8.0], "size_m": [5.0, 9.0, 8.0], "material": "stone"},
            {"id": "lighthouse_shaft", "kind": "cylinder", "region_id": "castle_core", "center_m": [light[0], light[1], 28.0], "size_m": [5.0, 5.0, 42.0], "material": "stone"},
            {"id": "lighthouse_cap", "kind": "cone", "region_id": "castle_core", "center_m": [light[0], light[1], 49.0], "size_m": [7.0, 7.0, 4.0], "material": "stone"},
            {"id": "castle_gate", "kind": "box", "region_id": "castle_core", "center_m": [base[0], base[1] - 5.7, 4.0], "size_m": [4.0, 0.8, 5.0], "material": "wood", "collision": "walkable_opening"},
        ],
        "unseen_surface_policy": "conservative_proxy_outside_source_camera",
    }
