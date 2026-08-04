from __future__ import annotations

import copy
import json
from pathlib import Path

from lowvram3d.castlegrounds_fixture_terrain import build_terrain_plan


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return json.loads((ROOT / "configs/scene/castlegrounds_scene_spec_v1.json").read_text(encoding="utf-8"))


def test_terrain_plan_is_deterministic_and_modular() -> None:
    first = build_terrain_plan(_spec())
    assert first == build_terrain_plan(_spec())
    assert len(first["features"]) >= 4
    assert all(feature["primitive"] != "one_giant_flat_cube" for feature in first["features"])
    assert any(feature["id"] == "foreground_cliff_mass" for feature in first["features"])


def test_terrain_plan_uses_authoritative_landmark() -> None:
    spec = _spec()
    spec["landmarks"][1]["world_m"] = [20.0, 30.0, 5.0]
    plan = build_terrain_plan(spec)
    island = next(feature for feature in plan["features"] if feature["id"] == "playable_castle_island")
    assert island["center_m"][:2] == [20.0, 30.0]
