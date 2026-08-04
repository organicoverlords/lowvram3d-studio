from __future__ import annotations

import json
from pathlib import Path

from lowvram3d.castlegrounds_fixture_vegetation import build_vegetation_plan


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return json.loads((ROOT / "configs/scene/castlegrounds_scene_spec_v1.json").read_text(encoding="utf-8"))


def test_vegetation_plan_is_cpu_deterministic_and_excluded() -> None:
    plan = build_vegetation_plan(_spec())
    assert plan == build_vegetation_plan(_spec())
    assert plan["execution"] == "cpu_deterministic"
    assert "river_main" in plan["exclusions"]
    assert "bridge_axis_main" in plan["exclusions"]
    assert "castle_core" in plan["exclusions"]
    assert len(plan["placements_m"]) == sum(species["count"] for species in plan["species"])
