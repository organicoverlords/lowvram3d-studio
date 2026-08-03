from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lowvram3d.scene_gameplay_proxy import build_gameplay_proxy_plan


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "evidence" / "latest-scene-hybrid" / "authoritative_hybrid_scene_spec.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_plan_is_proven_and_uses_required_scene_ids() -> None:
    plan, receipt = build_gameplay_proxy_plan(_spec(), spec_path=SPEC_PATH)
    assert plan["classification"] == "PROVEN"
    assert receipt["classification"] == "PROVEN"
    assert plan["asset_id"] == "castle_proxy"
    assert plan["region_id"] == "castle_core"
    assert plan["landmark_id"] == "castle_base"
    assert plan["depth_band_id"] == "castle"
    assert plan["primitive_type"] == "engine_cube"
    assert plan["collision"] == "blocking/simple"
    assert plan["navigation"] == "walkable intent"
    assert plan["promotion"] is False
    assert plan["tags"] == ["gameplay_proxy", "replaceable", "unpromoted", "scene_spec_generated", "castle_proxy"]


def test_plan_is_deterministic_and_offsets_actor_above_base_anchor() -> None:
    first, first_receipt = build_gameplay_proxy_plan(_spec(), spec_path=SPEC_PATH)
    second, second_receipt = build_gameplay_proxy_plan(_spec(), spec_path=SPEC_PATH)
    assert first == second
    assert first_receipt == second_receipt
    assert first["actor_center_m"][0:2] == first["base_anchor_m"][0:2]
    assert first["actor_center_m"][2] == pytest.approx(
        first["base_anchor_m"][2] + first["dimensions_m"]["height_m"] / 2.0
    )
    assert first["actor_center_cm"] == pytest.approx([value * 100.0 for value in first["actor_center_m"]])


def test_invalid_spec_fails_closed() -> None:
    spec = _spec()
    spec["camera"]["source_camera"]["field_of_view_deg"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        build_gameplay_proxy_plan(spec, spec_path=SPEC_PATH)


def test_missing_asset_region_landmark_and_depth_band_fail_closed() -> None:
    for key, value in (("assets", "castle_proxy"), ("regions", "castle_core"), ("landmarks", "castle_base"), ("depth_bands", "castle")):
        spec = _spec()
        spec[key] = [item for item in spec[key] if item.get("id") != value]
        with pytest.raises(ValueError):
            build_gameplay_proxy_plan(spec, spec_path=SPEC_PATH)


def test_source_mesh_collision_and_promotion_fail_closed() -> None:
    spec = _spec()
    spec["assets"][1]["source_uri"] = "C:/should-not-be-collision.glb"
    with pytest.raises(ValueError, match="source mesh"):
        build_gameplay_proxy_plan(spec, spec_path=SPEC_PATH)

    promoted = copy.deepcopy(_spec())
    promoted["assets"][1]["tags"].append("promoted")
    with pytest.raises(ValueError, match="promoted"):
        build_gameplay_proxy_plan(promoted, spec_path=SPEC_PATH)


def test_malformed_bbox_exceeding_bounds_fails_closed() -> None:
    spec = _spec()
    spec["camera"]["source_camera"]["field_of_view_deg"] = 179.0
    with pytest.raises(ValueError, match="outside conservative bounds"):
        build_gameplay_proxy_plan(spec, spec_path=SPEC_PATH)
