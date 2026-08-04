from __future__ import annotations

import copy

import pytest

from lowvram3d.scene_composition import build_scene_content_manifest, material_build_receipt, validate_scene_overrides


def _spec(scene_id: str = "demo_scene") -> dict:
    return {
        "schema_version": "scene_spec_v1",
        "scene_id": scene_id,
        "intent": {"deterministic_seed": 17},
        "source": {"sha256": "a" * 64},
        "camera": {"projection": "perspective"},
        "world_extent_m": [-20.0, -20.0, 20.0, 20.0],
        "regions": [
            {"id": "land", "layer_type": "terrain", "bbox_norm_xyxy": [0.1, 0.1, 0.9, 0.9], "evidence": ["analysis_bundle.land"]},
            {"id": "building", "layer_type": "architecture", "center_m": [0.0, 2.0, 2.0], "size_m": [4.0, 3.0, 4.0]},
            {"id": "plants", "layer_type": "vegetation", "center_m": [2.0, 1.0, 0.0], "size_m": [1.0, 1.0, 2.0]},
        ],
        "splines": [
            {"id": "stream_a", "kind": "river", "points_m": [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], "width_m": 2.0},
            {"id": "path_a", "kind": "crossing", "points_m": [[-2.0, 0.0, 1.0], [2.0, 0.0, 1.0]], "width_m": 1.5},
        ],
    }


def _manifest(spec: dict) -> dict:
    return build_scene_content_manifest(spec=spec, analysis_bundle={"schema_version": "analysis_bundle_v1"}, camera_contract=spec["camera"], semantic_masks={"regions": spec["regions"]}, depth_bands={}, contours={}, world_anchors={}, support_relationships={}, visibility={}, representation_manifest={}, material_plan={})


def test_composition_emits_all_required_actor_fields_and_is_deterministic() -> None:
    first = _manifest(_spec())
    second = _manifest(_spec())
    assert first == second
    assert first["classification"] == "PROVEN"
    assert {layer for layer in first["layers"] if first["layers"][layer]["actors"]} >= {"terrain", "architecture", "water", "crossing", "vegetation"}
    required = {"actor_id", "semantic_region_id", "builder_id", "builder_version", "source_evidence", "transform_derivation", "world_transform", "geometry_parameters", "material_class", "collision_policy", "navigation_policy", "deterministic_seed", "asset_path"}
    assert all(required <= set(actor) for actor in first["actors"])
    assert first["manual_only_actor_count"] == 0
    assert material_build_receipt(first)["classification"] == "PROVEN"


def test_scene_name_is_metadata_not_behavior() -> None:
    first = _manifest(_spec("alpha_scene"))
    second = _manifest(_spec("different_scene"))
    signature = lambda value: [(actor["actor_id"], actor["world_transform"], actor["material_class"]) for actor in value["actors"]]
    assert signature(first) == signature(second)


def test_override_schema_requires_explicit_reason_and_evidence() -> None:
    value = {"schema_version": "scene_overrides_v1", "scene_id": "demo_scene", "overrides": [{"override_id": "x", "target_region_id": "land", "original_inference": "uncertain", "corrected_value": {}, "reason": "r", "evidence": ["e"], "confidence": 0.5, "pipeline_stage": "scene_composition"}]}
    report = validate_scene_overrides(value, scene_id="demo_scene")
    assert report["classification"] == "PROVEN"
    invalid = copy.deepcopy(value)
    del invalid["overrides"][0]["evidence"]
    with pytest.raises(ValueError, match="missing evidence"):
        validate_scene_overrides(invalid, scene_id="demo_scene")


def test_manifest_rejects_duplicate_actor_ids_from_builder_data() -> None:
    spec = _spec()
    spec["regions"][0]["parts"] = [{"id": "same", "center_m": [0, 0, 0], "size_m": [1, 1, 1], "primitive": "box"}, {"id": "same", "center_m": [1, 0, 0], "size_m": [1, 1, 1], "primitive": "box"}]
    with pytest.raises(ValueError, match="duplicate actor_id"):
        _manifest(spec)
