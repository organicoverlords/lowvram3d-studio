from __future__ import annotations

from lowvram3d.asset_strategy import build_asset_strategy
from lowvram3d.pipeline_graph import DEFAULT_STAGES, downstream_stages, validate_dag
from lowvram3d.pipeline_state import invalidate_from, new_state, reusable_stage, stage_input_hash
from lowvram3d.scene_graph import build_scene_graph, validate_scene_graph
from lowvram3d.scene_material_analysis import build_material_regions
from lowvram3d.scene_representation import build_representation_manifest
from lowvram3d.scene_visibility import build_visibility_manifest


def test_pipeline_dag_is_complete_and_acyclic():
    result = validate_dag()
    assert result["classification"] == "PROVEN"
    assert result["stage_count"] >= 48
    assert "cross_scene_regression" in {stage.stage_id for stage in DEFAULT_STAGES}


def test_targeted_invalidation_does_not_invalidate_unrelated_branch():
    state = new_state("demo_scene", "a" * 64, "/Game/GeneratedScenes/demo_scene/Maps/L_demo", "run-1")
    state["stages"] = {"source_image_audit": {"classification": "PROVEN"}, "final_packaging": {"classification": "PROVEN"}, "cross_scene_regression": {"classification": "PROVEN"}}
    invalidated = invalidate_from(state, "source_image_audit")
    assert "source_image_audit" in invalidated
    assert "cross_scene_regression" in invalidated


def test_stage_reuse_requires_hashes_and_versions():
    state = new_state("demo_scene", "a" * 64, "/Game/GeneratedScenes/demo_scene/Maps/L_demo", "run-1")
    stage = DEFAULT_STAGES[0]
    input_hash = stage_input_hash(stage, {"source": "a"})
    state["stages"][stage.stage_id] = {"classification": "PROVEN", "stage_version": stage.version, "input_hash": input_hash, "config_hash": "cfg"}
    assert reusable_stage(state, stage, input_hash, "cfg")
    assert not reusable_stage(state, stage, input_hash, "changed")


def test_generic_contracts_construct_scene_bundle():
    spec = {"scene_id": "demo_scene", "regions": [{"id": "wall", "layer_type": "architecture", "representation": "editable_mesh", "confidence": 0.8}, {"id": "ground", "layer_type": "ground_surface", "walkable": True}, {"id": "pond", "layer_type": "water"}, {"id": "trees", "layer_type": "vegetation"}]}
    visibility = build_visibility_manifest(spec)
    representation = build_representation_manifest(spec, visibility)
    graph = build_scene_graph(spec)
    assert validate_scene_graph(graph)["classification"] == "PROVEN"
    assert build_material_regions(spec)["classification"] == "PROVEN"
    assert build_asset_strategy(spec, representation)["classification"] == "PROVEN"
    assert "modular_architecture" in representation["records"][0]["representations"]
    assert {record["selected"] for record in representation["records"]} >= {"editable_mesh", "terrain", "water_surface", "procedural_population"}
