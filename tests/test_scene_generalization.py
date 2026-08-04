from __future__ import annotations

import json
from pathlib import Path

from lowvram3d.scene_registry import builder_manifest, select_builders
from lowvram3d.scene_paths import derive_scene_paths
from lowvram3d.scene_validation import validate_scene_plan


def _spec(regions):
    return {"scene_id": "fixture", "regions": regions, "landmarks": [], "depth_bands": [], "camera": {}, "assets": [], "intent": {}, "source": {}}


def test_outdoor_water_selects_relevant_builders_only():
    selected = select_builders(_spec([{"id": "water_001", "layer_type": "water", "representation": "gameplay_proxy"}, {"id": "architecture_001", "layer_type": "architecture", "representation": "editable_geometry"}]))
    assert set(selected) == {"architecture", "water"}
    assert "vegetation" not in selected


def test_natural_landscape_without_architecture():
    selected = select_builders(_spec([{"id": "ground_001", "layer_type": "ground_surface", "representation": "procedural_module"}, {"id": "vegetation_region_001", "layer_type": "vegetation", "representation": "procedural_population"}]))
    assert set(selected) == {"ground_surface", "vegetation"}
    assert "architecture" not in selected


def test_indoor_and_urban_select_different_capabilities():
    indoor = select_builders(_spec([{"id": "room_001", "layer_type": "interior_structure", "representation": "editable_geometry"}, {"id": "ceiling_001", "layer_type": "sky_or_ceiling", "representation": "sky"}]))
    urban = select_builders(_spec([{"id": "street_001", "layer_type": "road_or_path", "representation": "procedural_module"}, {"id": "building_001", "layer_type": "architecture", "representation": "editable_geometry"}]))
    assert set(indoor) == {"interior_structure", "sky_or_ceiling"}
    assert set(urban) == {"architecture", "road_or_path"}


def test_scene_local_paths_are_isolated():
    first = derive_scene_paths("alpha", "/Game/GeneratedScenes", "evidence/scenes/alpha/run-1")
    second = derive_scene_paths("beta", "/Game/GeneratedScenes", "evidence/scenes/beta/run-1")
    assert first["map"] != second["map"]
    assert first["evidence"] != second["evidence"]


def test_fixture_identifiers_do_not_enter_generic_registry_source():
    root = Path(__file__).resolve().parents[1] / "src" / "lowvram3d"
    generic = [root / name for name in ("scene_registry.py", "scene_paths.py", "scene_analysis.py", "scene_validation.py", "scene_completeness.py", "scene_architecture.py", "scene_terrain.py", "scene_vegetation.py", "image_to_scene_pipeline.py")]
    forbidden = ("Castlegrounds", "castle_proxy", "castle_core", "castle_base", "lighthouse_top", "river_main", "bridge_axis_main", "L_Castlegrounds_Hybrid_V1", "ImageToSceneSmoke_20260803")
    text = "\n".join(path.read_text(encoding="utf-8") for path in generic)
    assert not any(token in text for token in forbidden)


def test_budget_validation_is_data_driven():
    spec = _spec([{"id": "object_001", "layer_type": "hero_object", "representation": "editable_geometry"}])
    manifest = builder_manifest(spec)
    result = validate_scene_plan(spec, manifest)
    assert result["filename_not_used_for_selection"] is True
    assert result["gpu_work_requested"] is False


def test_object_image_is_not_selected_as_scene_builder():
    spec = _spec([{"id": "object_001", "layer_type": "hero_object", "representation": "editable_geometry"}])
    manifest = {
        "classification": "NOT_APPLICABLE",
        "selected": {},
        "selection_source": "explicit input_kind=object",
        "filename_not_used_for_selection": True,
    }
    result = validate_scene_plan(spec, manifest)
    assert result["selected_builder_count"] == 0
    assert result["gpu_work_requested"] is False
