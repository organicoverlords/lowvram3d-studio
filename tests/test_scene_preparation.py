from __future__ import annotations

import copy
import json
from pathlib import Path

from lowvram3d.scene_preparation import REQUIRED_COLLECTIONS, build_scene_preparation_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "configs" / "scene" / "castlegrounds_scene_spec_v1.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_preparation_plan_classifies_assets_and_splines() -> None:
    plan = build_scene_preparation_plan(_fixture())
    assert plan["classification"] == "PROVEN"
    assert plan["collections"] == REQUIRED_COLLECTIONS
    assert plan["counts"] == {
        "asset_tasks": 4,
        "import_tasks": 1,
        "placeholder_tasks": 3,
        "spline_tasks": 2,
    }
    assert plan["gpu_work_required"] is False
    assert plan["neural_work_required"] is False
    assert plan["geometry_generation_required"] is False

    by_id = {task["asset_id"]: task for task in plan["asset_tasks"]}
    assert by_id["source_mesh_v2"]["collection"] == "SCENE_VISUAL_SHELL"
    assert by_id["source_mesh_v2"]["action"] == "import_gltf"
    assert by_id["castle_proxy"]["collection"] == "SCENE_GAMEPLAY_PROXY"
    assert by_id["castle_proxy"]["action"] == "placeholder"
    assert by_id["bridge_tile"]["collection"] == "SCENE_PROCEDURAL_MODULES"


def test_visual_shell_cannot_gain_gameplay_policy() -> None:
    spec = _fixture()
    spec["assets"][0]["collision"] = "simple"
    plan = build_scene_preparation_plan(spec)
    assert plan["classification"] == "REJECTED"
    assert any(
        error["code"] == "UNSAFE_VISUAL_SHELL_GAMEPLAY_POLICY"
        for error in plan["errors"]
    )


def test_gpu_population_cannot_reference_collision_asset() -> None:
    spec = _fixture()
    spec["assets"][3]["collision"] = "simple"
    plan = build_scene_preparation_plan(spec)
    assert plan["classification"] == "REJECTED"
    assert any(
        error["code"] == "GPU_POPULATION_REFERENCES_GAMEPLAY_ASSET"
        for error in plan["errors"]
    )


def test_unsupported_import_format_is_rejected() -> None:
    spec = _fixture()
    spec = copy.deepcopy(spec)
    spec["assets"][0]["source_uri"] = "C:/bad/source.obj"
    plan = build_scene_preparation_plan(spec)
    assert plan["classification"] == "REJECTED"
    assert any(
        error["code"] == "UNSUPPORTED_SCENE_ASSET_FORMAT"
        for error in plan["errors"]
    )
