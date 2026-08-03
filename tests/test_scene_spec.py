from __future__ import annotations

import copy
import json
from pathlib import Path

from lowvram3d.scene_spec import validate_scene_spec, validate_scene_spec_file


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "configs" / "scene" / "castlegrounds_scene_spec_v1.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_castlegrounds_scene_spec_is_valid() -> None:
    report = validate_scene_spec_file(FIXTURE)
    assert report["scene_spec_valid"] is True
    assert report["classification"] == "PROVEN"
    assert report["error_count"] == 0
    assert report["counts"]["assets"] == 4
    assert report["counts"]["pcg_layers"] == 4


def test_duplicate_asset_id_is_rejected() -> None:
    spec = _fixture()
    spec["assets"].append(copy.deepcopy(spec["assets"][0]))
    report = validate_scene_spec(spec)
    assert report["scene_spec_valid"] is False
    assert any(error["code"] == "DUPLICATE_ID" for error in report["errors"])


def test_gpu_population_cannot_require_collision() -> None:
    spec = _fixture()
    spec["populations"][0]["requires_collision"] = True
    report = validate_scene_spec(spec)
    assert any(error["code"] == "UNSAFE_GPU_PCG" for error in report["errors"])


def test_gameplay_proxy_requires_collision() -> None:
    spec = _fixture()
    spec["assets"][1]["collision"] = "none"
    report = validate_scene_spec(spec)
    assert any(error["code"] == "UNSAFE_GAMEPLAY_PROXY" for error in report["errors"])


def test_missing_population_asset_reference_is_rejected() -> None:
    spec = _fixture()
    spec["populations"][0]["asset_refs"] = ["missing_asset"]
    report = validate_scene_spec(spec)
    assert any(error["code"] == "MISSING_REFERENCE" for error in report["errors"])


def test_invalid_camera_range_is_rejected() -> None:
    spec = _fixture()
    spec["camera"]["source_camera"]["near_m"] = 10
    spec["camera"]["source_camera"]["far_m"] = 5
    report = validate_scene_spec(spec)
    assert any(error["code"] == "INVALID_CAMERA_RANGE" for error in report["errors"])


def test_gpu_layer_cannot_emit_gameplay_geometry() -> None:
    spec = _fixture()
    spec["pcg"]["layers"][-1]["outputs"] = ["gameplay_collision"]
    report = validate_scene_spec(spec)
    assert any(error["code"] == "GPU_GAMEPLAY_OUTPUT" for error in report["errors"])


def test_gpu_concurrency_must_be_disabled() -> None:
    spec = _fixture()
    spec["budgets"]["allow_concurrent_neural_and_unreal_gpu"] = True
    report = validate_scene_spec(spec)
    assert any(error["code"] == "UNSAFE_GPU_CONCURRENCY" for error in report["errors"])
