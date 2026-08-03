from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lowvram3d.scene_spec import validate_scene_spec
from lowvram3d.scene_spec_legacy import build_scene_spec_from_legacy


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "proof" / "scene" / "20260803-image-to-scene-smoke"
SOURCE_MESH = REPO_ROOT / "configs" / "scene" / "castlegrounds_source_mesh_v2.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict]:
    return (
        _read(EVIDENCE / "scene_interpretation.json"),
        _read(EVIDENCE / "geometry_spec.json"),
        _read(EVIDENCE / "selected_asset.json"),
        _read(SOURCE_MESH),
    )


def test_legacy_migration_preserves_source_camera_landmarks_mesh_and_transform() -> None:
    interpretation, geometry, selected_asset, source_mesh = _inputs()
    spec, receipt = build_scene_spec_from_legacy(
        interpretation, geometry, selected_asset, source_mesh
    )

    assert receipt["classification"] == "PROVEN"
    assert receipt["scene_spec_valid"] is True
    assert receipt["source_sha256_preserved"] is True
    assert receipt["source_dimensions_preserved"] is True
    assert receipt["camera_contract_preserved"] is True
    assert receipt["landmark_contract_preserved"] is True
    assert receipt["mesh_identity_preserved"] is True
    assert receipt["transform_contract_preserved"] is True
    assert receipt["neural_work_started"] is False
    assert receipt["blender_work_started"] is False
    assert receipt["unreal_work_started"] is False

    camera = interpretation["camera_estimate"]
    migrated = spec["camera"]["source_camera"]
    assert migrated["projection"] == camera["projection"]
    assert migrated["field_of_view_deg"] == camera["field_of_view_deg"]
    assert migrated["principal_point_norm"] == camera["principal_point"]
    assert migrated["position_m"] == camera["camera_position"]
    assert migrated["look_at_m"] == camera["look_at"]
    assert migrated["near_m"] == camera["near_plane"]
    assert migrated["far_m"] == camera["far_plane"]

    assert spec["assets"][0]["source_uri"] == source_mesh["path"]
    assert spec["assets"][0]["triangle_budget"] == source_mesh["triangles"]
    assert spec["assets"][0]["transform"] == source_mesh["transform"]
    assert validate_scene_spec(spec)["scene_spec_valid"] is True


def test_legacy_migration_rejects_source_identity_mismatch() -> None:
    interpretation, geometry, selected_asset, source_mesh = _inputs()
    selected_asset = copy.deepcopy(selected_asset)
    selected_asset["source"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        build_scene_spec_from_legacy(
            interpretation, geometry, selected_asset, source_mesh
        )


def test_legacy_migration_rejects_source_dimensions_mismatch() -> None:
    interpretation, geometry, selected_asset, source_mesh = _inputs()
    selected_asset = copy.deepcopy(selected_asset)
    selected_asset["source"]["dimensions"] = [1, 1]

    with pytest.raises(ValueError, match="source dimensions mismatch"):
        build_scene_spec_from_legacy(
            interpretation, geometry, selected_asset, source_mesh
        )


def test_legacy_migration_rejects_invalid_mesh_contract() -> None:
    interpretation, geometry, selected_asset, source_mesh = _inputs()
    source_mesh = copy.deepcopy(source_mesh)
    source_mesh["schema_version"] = "wrong"

    with pytest.raises(ValueError, match="mesh contract schema mismatch"):
        build_scene_spec_from_legacy(
            interpretation, geometry, selected_asset, source_mesh
        )
