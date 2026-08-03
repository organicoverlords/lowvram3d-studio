from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lowvram3d.scene_hybrid import compose_authoritative_hybrid_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORED = REPO_ROOT / "configs" / "scene" / "castlegrounds_scene_spec_v1.json"
CAMERA = REPO_ROOT / "evidence" / "latest-scene-camera-local-worker" / "camera_contract.json"
BUILD = REPO_ROOT / "proof" / "scene" / "20260803-image-to-scene-smoke" / "scene_build_receipt.json"


def _inputs() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in (AUTHORED, CAMERA, BUILD)
    )  # type: ignore[return-value]


def test_hybrid_composition_uses_authoritative_camera_and_passes_validation() -> None:
    spec, receipt = compose_authoritative_hybrid_spec(*_inputs())
    assert receipt["classification"] == "PROVEN"
    assert receipt["scene_spec_valid"] is True
    assert receipt["validation_errors"] == []
    assert receipt["authoritative_fov_x_deg"] == 66.50838470458984
    assert receipt["legacy_fov_removed"] is True
    assert receipt["parallel_offset_views"] is True

    views = spec["camera"]["required_views"]
    assert [view["id"] for view in views] == ["source", "offset_left", "offset_right"]
    assert all(view["field_of_view_deg"] == 66.50838470458984 for view in views)
    assert spec["outputs"]["unreal_map"] == BUILD.read_text(encoding="utf-8").split('"map": "', 1)[1].split('"', 1)[0]
    assert spec["outputs"]["unreal_source_mesh_asset"].endswith("castlegrounds_source_mesh_v2")


def test_offsets_are_parallel_translations_along_camera_right() -> None:
    spec, receipt = compose_authoritative_hybrid_spec(*_inputs())
    assert receipt["parallel_offset_views"] is True
    source, left, right = spec["camera"]["required_views"]
    assert left["position_m"] != source["position_m"]
    assert right["position_m"] != source["position_m"]
    assert left["look_at_m"] != source["look_at_m"]
    assert right["look_at_m"] != source["look_at_m"]
    assert [left["position_m"][i] - source["position_m"][i] for i in range(3)] == pytest.approx(
        [-1.0, 0.0, 0.0], abs=1e-9
    )
    assert [right["position_m"][i] - source["position_m"][i] for i in range(3)] == pytest.approx(
        [1.0, 0.0, 0.0], abs=1e-9
    )


def test_unbuilt_assets_are_explicitly_not_promoted() -> None:
    spec, receipt = compose_authoritative_hybrid_spec(*_inputs())
    assert receipt["unbuilt_assets_marked_not_promoted"] is True
    for asset in spec["assets"]:
        if asset["id"] != "source_mesh_v2":
            assert "not_promoted" in asset["tags"]
            assert "placement_unproven" in asset["tags"]


def test_unproven_camera_is_rejected_before_composition() -> None:
    authored, camera, build = _inputs()
    camera["classification"] = "BLOCKED"
    with pytest.raises(ValueError, match="camera contract is not proven"):
        compose_authoritative_hybrid_spec(authored, camera, build)


def test_authored_spec_is_not_mutated() -> None:
    authored, camera, build = _inputs()
    before = copy.deepcopy(authored)
    compose_authoritative_hybrid_spec(authored, camera, build)
    assert authored == before
