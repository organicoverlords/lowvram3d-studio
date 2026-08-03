from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lowvram3d.scene_camera import (
    apply_camera_contract_to_scene_spec,
    build_camera_contract,
)
from lowvram3d.scene_spec import validate_scene_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "proof" / "scene" / "20260803-image-to-scene-smoke"
MIGRATED = REPO_ROOT / "evidence" / "latest-scene-spec-local-worker" / "migrated_scene_spec.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _inputs() -> tuple[dict, dict, dict]:
    return (
        _read(EVIDENCE / "camera_calibration.json"),
        _read(EVIDENCE / "blender_exact_source_receipt.json"),
        _read(EVIDENCE / "scene_interpretation.json"),
    )


def test_authoritative_camera_supersedes_legacy_estimate() -> None:
    calibration, exact, interpretation = _inputs()
    contract, bundle = build_camera_contract(calibration, exact, interpretation)

    assert contract["classification"] == "PROVEN"
    assert contract["fov_x_deg"] == pytest.approx(66.50838470458984)
    assert contract["fov_y_deg"] == pytest.approx(52.37591552734375)
    assert contract["legacy_interpretation_fov_x_deg"] == 48.0
    assert contract["legacy_interpretation_superseded"] is True
    assert contract["basis_handedness_error"] <= 1e-5
    assert bundle["receipt"]["basis_normalized"] is True
    assert bundle["receipt"]["basis_orthogonal"] is True
    assert bundle["receipt"]["basis_right_handed"] is True


def test_camera_contract_updates_scene_spec_source_view() -> None:
    calibration, exact, interpretation = _inputs()
    contract, _ = build_camera_contract(calibration, exact, interpretation)
    spec = _read(MIGRATED)
    updated = apply_camera_contract_to_scene_spec(spec, contract)

    source = updated["camera"]["source_camera"]
    assert source["field_of_view_deg"] == contract["fov_x_deg"]
    assert source["position_m"] == contract["origin_m"]
    assert source["look_at_m"] == pytest.approx(
        [
            contract["origin_m"][0] + contract["forward"][0],
            contract["origin_m"][1] + contract["forward"][1],
            contract["origin_m"][2] + contract["forward"][2],
        ]
    )
    assert updated["outputs"]["authoritative_camera_contract"].endswith(
        "camera_contract.json"
    )
    assert validate_scene_spec(updated)["scene_spec_valid"] is True


def test_camera_contract_rejects_disagreeing_proven_fov() -> None:
    calibration, exact, interpretation = _inputs()
    calibration = copy.deepcopy(calibration)
    calibration["fov_x_deg"] = 10.0
    with pytest.raises(ValueError, match="horizontal FOV mismatch"):
        build_camera_contract(calibration, exact, interpretation)


def test_camera_contract_rejects_non_orthogonal_basis() -> None:
    calibration, exact, interpretation = _inputs()
    exact = copy.deepcopy(exact)
    exact["camera"]["up"] = list(exact["camera"]["forward"])
    with pytest.raises(ValueError, match="not orthogonal"):
        build_camera_contract(calibration, exact, interpretation)
