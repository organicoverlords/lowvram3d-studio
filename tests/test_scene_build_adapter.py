from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "evidence" / "latest-scene-hybrid" / "authoritative_hybrid_scene_spec.json"
PLAN_PATH = ROOT / "evidence" / "latest-scene-gameplay-proxy" / "gameplay_proxy_plan.json"
BUILD_SCRIPT = ROOT / "unreal" / "build_scene_from_spec.py"
VALIDATOR_SCRIPT = ROOT / "unreal" / "validate_scene_from_spec.py"


def _load_build_adapter():
    # The pure input gate is intentionally importable without an Unreal runtime.
    fake_unreal = types.ModuleType("unreal")
    sys.modules.setdefault("unreal", fake_unreal)
    module_spec = importlib.util.spec_from_file_location("build_scene_from_spec_test", BUILD_SCRIPT)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[dict, dict, argparse.Namespace]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    args = argparse.Namespace(
        source_map="/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke",
        output_map="/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1",
    )
    return spec, plan, args


def test_build_input_gate_accepts_authoritative_spec_and_unpromoted_plan() -> None:
    adapter = _load_build_adapter()
    source_asset, proxy_asset, camera = adapter._validate_inputs(*_inputs())
    assert source_asset["id"] == "source_mesh_v2"
    assert proxy_asset["id"] == "castle_proxy"
    assert camera["field_of_view_deg"] == pytest.approx(66.50838470458984)


def test_build_input_gate_protects_source_map_and_output_identity() -> None:
    adapter = _load_build_adapter()
    spec, plan, args = _inputs()
    args.source_map = "/Game/Other/Map"
    with pytest.raises(RuntimeError, match="protected authoritative"):
        adapter._validate_inputs(spec, plan, args)
    _, _, args = _inputs()
    args.output_map = args.source_map
    with pytest.raises(RuntimeError, match="output map"):
        adapter._validate_inputs(spec, plan, args)


def test_build_input_gate_rejects_promotion_and_camera_drift() -> None:
    adapter = _load_build_adapter()
    spec, plan, args = _inputs()
    promoted = copy.deepcopy(plan)
    promoted["promotion"] = True
    with pytest.raises(RuntimeError, match="promoted"):
        adapter._validate_inputs(spec, promoted, args)
    drifted = copy.deepcopy(spec)
    drifted["camera"]["source_camera"]["field_of_view_deg"] = 48.0
    with pytest.raises(RuntimeError, match="FOV"):
        adapter._validate_inputs(drifted, plan, args)


def test_build_input_gate_rejects_invalid_units_or_bounds() -> None:
    adapter = _load_build_adapter()
    spec, plan, args = _inputs()
    invalid_units = copy.deepcopy(plan)
    invalid_units["dimensions_cm"]["width_cm"] = 0.1
    with pytest.raises(RuntimeError, match="outside bounded limits"):
        adapter._validate_inputs(spec, invalid_units, args)
    non_finite = copy.deepcopy(plan)
    non_finite["dimensions_cm"]["height_cm"] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        adapter._validate_inputs(spec, non_finite, args)


def test_adapter_scripts_emit_required_manifest_and_protection_fields() -> None:
    build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
    validator_text = VALIDATOR_SCRIPT.read_text(encoding="utf-8")
    for token in (
        "hybrid_level_build_receipt_v1",
        "source_map_sha256_before",
        "source_map_sha256_after",
        "source_camera_contract_preserved",
        "proxy_not_promoted",
        "gpu_work_requested",
        "pcg_work_started",
    ):
        assert token in build_text
    for token in (
        "hybrid_level_validation_receipt_v1",
        "source_map_unmodified",
        "source_camera_contract_preserved",
    ):
        assert token in validator_text
