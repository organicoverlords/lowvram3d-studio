import json

import pytest

from run_asset_pipeline import Pipeline, StageResult, hash_inputs
from pipeline_v2_production_stages import validate_mvadapter_inputs
from unified_pipeline_v2 import CANONICAL_STAGES, normalize_manifest
from uv_xatlas_isolated import PRESET_ORDER, choose_next


def manifest(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"fixture-source")
    return {
        "asset_id": "fixture_static_asset",
        "profile": "humanoid",
        "source": {"path": str(source), "sha256": "fixture-sha"},
        "output_root": str(tmp_path / "run"),
        "texture": {"resolution": 1024},
    }


def test_canonical_static_stage_order_excludes_rigging():
    assert CANONICAL_STAGES == (
        "INGEST", "GENERATE", "GEOMETRY_QA", "CLEAN", "LOD", "UV",
        "BAKE", "TEXTURE", "TEXTURE_QA", "EXPORT_QA",
    )
    assert "RIG" not in CANONICAL_STAGES
    assert "ANIMATION" not in CANONICAL_STAGES


def test_manifest_normalization_preserves_source_and_adds_static_contract(tmp_path):
    original = manifest(tmp_path)
    result = normalize_manifest(original, stage_from="CLEAN", stage_to="EXPORT_QA",
                                existing_master="C:\\fixture\\panda.glb")
    assert result["source"] == original["source"]
    assert result["intent"] == {"static": True, "rigged": False, "animated": False}
    assert result["geometry_source"]["kind"] == "existing_master"
    assert result["stage_range"] == {"from": "CLEAN", "to": "EXPORT_QA"}


def test_manifest_rejects_missing_source_hash(tmp_path):
    value = manifest(tmp_path)
    del value["source"]["sha256"]
    with pytest.raises(ValueError, match="SOURCE_HASH_REQUIRED"):
        normalize_manifest(value)


def test_resume_skips_unchanged_stage_and_invalidates_changed_input(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"one")
    pipeline = Pipeline(manifest(tmp_path), tmp_path / "state", "python", "blender", verbose=False)
    calls = []

    def runner(_overrides):
        calls.append(1)
        output = tmp_path / f"candidate-{len(calls)}.json"
        output.write_text(json.dumps({"call": len(calls)}), encoding="utf-8")
        return StageResult("passed", outputs={"result": output})

    pipeline.execute("FIXTURE", [source], runner)
    pipeline.execute("FIXTURE", [source], runner)
    assert len(calls) == 1
    source.write_bytes(b"two")
    pipeline.execute("FIXTURE", [source], runner)
    assert len(calls) == 2


def test_directory_inputs_are_fingerprinted_for_resume(tmp_path):
    bundle = tmp_path / "controls"
    bundle.mkdir()
    (bundle / "camera_contract.json").write_text("{}", encoding="utf-8")
    first = hash_inputs([bundle])
    (bundle / "front_mask.png").write_bytes(b"mask")
    second = hash_inputs([bundle])
    assert first[str(bundle)] != second[str(bundle)]


def test_mvadapter_preflight_rejects_partial_external_bundle(tmp_path):
    bundle = tmp_path / "controls"
    bundle.mkdir()
    views = ["front", "right", "rear", "left", "top", "bottom"]
    (bundle / "camera_contract.json").write_text(json.dumps({
        "views": [{"index": i, "semantic_name": name} for i, name in enumerate(views)]
    }), encoding="utf-8")
    receipt = tmp_path / "inference_receipt.json"
    receipt.write_text(json.dumps({"output_images": []}), encoding="utf-8")
    result = validate_mvadapter_inputs(bundle, receipt)
    assert not result["passed"]
    assert any("front_mask.png" in value for value in result["missing_inputs"])


def test_xatlas_policy_is_isolated_and_stops_at_first_valid_preset():
    assert PRESET_ORDER == ("A", "B", "C")
    assert choose_next([]) == "A"
    assert choose_next(["A"]) == "B"
    assert choose_next(["A", "B"]) == "C"
    assert choose_next(["A", "B", "C"]) is None


def test_normalized_manifest_defaults_to_bounded_uv_contract(tmp_path):
    result = normalize_manifest(manifest(tmp_path))
    assert result["uv"] == {
        "resolution": 1024, "padding": 4, "candidate_timeout_seconds": 600,
        "route": "fast_blender",
    }


def test_lod_policy_accepts_only_bounded_modes(tmp_path):
    value = manifest(tmp_path)
    value["lod"] = {"mode": "preserve_source"}
    assert normalize_manifest(value)["lod"]["mode"] == "preserve_source"
    value["lod"] = {"mode": "required"}
    assert normalize_manifest(value)["lod"]["mode"] == "required"
    value["lod"] = {"mode": "invent"}
    with pytest.raises(ValueError, match="LOD_MODE_INVALID"):
        normalize_manifest(value)
