import json

import pytest

from run_asset_pipeline import Pipeline, StageResult
from unified_pipeline_v2 import CANONICAL_STAGES, normalize_manifest


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
