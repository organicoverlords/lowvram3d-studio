from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from mvadapter_sd21_cpu_control_job import load_sd21_pipeline, validate_inputs  # noqa: E402


def _inputs(tmp_path: Path, shape=(6, 6, 20, 20)) -> tuple[Path, Path, Path]:
    tensor = tmp_path / "control.npy"
    np.save(tensor, np.full(shape, 0.5, np.float32))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema": "lowvram3d_mvadapter_camera_contract_v1",
        "view_count": 6,
        "fixture_gate_passed": True,
        "semantic_mapping_proven": True,
        "handedness_proven": True,
        "top_rotation_proven": True,
        "bottom_rotation_proven": True,
        "top_bottom_rotation_proven": True,
        "front_rear_direction_dot": -1.0,
        "left_right_direction_dot": -1.0,
        "top_bottom_direction_dot": -1.0,
        "projection_half_span": 0.55,
        "index_semantics": {
            "0": "front", "1": "right", "2": "rear",
            "3": "left", "4": "top", "5": "bottom",
        },
        "fixture_evidence": {"evidence": [{"index": i, "passed": True} for i in range(6)]},
    }))
    reference = tmp_path / "reference.png"
    image = np.full((20, 20, 4), (127, 127, 127, 0), np.uint8)
    image[1:19, 1:19] = (50, 90, 140, 255)
    Image.fromarray(image, "RGBA").save(reference)
    return tensor, contract, reference


def test_direct_worker_accepts_only_six_view_control_shape(tmp_path: Path) -> None:
    tensor, contract, reference = _inputs(tmp_path)
    result = validate_inputs(tensor, contract, reference, 20)
    assert result["control_shape"] == [6, 6, 20, 20]
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((4, 6, 20, 20), np.float32))
    with pytest.raises(RuntimeError, match="SHAPE_INVALID"):
        validate_inputs(bad, contract, reference, 20)


def test_direct_worker_rejects_unproven_contract(tmp_path: Path) -> None:
    tensor, contract, reference = _inputs(tmp_path)
    payload = json.loads(contract.read_text())
    payload["fixture_gate_passed"] = False
    contract.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="UNPROVEN"):
        validate_inputs(tensor, contract, reference, 20)


def test_direct_worker_rejects_sdxl_before_model_loading(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="REJECTS_SDXL"):
        load_sd21_pipeline(
            tmp_path / "stable-diffusion-xl-base-1.0",
            tmp_path / "mvadapter_ig2mv_sd21.safetensors",
            tmp_path,
            "sequential",
        )


def test_direct_worker_requires_sequential_offload(tmp_path: Path) -> None:
    (tmp_path / "sd21").mkdir()
    (tmp_path / "sd21" / "model_index.json").write_text("{}")
    (tmp_path / "mvadapter_ig2mv_sd21.safetensors").write_bytes(b"")
    with pytest.raises(RuntimeError, match="MUST_BE_SEQUENTIAL"):
        load_sd21_pipeline(
            tmp_path / "sd21",
            tmp_path / "mvadapter_ig2mv_sd21.safetensors",
            tmp_path,
            "model",
        )


def test_direct_worker_rejects_text_conditioned_index_semantics(tmp_path: Path) -> None:
    tensor, contract, reference = _inputs(tmp_path)
    payload = json.loads(contract.read_text())
    payload["index_semantics"]["4"] = "front"
    contract.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="TOP_BOTTOM_INDEX_INVALID"):
        validate_inputs(tensor, contract, reference, 20)
