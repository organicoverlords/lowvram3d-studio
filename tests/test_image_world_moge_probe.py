import json

import numpy as np
import pytest

from lowvram3d.image_world.contracts import ContractError
from lowvram3d.image_world.moge_probe import (
    MogeProbeReport,
    MogeProbeSettings,
    save_moge_maps,
    validate_moge_output,
)


def fake_output():
    return {
        "points": np.zeros((4, 5, 3), dtype=np.float32),
        "depth": np.ones((4, 5), dtype=np.float32),
        "normal": np.dstack([
            np.zeros((4, 5), dtype=np.float32),
            np.zeros((4, 5), dtype=np.float32),
            np.ones((4, 5), dtype=np.float32),
        ]),
        "mask": np.ones((4, 5), dtype=bool),
        "intrinsics": np.eye(3, dtype=np.float32),
    }


def test_default_probe_settings_match_low_vram_policy():
    settings = MogeProbeSettings()
    settings.validate()
    assert settings.model == "Ruicheng/moge-2-vits-normal"
    assert settings.num_tokens == 1200
    assert settings.max_gpu_memory_mb == 5600
    assert not settings.allow_download


def test_probe_rejects_out_of_range_token_count():
    with pytest.raises(ContractError, match="1200..2500"):
        MogeProbeSettings(num_tokens=1000).validate()


def test_output_validation_requires_matching_finite_maps():
    output = fake_output()
    summary = validate_moge_output(output)
    assert summary.height == 4
    assert summary.width == 5
    assert summary.valid_fraction == 1.0
    output["depth"][0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        validate_moge_output(output)


def test_save_maps_writes_non_pickle_numpy_artifacts(tmp_path):
    summary = save_moge_maps(fake_output(), tmp_path)
    assert summary.valid_fraction == 1.0
    assert np.load(tmp_path / "points.npy", allow_pickle=False).shape == (4, 5, 3)
    assert np.load(tmp_path / "intrinsics.npy", allow_pickle=False).shape == (3, 3)


def test_report_serializes_proof_boundary():
    report = MogeProbeReport(
        status="FAILED",
        source_sha256="a" * 64,
        settings=MogeProbeSettings(),
        output=None,
        wall_time_seconds=1.25,
        peak_gpu_allocated_mb=None,
        peak_gpu_reserved_mb=None,
        versions={"python": "3.11"},
        errors=("CUDA unavailable",),
    )
    payload = json.loads(report.to_json())
    assert payload["status"] == "FAILED"
    assert payload["errors"] == ["CUDA unavailable"]
