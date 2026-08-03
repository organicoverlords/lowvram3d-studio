from __future__ import annotations

import json
from pathlib import Path


def test_sdxl_comfy_route_cannot_be_default_gtx1660_route() -> None:
    root = Path(__file__).resolve().parents[1]
    status = json.loads((root / "configs" / "texture" / "gpu_panda_route_status.json").read_text())
    assert status["status"] == "REJECTED_FOR_6GB_HARDWARE"
    assert status["default_gtx1660_super_route"] == "mvadapter_sd21_direct_cpu_controls"
    assert status["gpu_sequence_consumed"] is False

