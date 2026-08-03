from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from run_mv_adapter_fp32_canary_camera_only import (  # noqa: E402
    install_camera_only_nvdiffrast_shim,
)


def test_camera_only_shim_satisfies_import_and_refuses_raster_calls(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "nvdiffrast", raising=False)
    monkeypatch.delitem(sys.modules, "nvdiffrast.torch", raising=False)

    mode = install_camera_only_nvdiffrast_shim(force=True)

    assert mode == "camera_only_import_shim"
    import nvdiffrast.torch as dr

    try:
        with pytest.raises(RuntimeError, match="camera-only MV-Adapter mode"):
            _ = dr.rasterize
    finally:
        # The shim is intentionally installed only for this test.  Do not let
        # it contaminate later tests that prove the production route never
        # imports nvdiffrast.
        sys.modules.pop("nvdiffrast.torch", None)
        sys.modules.pop("nvdiffrast", None)
