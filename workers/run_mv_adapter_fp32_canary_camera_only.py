"""Launch the MV-Adapter numerical canary without requiring nvdiffrast.

The official SD2.1 image-to-multiview script imports camera helpers through
``mvadapter.utils.mesh_utils``.  That package imports ``nvdiffrast.torch`` at
module import time even though the image-to-multiview camera path never calls a
rasterizer.  On the GTX 1660 SUPER lane nvdiffrast is deliberately unavailable
because its raster path is unsupported and previously failed with error 209.

This launcher installs a fail-closed import-only shim for ``nvdiffrast.torch``
when the real package is absent, then delegates to
``run_mv_adapter_fp32_canary.main``.  Any attempt to call a raster operation
through the shim raises immediately.  The canary itself remains isolated from
texture projection and prior jobs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Any


class _UnavailableRasterModule(types.ModuleType):
    """Import-compatible module that refuses every raster operation."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            "nvdiffrast is unavailable in camera-only MV-Adapter mode; "
            f"attempted to access nvdiffrast.torch.{name}"
        )


def install_camera_only_nvdiffrast_shim(*, force: bool = False) -> str:
    """Install the import-only shim when nvdiffrast is unavailable.

    Returns ``real_nvdiffrast`` when an installed package is available and
    ``camera_only_import_shim`` when the fail-closed shim was installed.
    ``force`` exists only for deterministic unit testing.
    """

    if not force:
        try:
            available = importlib.util.find_spec("nvdiffrast.torch") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if available:
            return "real_nvdiffrast"

    package = types.ModuleType("nvdiffrast")
    torch_module = _UnavailableRasterModule("nvdiffrast.torch")
    package.torch = torch_module  # type: ignore[attr-defined]
    package.__path__ = []  # type: ignore[attr-defined]

    sys.modules["nvdiffrast"] = package
    sys.modules["nvdiffrast.torch"] = torch_module
    return "camera_only_import_shim"


def main() -> None:
    mode = install_camera_only_nvdiffrast_shim()
    print(f"NVDIFFRAST_MODE={mode}", flush=True)
    print("NVDIFFRAST_RASTER_OPERATIONS_ALLOWED=false", flush=True)

    workers_dir = Path(__file__).resolve().parent
    if str(workers_dir) not in sys.path:
        sys.path.insert(0, str(workers_dir))

    from run_mv_adapter_fp32_canary import main as canary_main

    canary_main()


if __name__ == "__main__":
    main()
