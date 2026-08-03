from __future__ import annotations

"""Lazy compatibility bridge for the pinned 3DDFA_V2 import path.

The scene worker normally uses ONNX inference, but the pinned upstream module imports
PyTorch at module-load time and may use it once to create the BFM decoder ONNX file.
This bridge installs the official CPU wheel only when that import is reached, then
replaces itself with the real package. It also restores NumPy's removed ``np.long``
alias required by the pinned 2022 code.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np


if not hasattr(np, "long"):
    np.long = np.int64  # type: ignore[attr-defined]

_LOCAL_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
_ORIGINAL_SYS_PATH = list(sys.path)


def _resolved(path_entry: str) -> Path | None:
    try:
        return Path(path_entry or ".").resolve()
    except OSError:
        return None


def _install_runtime() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.8.0",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "onnxscript",
        ]
    )


_install_runtime()

# Remove this compatibility package from import resolution, then load the real wheel.
sys.modules.pop(__name__, None)
sys.path = [
    entry
    for entry in sys.path
    if _resolved(entry) != _LOCAL_PACKAGE_PARENT
]
try:
    _real_torch = importlib.import_module(__name__)
finally:
    sys.path = _ORIGINAL_SYS_PATH

sys.modules[__name__] = _real_torch
globals().update(_real_torch.__dict__)
