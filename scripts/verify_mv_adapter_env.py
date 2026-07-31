from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _probe(name: str, callback) -> dict[str, Any]:
    try:
        details = callback() or {}
        return {"ok": True, "details": details}
    except BaseException as exc:  # readiness diagnostics must also capture native import failures
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--proof", required=True)
    args = parser.parse_args()

    repo = str(Path(args.repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    probes: dict[str, dict[str, Any]] = {}

    def probe_torch() -> dict[str, Any]:
        import torch

        cuda_ready = bool(torch.cuda.is_available())
        details: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": cuda_ready,
            "torch_cuda": torch.version.cuda,
        }
        if cuda_ready:
            details["device"] = torch.cuda.get_device_name(0)
            details["capability"] = list(torch.cuda.get_device_capability(0))
        if not cuda_ready:
            raise RuntimeError("PyTorch imported but CUDA is not available")
        return details

    def probe_diffusers() -> dict[str, Any]:
        module = importlib.import_module("diffusers")
        return {"version": getattr(module, "__version__", "unknown")}

    def probe_transformers() -> dict[str, Any]:
        module = importlib.import_module("transformers")
        model_class = getattr(module, "AutoModelForImageSegmentation")
        return {
            "version": getattr(module, "__version__", "unknown"),
            "auto_image_segmentation": model_class.__name__,
        }

    def probe_opencv() -> dict[str, Any]:
        module = importlib.import_module("cv2")
        installed = {
            name: _distribution_version(name)
            for name in (
                "opencv-python",
                "opencv-python-headless",
                "opencv-contrib-python",
                "opencv-contrib-python-headless",
            )
        }
        active = [name for name, version in installed.items() if version != "not-installed"]
        if len(active) != 1 or active[0] != "opencv-contrib-python":
            raise RuntimeError(
                "Expected exactly opencv-contrib-python, found: "
                + ", ".join(f"{name}={installed[name]}" for name in active)
            )
        return {"cv2_version": module.__version__, "distributions": installed}

    def probe_mediapipe() -> dict[str, Any]:
        module = importlib.import_module("mediapipe")
        if not hasattr(module, "solutions"):
            raise RuntimeError("MediaPipe imported but mediapipe.solutions is unavailable")
        if not hasattr(module.solutions, "pose"):
            raise RuntimeError("MediaPipe imported but the Pose solution is unavailable")
        return {"version": getattr(module, "__version__", "unknown"), "pose": True}

    def probe_mvadapter() -> dict[str, Any]:
        module = importlib.import_module("mvadapter")
        return {"path": str(Path(module.__file__).resolve())}

    probes["torch_cuda"] = _probe("torch_cuda", probe_torch)
    probes["diffusers"] = _probe("diffusers", probe_diffusers)
    probes["transformers_birefnet_api"] = _probe("transformers_birefnet_api", probe_transformers)
    probes["opencv_single_distribution"] = _probe("opencv_single_distribution", probe_opencv)
    probes["mediapipe_pose"] = _probe("mediapipe_pose", probe_mediapipe)
    probes["mvadapter"] = _probe("mvadapter", probe_mvadapter)

    success = all(item["ok"] for item in probes.values())
    report = {
        "success": success,
        "python": sys.executable,
        "python_version": sys.version,
        "repo": repo,
        "probes": probes,
    }
    proof = Path(args.proof)
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
