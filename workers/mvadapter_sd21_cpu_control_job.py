"""Load-only direct SD2.1 MV-Adapter preflight.

This worker deliberately accepts CPU-created six-view controls and never calls
mesh, UV, raster, projection, nvdiffrast, or diffusion-generation helpers.
Inference is opt-in in a later task; this task exposes only ``--load-only``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_BASE = Path(r"C:\AI\HY3D2\HuggingFaceHub\models--stabilityai--stable-diffusion-2-1-base")
DEFAULT_ADAPTER = Path(r"C:\AI\HY3D2\HuggingFaceHub\models--huanngzh--mv-adapter\mvadapter_ig2mv_sd21.safetensors")
DEFAULT_SOURCE = Path(r"C:\AI\mvadapter-upstream-inspection")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"name": None, "total_mb": None, "free_mb": None}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        name, total, free = [part.strip() for part in completed.stdout.strip().splitlines()[0].rsplit(",", 2)]
        result = {"name": name, "total_mb": int(total), "free_mb": int(free)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            result["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 3)
            result["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 3)
            result["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 3)
    except Exception as exc:  # telemetry is recorded as not-proven, never guessed
        result["torch_error"] = type(exc).__name__
    return result


def validate_conditioning_image(path: Path, resolution: int) -> dict[str, Any]:
    image = Image.open(path).convert("RGBA")
    if image.size != (resolution, resolution):
        raise RuntimeError(f"SD21_CONDITIONING_DIMENSIONS_INVALID:{image.size}")
    array = np.asarray(image)
    alpha = array[:, :, 3] > 32
    if not alpha.any():
        raise RuntimeError("SD21_CONDITIONING_ALPHA_EMPTY")
    ys, xs = np.nonzero(alpha)
    occupancy = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) / float(resolution)
    if not 0.88 <= occupancy <= 0.92:
        raise RuntimeError(f"SD21_CONDITIONING_OCCUPANCY_INVALID:{occupancy:.6f}")
    outside = array[~alpha, :3]
    if outside.size and float(np.max(np.abs(outside.astype(np.int16) - 127))) > 1.0:
        raise RuntimeError("SD21_CONDITIONING_BACKGROUND_INVALID")
    centre_x = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0 / resolution
    centre_y = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0 / resolution
    if abs(centre_x - 0.5) > 0.04 or abs(centre_y - 0.5) > 0.04:
        raise RuntimeError("SD21_CONDITIONING_SUBJECT_NOT_CENTERED")
    return {
        "conditioning_sha256": sha256(path),
        "conditioning_dimensions": list(image.size),
        "conditioning_occupancy": round(float(occupancy), 6),
        "conditioning_foreground_pixels": int(alpha.sum()),
        "conditioning_background_rgb": [127, 127, 127],
    }


def validate_inputs(control_tensor: Path, contract_path: Path, conditioning_image: Path, resolution: int) -> dict[str, Any]:
    if not control_tensor.is_file() or not contract_path.is_file() or not conditioning_image.is_file():
        raise RuntimeError("SD21_PREFLIGHT_INPUT_MISSING")
    tensor = np.load(control_tensor, allow_pickle=False)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if tuple(tensor.shape) != (6, 6, resolution, resolution):
        raise RuntimeError(f"SD21_CONTROL_SHAPE_INVALID:{tuple(tensor.shape)}")
    if tensor.dtype not in (np.float16, np.float32):
        raise RuntimeError(f"SD21_CONTROL_DTYPE_INVALID:{tensor.dtype}")
    if not np.isfinite(tensor).all() or float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise RuntimeError("SD21_CONTROL_VALUES_INVALID")
    if contract.get("schema") != "lowvram3d_mvadapter_camera_contract_v1" or contract.get("view_count") != 6:
        raise RuntimeError("SD21_CAMERA_CONTRACT_INVALID")
    if not contract.get("fixture_gate_passed"):
        raise RuntimeError("SD21_CAMERA_CONTRACT_UNPROVEN")
    for proof_key in ("semantic_mapping_proven", "handedness_proven", "top_bottom_rotation_proven"):
        if not contract.get(proof_key):
            raise RuntimeError(f"SD21_CAMERA_{proof_key.upper()}_UNPROVEN")
    for key in ("front_rear_direction_dot", "left_right_direction_dot", "top_bottom_direction_dot"):
        if float(contract.get(key, 0.0)) > -0.999:
            raise RuntimeError(f"SD21_CAMERA_OPPOSITE_DOT_INVALID:{key}")
    conditioning = validate_conditioning_image(conditioning_image, resolution)
    return {
        "control_sha256": sha256(control_tensor),
        "contract_sha256": sha256(contract_path),
        **conditioning,
        "control_shape": list(tensor.shape),
        "control_dtype": str(tensor.dtype),
    }


def _assert_no_raster_imports() -> None:
    forbidden = [name for name in sys.modules if name == "nvdiffrast" or name.startswith("nvdiffrast.") or name.startswith("mvadapter.utils.mesh_utils")]
    if forbidden:
        raise RuntimeError("SD21_FORBIDDEN_RASTER_IMPORT:" + ",".join(sorted(forbidden)))


def load_sd21_pipeline(base_model: Path, adapter_path: Path, source_root: Path, offload_mode: str) -> dict[str, Any]:
    if "xl" in str(base_model).lower() or "sdxl" in str(base_model).lower():
        raise RuntimeError("SD21_WORKER_REJECTS_SDXL_BASE")
    if "ig2mv_sd21" not in adapter_path.name.lower():
        raise RuntimeError("SD21_WORKER_REQUIRES_IG2MV_SD21_ADAPTER")
    if not base_model.is_dir() or not (base_model / "model_index.json").is_file():
        raise RuntimeError("SD21_BASE_MODEL_UNAVAILABLE")
    if not adapter_path.is_file():
        raise RuntimeError("SD21_ADAPTER_WEIGHT_UNAVAILABLE")
    if not source_root.is_dir() or not (source_root / "mvadapter").is_dir():
        raise RuntimeError("SD21_UPSTREAM_MVADAPTER_SOURCE_UNAVAILABLE")
    sys.path.insert(0, str(source_root))
    _assert_no_raster_imports()
    import torch
    from diffusers import DDPMScheduler
    from safetensors.torch import load_file
    from mvadapter.models.attention_processor import DecoupledMVRowSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    if "nvdiffrast" in sys.modules:
        raise RuntimeError("SD21_FORBIDDEN_RASTER_IMPORT_AFTER_PIPELINE_IMPORT")
    pipe = MVAdapterI2MVSDPipeline.from_pretrained(
        str(base_model), torch_dtype=torch.float16, local_files_only=True, safety_checker=None
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler, shift_mode="interpolated", shift_scale=8.0, scheduler_class=DDPMScheduler
    )
    pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowSelfAttnProcessor2_0)
    adapter_state = load_file(str(adapter_path), device="cpu")
    pipe.load_custom_adapter(adapter_state, weight_name=adapter_path.name)
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if offload_mode == "model":
        pipe.enable_model_cpu_offload(device="cuda")
    elif offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device="cuda")
    else:
        raise RuntimeError("SD21_OFFLOAD_MODE_INVALID")
    _assert_no_raster_imports()
    return {"pipeline": pipe, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "offload_mode": offload_mode, "denoising_called": False}


def run_load_only(args: argparse.Namespace) -> dict[str, Any]:
    if not args.load_only:
        raise RuntimeError("SD21_LOAD_ONLY_REQUIRED_FOR_THIS_TASK")
    started = time.time()
    receipt: dict[str, Any] = {
        "schema": "lowvram3d_mvadapter_sd21_load_only_v1",
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "source_root": str(args.mvadapter_source),
        "seed": args.seed,
        "resolution": args.resolution,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "inference_started": False,
        "output_images": [],
        "nvdiffrast_imported": False,
        "offload_mode_requested": args.offload_mode,
    }
    try:
        receipt["python_executable"] = sys.executable
        receipt["python_version"] = sys.version
        import torch
        receipt["torch_version"] = torch.__version__
        receipt["torch_cuda"] = torch.version.cuda
        receipt["cuda_available"] = bool(torch.cuda.is_available())
        receipt["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        receipt["memory_before"] = memory_snapshot()
        receipt["model_cache_paths"] = {"base_model": str(args.base_model), "adapter": str(args.adapter)}
        receipt["base_model_sha256"] = sha256(args.base_model / "model_index.json")
        receipt["adapter_sha256"] = sha256(args.adapter)
        receipt["inputs"] = validate_inputs(args.control_tensor, args.camera_contract, args.conditioning_image, args.resolution)
        loaded = load_sd21_pipeline(args.base_model, args.adapter, args.mvadapter_source, args.offload_mode)
        receipt["pipeline_constructed"] = True
        receipt["adapter_loaded"] = True
        receipt["offload_hooks_installed"] = True
        receipt["offload_mode"] = loaded["offload_mode"]
        receipt["memory_after"] = memory_snapshot()
        receipt["peak_vram_mb"] = receipt["memory_after"].get("torch_peak_allocated_mb")
        receipt["nvdiffrast_imported"] = any(name == "nvdiffrast" or name.startswith("nvdiffrast.") for name in sys.modules)
        if receipt["nvdiffrast_imported"] or loaded["denoising_called"] or receipt["output_images"]:
            raise RuntimeError("SD21_LOAD_ONLY_SIDE_EFFECT_DETECTED")
        receipt["success"] = True
        receipt["status"] = "PROVEN"
    except Exception as exc:
        receipt["success"] = False
        receipt["status"] = "REJECTED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["memory_failure"] = memory_snapshot()
    receipt["wall_seconds"] = round(time.time() - started, 3)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-tensor", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--conditioning-image", "--reference-image", dest="conditioning_image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--mvadapter-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--reference-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--offload-mode", choices=("model", "sequential"), default="model")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (args.output_dir / "sd21_load_only_receipt.json")
    receipt = run_load_only(args)
    report_path.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    return 0 if receipt.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
