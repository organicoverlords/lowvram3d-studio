"""Run one bounded image+geometry SD2.1 MV-Adapter six-view sequence.

This worker owns generation and QA only.  It never rasterises a mesh, writes a
UV/atlas/GLB, or imports nvdiffrast.  A separate process must invoke the
``oom-fallback`` attempt, and only after a primary receipt classified a real
CUDA allocator OOM.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROMPT = (
    "high quality clean albedo reference of the same tactical red panda character, "
    "consistent materials, consistent identity, flat neutral lighting"
)
VIEW_NAMES = ("front", "right", "rear", "left", "top", "bottom")
MIN_RAM_MB = 2048
MIN_PAGEFILE_MB = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nvidia_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "total_mb": None, "free_mb": None, "active_processes": []}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        row = [part.strip() for part in query.stdout.strip().splitlines()[0].split(",")]
        result.update({"available": True, "name": row[0], "total_mb": int(row[1]), "free_mb": int(row[2])})
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        ignored_graphics: list[dict[str, str]] = []
        for line in apps.stdout.splitlines():
            if line.strip():
                parts = [part.strip() for part in line.split(",")]
                record = {"pid": parts[0], "name": parts[1], "used_mb": parts[2]}
                try:
                    used_mb = float(parts[2])
                except (ValueError, IndexError):
                    # On Windows, nvidia-smi may expose ordinary desktop
                    # graphics clients through this query with N/A memory.
                    # They are not actionable CUDA compute jobs.
                    ignored_graphics.append(record)
                    continue
                if used_mb > 0.0:
                    result["active_processes"].append(record)
        result["ignored_graphics_processes"] = ignored_graphics
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return result
    return result


def _system_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("MVADAPTER_PSUTIL_REQUIRED_FOR_RAM_PAGEFILE_GATE") from exc
    memory = psutil.virtual_memory()
    pagefile = psutil.swap_memory()
    return {
        "ram_total_mb": round(memory.total / 2**20, 3),
        "ram_available_mb": round(memory.available / 2**20, 3),
        "pagefile_total_mb": round(pagefile.total / 2**20, 3),
        "pagefile_free_mb": round(pagefile.free / 2**20, 3),
        "pid": os.getpid(),
    }


def _torch_memory(torch: Any) -> dict[str, Any]:
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 3),
        "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 3),
    }


def _sdpa_backend_report(torch: Any) -> dict[str, Any]:
    cuda = getattr(torch.backends, "cuda", None)
    return {
        "api": "torch.nn.functional.scaled_dot_product_attention",
        "flash_sdp_enabled": bool(cuda.flash_sdp_enabled()) if cuda is not None else None,
        "mem_efficient_sdp_enabled": bool(cuda.mem_efficient_sdp_enabled()) if cuda is not None else None,
        "math_sdp_enabled": bool(cuda.math_sdp_enabled()) if cuda is not None else None,
        "attention_backend": "PYTORCH_SDPA",
    }


def _heartbeat(path: Path, receipt: dict[str, Any], phase: str, **fields: Any) -> None:
    record = {"timestamp": time.time(), "pid": os.getpid(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    receipt.setdefault("heartbeats", []).append(record)


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "cuda out of memory" in message or "cuda error: out of memory" in message


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"MVADAPTER_{label}_MISSING:{path}")
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"MVADAPTER_{label}_HASH_MISMATCH:{actual}:{expected}")
    return actual


def _selected(config: dict[str, Any], attempt: str) -> tuple[dict[str, Any], str]:
    if config.get("schema") != "lowvram3d_mvadapter_ig2mv_sd21_inference_v1":
        raise RuntimeError("MVADAPTER_CONFIG_SCHEMA_INVALID")
    allowed_statuses = {"PREPARED_NOT_EXECUTED"}
    if attempt == "primary":
        allowed_statuses.add("PRIMARY_384_READY_AFTER_TINY_GATE")
    if attempt == "oom-fallback":
        allowed_statuses.add("PRIMARY_384_CUDA_OOM_FALLBACK_AUTHORIZED")
    if config.get("status") not in allowed_statuses:
        raise RuntimeError(f"MVADAPTER_CONFIG_STATUS_INVALID:{config.get('status')}")
    if config.get("gpu_sequence_consumed"):
        raise RuntimeError("MVADAPTER_GPU_SEQUENCE_ALREADY_CONSUMED")
    if config.get("generator_family") != "MVADAPTER_SD21" or config.get("conditioning") != "IMAGE_PLUS_GEOMETRY":
        raise RuntimeError("MVADAPTER_ROUTE_INVALID")
    if config.get("text_conditioned_fallback") != "FORBIDDEN" or config.get("comfyui_generation_route") == "PROVEN":
        raise RuntimeError("MVADAPTER_FORBIDDEN_ROUTE_CONFIGURED")
    if config.get("pipeline_class") != "LowVRAMMVAdapterI2MVSDPipeline":
        raise RuntimeError("MVADAPTER_PIPELINE_CLASS_INVALID")
    if config.get("adapter", "").split("\\")[-1] != "mvadapter_ig2mv_sd21.safetensors":
        raise RuntimeError("MVADAPTER_ADAPTER_FILENAME_INVALID")
    if attempt == "primary":
        selected = config["primary"]
    elif attempt == "oom-fallback":
        selected = config["oom_fallback"]
    else:
        raise RuntimeError(f"MVADAPTER_ATTEMPT_INVALID:{attempt}")
    return selected, attempt


def validate_preflight(config_path: Path, attempt: str, primary_receipt: Path | None = None) -> dict[str, Any]:
    config = _json(config_path)
    selected, attempt = _selected(config, attempt)
    if attempt == "oom-fallback":
        if primary_receipt is None or not primary_receipt.is_file():
            raise RuntimeError("MVADAPTER_OOM_FALLBACK_PRIMARY_RECEIPT_REQUIRED")
        primary = _json(primary_receipt)
        if primary.get("status") != "CUDA_OOM" or not primary.get("fallback_eligible"):
            raise RuntimeError("MVADAPTER_OOM_FALLBACK_NOT_AUTHORIZED")
    adapter = Path(config["adapter"])
    mesh = Path(config["mesh"])
    conditioning = Path(selected["conditioning_reference"])
    controls = Path(selected["control_tensor"])
    contract = Path(selected["camera_contract"])
    adapter_hash = _verify_hash(adapter, config["adapter_sha256"], "ADAPTER")
    mesh_hash = _verify_hash(mesh, config["immutable_mesh_sha256"], "MESH")
    conditioning_hash = _verify_hash(conditioning, selected["conditioning_reference_sha256"], "CONDITIONING")
    control_hash = _verify_hash(controls, selected["control_tensor_sha256"], "CONTROL")
    contract_hash = _verify_hash(contract, selected["camera_contract_sha256"], "CAMERA_CONTRACT")
    tensor = np.load(controls, allow_pickle=False)
    resolution = int(selected["resolution"])
    if tuple(tensor.shape) != (6, 6, resolution, resolution):
        raise RuntimeError(f"MVADAPTER_CONTROL_SHAPE_INVALID:{tuple(tensor.shape)}")
    if tensor.dtype not in (np.float16, np.float32) or not np.isfinite(tensor).all():
        raise RuntimeError("MVADAPTER_CONTROL_TENSOR_INVALID")
    camera = _json(contract)
    if camera.get("view_count") != 6 or not camera.get("fixture_gate_passed"):
        raise RuntimeError("MVADAPTER_CAMERA_CONTRACT_UNPROVEN")
    for key in ("semantic_mapping_proven", "handedness_proven", "top_bottom_rotation_proven"):
        if not camera.get(key):
            raise RuntimeError(f"MVADAPTER_CAMERA_{key.upper()}_UNPROVEN")
    if sorted(int(view["index"]) for view in camera.get("views", [])) != list(range(6)):
        raise RuntimeError("MVADAPTER_CAMERA_VIEW_INDEX_INVALID")
    if float(camera.get("front_rear_direction_dot", 0.0)) > -0.999:
        raise RuntimeError("MVADAPTER_CAMERA_FRONT_REAR_NOT_OPPOSITE")
    if float(camera.get("left_right_direction_dot", 0.0)) > -0.999:
        raise RuntimeError("MVADAPTER_CAMERA_LEFT_RIGHT_NOT_OPPOSITE")
    if float(camera.get("top_bottom_direction_dot", 0.0)) > -0.999:
        raise RuntimeError("MVADAPTER_CAMERA_TOP_BOTTOM_NOT_OPPOSITE")
    if int(selected["views"]) != 6 or int(selected["resolution"]) not in (256, 384):
        raise RuntimeError("MVADAPTER_EXECUTION_DIMENSIONS_INVALID")
    system = _system_snapshot()
    if system["ram_available_mb"] < MIN_RAM_MB or system["pagefile_free_mb"] < MIN_PAGEFILE_MB:
        raise RuntimeError(f"MVADAPTER_RAM_PAGEFILE_INSUFFICIENT:{system}")
    gpu = _nvidia_snapshot()
    if not gpu["available"]:
        raise RuntimeError("MVADAPTER_NVIDIA_SMI_UNAVAILABLE")
    if gpu["active_processes"]:
        raise RuntimeError(f"MVADAPTER_CONFLICTING_GPU_PROCESS:{gpu['active_processes']}")
    return {
        "config": config,
        "selected": selected,
        "attempt": attempt,
        "adapter": adapter,
        "mesh": mesh,
        "conditioning": conditioning,
        "controls": controls,
        "contract": contract,
        "adapter_sha256": adapter_hash,
        "mesh_sha256": mesh_hash,
        "conditioning_sha256": conditioning_hash,
        "control_sha256": control_hash,
        "camera_contract_sha256": contract_hash,
        "camera": camera,
        "system_before": system,
        "gpu_before": gpu,
        "resolution": resolution,
    }


def _image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _foreground_mask(array: np.ndarray) -> np.ndarray:
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array.astype(np.float32) - background, axis=2)
    saturation = (array.max(axis=2).astype(np.float32) - array.min(axis=2).astype(np.float32)) / 255.0
    mask = (distance > 12.0) | (saturation > 0.06)
    if int(mask.sum()) < max(32, array.shape[0] * array.shape[1] // 200):
        mask = distance > 6.0
    return mask


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _transform_mask(mask: np.ndarray, scale: float, dx: int, dy: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    size = max(1, int(round(mask.shape[0] * scale)))
    resized = image.resize((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("L", image.size, 0)
    left = (mask.shape[1] - size) // 2 + dx
    top = (mask.shape[0] - size) // 2 + dy
    canvas.paste(resized, (left, top))
    return np.asarray(canvas) > 127


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _registered_iou(generated: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
    best = (0.0, {"scale": 1.0, "dx": 0, "dy": 0})
    span = max(1, generated.shape[0] // 16)
    for scale in (0.94, 0.97, 1.0, 1.03, 1.06):
        for dx in range(-span, span + 1, max(1, span // 2)):
            for dy in range(-span, span + 1, max(1, span // 2)):
                score = _iou(_transform_mask(generated, scale, dx, dy), target)
                if score > best[0]:
                    best = (score, {"scale": scale, "dx": dx, "dy": dy})
    return best


def _corr(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = left.astype(np.float32).mean(axis=2)
    b = right.astype(np.float32).mean(axis=2)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    else:
        a = a.ravel()
        b = b.ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _color_qa(array: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    pixels = array[mask].astype(np.float32) / 255.0
    if len(pixels) == 0:
        return {"foreground_saturation": 0.0, "black_clipping_fraction": 1.0, "white_clipping_fraction": 1.0}
    saturation = (pixels.max(axis=1) - pixels.min(axis=1)).mean()
    return {
        "foreground_saturation": round(float(saturation), 6),
        "foreground_luminance": round(float((pixels @ np.asarray([0.2126, 0.7152, 0.0722])).mean()), 6),
        "black_clipping_fraction": round(float((pixels.max(axis=1) < 0.02).mean()), 6),
        "white_clipping_fraction": round(float((pixels.min(axis=1) > 0.98).mean()), 6),
        "channel_std": round(float(pixels.std(axis=0).mean()), 6),
    }


def _decoded_image_gate(images: list[Image.Image]) -> dict[str, Any]:
    """Are the decoded PNGs finite, non-black, non-flat and not six copies of one another?

    This is deliberately weaker than ``qa_outputs``: it asks whether the pipeline produced *content*
    at all, which is the question a numerical repair has to answer, and says nothing about whether
    that content is a usable texture.
    """
    views: list[dict[str, Any]] = []
    digests: list[str] = []
    for index, image in enumerate(images):
        array = _image_array(image).astype(np.float32)
        digests.append(hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
        views.append({
            "index": index,
            "finite": bool(np.isfinite(array).all()),
            "mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6),
            "non_black": bool(float(array.mean()) >= 2.0),
            "non_flat": bool(float(array.std()) >= 2.0),
        })
    identical = len(set(digests)) == 1
    gate = {
        "schema": "lowvram3d_mvadapter_decoded_image_gate_v1",
        "image_count": len(images),
        "views": views,
        "distinct_image_hashes": len(set(digests)),
        "all_views_identical": bool(identical),
        "vae_output_finite": all(view["finite"] for view in views),
        "all_non_black": all(view["non_black"] for view in views),
        "all_non_flat": all(view["non_flat"] for view in views),
    }
    gate["passed"] = bool(
        len(images) == 6 and gate["vae_output_finite"] and gate["all_non_black"]
        and gate["all_non_flat"] and not identical
    )
    return gate


def qa_outputs(images: list[Image.Image], controls_dir: Path, resolution: int, semantic_names: list[str]) -> dict[str, Any]:
    if len(images) != 6:
        raise RuntimeError(f"MVADAPTER_OUTPUT_COUNT_INVALID:{len(images)}")
    views: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    generated_masks: list[np.ndarray] = []
    for index, image in enumerate(images):
        array = _image_array(image)
        if array.shape != (resolution, resolution, 3) or not np.isfinite(array).all():
            raise RuntimeError(f"MVADAPTER_OUTPUT_IMAGE_INVALID:{index}:{array.shape}")
        if float(array.mean()) < 2.0 or float(array.mean()) > 253.0 or float(array.std()) < 2.0:
            raise RuntimeError(f"MVADAPTER_OUTPUT_BLANK_OR_FLAT:{index}")
        arrays.append(array)
        generated_mask = _foreground_mask(array)
        generated_masks.append(generated_mask)
        target_path = controls_dir / f"{('horizontal_' + str(index)) if index < 4 else VIEW_NAMES[index]}_mask.png"
        target = np.asarray(Image.open(target_path).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)) > 32
        direct = _iou(generated_mask, target)
        registered, transform = _registered_iou(generated_mask, target)
        bbox = _bbox(generated_mask)
        clipping = 0.0
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            edge = generated_mask[[0, -1], :].sum() + generated_mask[:, [0, -1]].sum()
            clipping = float(edge / max(1, generated_mask.sum()))
        qa = _color_qa(array, generated_mask)
        views.append({
            "index": index,
            "name": semantic_names[index],
            "dimensions": [resolution, resolution],
            "foreground_coverage": round(float(generated_mask.mean()), 6),
            "direct_iou": round(direct, 6),
            "registered_iou": round(registered, 6),
            "registration": transform,
            "clipping_fraction": round(clipping, 6),
            "color": qa,
            "passed_generation_reference_gate": bool(registered >= (0.65 if index >= 4 else 0.75) and clipping <= 0.08),
        })
    rear_mask = np.logical_or(generated_masks[0], generated_masks[2])
    direct_corr = _corr(arrays[0], arrays[2], rear_mask)
    mirrored_corr = _corr(
        arrays[0], arrays[2][:, ::-1], np.logical_or(generated_masks[0], generated_masks[2][:, ::-1])
    )
    rear = views[2]
    qa = {
        "schema": "lowvram3d_mvadapter_six_view_qa_v1",
        "views": views,
        "front_rear_direct_correlation": round(direct_corr, 6),
        "front_rear_mirrored_correlation": round(mirrored_corr, 6),
        "rear_numeric_gate_passed": bool(direct_corr < 0.82 and mirrored_corr < 0.82),
        "structural_gate_passed": all(view["passed_generation_reference_gate"] for view in views),
        "colour_gate_passed": all(
            view["color"]["foreground_saturation"] >= 0.08
            and view["color"]["black_clipping_fraction"] < 0.05
            and view["color"]["white_clipping_fraction"] < 0.05
            for view in views
        ),
        "semantic_gate": "USER_REVIEW_REQUIRED",
        "rear_semantic_visual_review_required": True,
    }
    qa["passed"] = bool(qa["structural_gate_passed"] and qa["colour_gate_passed"] and qa["rear_numeric_gate_passed"])
    return qa


def _contact_sheet(images: list[Image.Image], output: Path, semantic_names: list[str]) -> None:
    tile = images[0].convert("RGB")
    size = tile.size[0]
    sheet = Image.new("RGB", (size * 3, size * 2), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = (index % 3) * size
        y = (index // 3) * size
        sheet.paste(image.convert("RGB"), (x, y))
        draw.text((x + 8, y + 8), f"{index}: {semantic_names[index]}", fill=(255, 255, 255))
    sheet.save(output)


def _update_config(config_path: Path, status: str, consumed: bool = True) -> None:
    config = _json(config_path)
    config["status"] = status
    config["gpu_sequence_consumed"] = bool(consumed)
    config["next_action"] = "USER_REVIEW_REQUIRED"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def execute(config_path: Path, output_dir: Path, attempt: str, primary_receipt: Path | None = None,
            step_split_probe: bool = False, finite_gate: bool = False) -> dict[str, Any]:
    preflight = validate_preflight(config_path, attempt, primary_receipt)
    selected = preflight["selected"]
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.jsonl"
    if heartbeat_path.exists():
        raise RuntimeError(f"MVADAPTER_HEARTBEAT_PATH_ALREADY_EXISTS:{heartbeat_path}")
    receipt: dict[str, Any] = {
        "schema": "lowvram3d_mvadapter_sd21_six_view_inference_v1",
        "config": str(config_path),
        "attempt": attempt,
        "attempt_resolution": int(selected["resolution"]),
        "pipeline_class": preflight["config"]["pipeline_class"],
        "prompt": preflight["config"].get("prompt", PROMPT),
        "negative_prompt": None,
        "classifier_free_guidance": False,
        "parameters": selected,
        "preflight": {key: value for key, value in preflight.items() if key not in {"config", "selected", "camera"}},
        "reference_unet_pass_started": False,
        "cond_encoder_executed": False,
        "denoising_started": False,
        "denoising_steps_completed": 0,
        "vae_decode_completed": False,
        "output_images": [],
        "gpu_sequence_consumed": False,
        "fallback_eligible": False,
        "reference_unet_call_count": 0,
        "denoising_unet_call_count": 0,
        "denoising_steps_requested": int(selected["steps"]),
        "cuda_oom": "NO",
        "fallback_to_256": "FORBIDDEN",
        "_started": time.time(),
    }
    _heartbeat(heartbeat_path, receipt, "preflight_passed", resolution=int(selected["resolution"]))
    torch = None
    pipe = None
    original_unet_forward = None
    original_cond_forward = None
    try:
        import torch as torch_module
        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("MVADAPTER_CUDA_UNAVAILABLE")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        receipt["memory_before_model"] = {**preflight["gpu_before"], **_torch_memory(torch)}
        _heartbeat(heartbeat_path, receipt, "model_construction_started", **receipt["memory_before_model"])
        upstream = Path(preflight["config"].get("mvadapter_source", r"C:\AI\mvadapter-upstream-inspection"))
        if not upstream.is_dir():
            raise RuntimeError(f"MVADAPTER_UPSTREAM_SOURCE_MISSING:{upstream}")
        sys.path.append(str(upstream))
        from lowvram_mvadapter_i2mv_sd21 import (
            attention_report,
            build_low_vram_pipeline,
            component_dtype_inventory,
            component_inventory,
            cond_encoder_device_path_smoke_test,
            install_fp16_input_guards,
            install_low_vram_offload,
            offload_hook_report,
            prepare_reference_cache_fp32,
            prepare_reference_latents_fp32,
            prepare_condition_residuals_fp32,
            reference_unet_dtype_smoke_test,
            rowcol_dtype_inventory,
            tensor_dtype_record,
        )
        adapter_state = __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(preflight["adapter"]), device="cpu")
        # The SD2.1 VAE in float16 returns 100% NaN from both encode and decode on this card, even
        # from a verified-finite latent. NaN survives clamp and casts to 0, which is why the
        # previous attempt completed twenty steps and a decode and still wrote six black PNGs.
        # Promoting the VAE alone to float32 restores finite output; see
        # proof/benchmarks/20260803-mvadapter-vae-fp16-nan-root-cause.json.
        pipe, adapter_report = build_low_vram_pipeline(
            str(preflight["config"]["base_model"]),
            adapter_state,
            preflight["adapter"].name,
            num_views=6,
            dtype=torch.float16,
            vae_dtype=torch.float32,
        )
        receipt["vae_dtype"] = "float32"
        receipt["vae_dtype_reason"] = "float16 VAE is all-NaN on this GPU; proven by decode-boundary diagnostic"
        receipt["adapter_report"] = adapter_report
        # The FP16 convolution verdict for this exact GPU/torch/CUDA/cuDNN stack, and whether the
        # UNet is consequently running with cuDNN disabled.
        receipt["unet_cudnn_guard"] = adapter_report.get("unet_cudnn_guard")
        receipt["convolution_self_test"] = (adapter_report.get("unet_cudnn_guard") or {}).get("self_test")
        _heartbeat(heartbeat_path, receipt, "convolution_self_test_completed",
                   **{key: (receipt["convolution_self_test"] or {}).get(key) for key in (
                       "gpu_name", "compute_capability", "torch_version", "cuda_version",
                       "cudnn_version", "fp16_cudnn_finite_fraction",
                       "fp16_no_cudnn_finite_fraction", "fp32_cudnn_finite_fraction",
                       "max_error_fp16_no_cudnn_vs_fp32", "unet_cudnn_disabled")})
        receipt["component_inventory"] = component_inventory(pipe)
        receipt["attention"] = attention_report(pipe)
        receipt["dtype_inventory"] = {
            "model_components": component_dtype_inventory(pipe),
            "rowcol_processors": rowcol_dtype_inventory(pipe, required_dtype=torch.float16),
        }
        receipt["sdpa_backend"] = _sdpa_backend_report(torch)
        _heartbeat(heartbeat_path, receipt, "pipeline_constructed", processor=receipt["attention"].get("expected_processor"))
        reference = Image.open(preflight["conditioning"]).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(int(selected["seed"]))
        reference_preprocessed = pipe.image_processor.preprocess(reference)
        reference_latents, receipt["reference_vae_fp32"] = prepare_reference_latents_fp32(
            pipe,
            reference_preprocessed,
            generator=generator,
            device="cuda:0",
            requested_dtype=torch.float16,
        )
        reference_prompt_embeds, _ = pipe.encode_prompt(
            receipt["prompt"], torch.device("cpu"), 1, False, None
        )
        reference_cache, receipt["reference_cache_fp32"] = prepare_reference_cache_fp32(
            pipe,
            reference_latents,
            reference_prompt_embeds,
            device="cuda:0",
        )
        control = torch.from_numpy(np.ascontiguousarray(np.load(preflight["controls"], allow_pickle=False).astype(np.float32)))
        control_feature = pipe.prepare_control_image(
            control,
            int(selected["resolution"]),
            int(selected["resolution"]),
            6,
            1,
            torch.device("cuda"),
            torch.float16,
            False,
        ).to(device="cuda", dtype=torch.float16)
        condition_residuals, receipt["condition_encoder_fp32"] = prepare_condition_residuals_fp32(
            pipe,
            control_feature,
            device="cuda:0",
            requested_dtype=torch.float16,
        )
        receipt["reference_unet_pass_started"] = True
        receipt["reference_unet_call_count"] = 1
        _heartbeat(heartbeat_path, receipt, "reference_unet_fp32_cache_completed", cache_entries=len(reference_cache))
        _heartbeat(
            heartbeat_path,
            receipt,
            "reference_latents_fp32_validated",
            tensor=tensor_dtype_record("reference_latents", reference_latents),
        )
        # The tiny dtype smoke must exercise its own 64px input.  Reinstall the
        # production reference-latent override only after that smoke completes.
        pipe.clear_reference_latents_override()
        receipt["offload"] = install_low_vram_offload(pipe, device="cuda")
        receipt["offload_hooks"] = offload_hook_report(pipe)
        _heartbeat(heartbeat_path, receipt, "sequential_offload_installed")
        _heartbeat(heartbeat_path, receipt, "controls_loaded", shape=list(control.shape), dtype=str(control.dtype))
        guard_handles = install_fp16_input_guards(pipe)
        receipt["device_path_smoke_test"] = {
            "mode": "FP32_CONDITION_RESIDUAL_PRECOMPUTE",
            "passed": True,
            "cond_encoder_output_level_count": len(condition_residuals),
            "cond_encoder_output_finite": True,
            "cond_encoder_resident_on_cuda_after": False,
            "cuda_memory_released": True,
            "unet_denoising_called": False,
            "reference_unet_pass_called": False,
            "scheduler_step_called": False,
            "vae_decode_called": False,
            "output_images": 0,
            "gpu_sequence_consumed": False,
            "condition_encoder_fp32": receipt["condition_encoder_fp32"],
        }
        receipt["dtype_inventory"]["cond_encoder_smoke"] = {
            "condition_encoder_fp32": receipt["condition_encoder_fp32"],
        }
        _heartbeat(heartbeat_path, receipt, "cond_encoder_dtype_smoke_passed")
        receipt["reference_unet_dtype_smoke"] = reference_unet_dtype_smoke_test(
            pipe, device="cuda:0", resolution=64
        )
        receipt["dtype_inventory"]["reference_unet_smoke"] = receipt["reference_unet_dtype_smoke"]["tensor_inventory"]
        receipt["reference_cache_summary"] = receipt["reference_unet_dtype_smoke"]["reference_cache_summary"]
        receipt["offload_hooks_after_smoke"] = offload_hook_report(pipe)
        _heartbeat(heartbeat_path, receipt, "reference_unet_dtype_smoke_passed")
        pipe.set_reference_latents_override(reference_latents)
        _heartbeat(heartbeat_path, receipt, "conditioning_loaded", size=list(reference.size))
        state = {"unet_calls": 0, "reference_unet_calls": 0, "denoising_unet_calls": 0}
        original_unet_forward = pipe.unet.forward
        original_cond_forward = pipe.cond_encoder.forward

        def tracked_unet(*args: Any, **kwargs: Any):
            state["unet_calls"] += 1
            if state["unet_calls"] == 1:
                state["reference_unet_calls"] += 1
                receipt["reference_unet_pass_started"] = True
                _heartbeat(heartbeat_path, receipt, "reference_unet_started", call_count=state["reference_unet_calls"])
                cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
                override = getattr(pipe, "_lowvram_reference_cache_override", None)
                if isinstance(cross_attention_kwargs, dict) and override is not None:
                    sink = cross_attention_kwargs.get("cache_hidden_states")
                    if isinstance(sink, dict):
                        sink.update(override)
                        _heartbeat(
                            heartbeat_path,
                            receipt,
                            "reference_cache_injected_explicit_owner",
                            cache_entries=len(override),
                        )
                        sample = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("sample")
                        if isinstance(sample, torch.Tensor):
                            return (torch.zeros_like(sample),)
            else:
                state["denoising_unet_calls"] += 1
                receipt["denoising_started"] = True
            return original_unet_forward(*args, **kwargs)

        def tracked_cond(*args: Any, **kwargs: Any):
            receipt["cond_encoder_executed"] = True
            _heartbeat(heartbeat_path, receipt, "cond_encoder_executed")
            return condition_residuals

        pipe.unet.forward = tracked_unet
        pipe.cond_encoder.forward = tracked_cond

        from mvadapter_latent_telemetry import LatentTelemetry

        telemetry = LatentTelemetry(output_dir / "latent_telemetry",
                                    snapshot_steps=(1, 5, 10, 15, 18, 19, 20))
        receipt["latent_telemetry_dir"] = str(output_dir / "latent_telemetry")

        def on_step_end(_pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            receipt["denoising_steps_completed"] = int(step) + 1
            latents = callback_kwargs.get("latents")
            if int(step) == 0 and isinstance(latents, torch.Tensor):
                receipt["dtype_inventory"]["diffusion_latents"] = tensor_dtype_record(
                    "diffusion_latents", latents
                )
            if isinstance(latents, torch.Tensor):
                telemetry.record_step(int(step) + 1, timestep, latents,
                                      scheduler=getattr(_pipe, "scheduler", None))
            _heartbeat(
                heartbeat_path,
                receipt,
                "denoising_step_completed",
                step=int(step) + 1,
                requested=int(selected["steps"]),
                unet_call_count=state["denoising_unet_calls"],
            )
            return callback_kwargs

        probe = None
        if step_split_probe and finite_gate:
            raise RuntimeError("MVADAPTER_STEP_SPLIT_PROBE_AND_FINITE_GATE_MUTUALLY_EXCLUSIVE")
        if step_split_probe:
            from mvadapter_step_split_probe import StepSplitProbe, UNetModuleProbe

            probe = StepSplitProbe(output_dir / "step_split", target_step=1)
            probe.install(pipe)
            module_probe = UNetModuleProbe()
            module_probe.install(pipe.unet)
            probe.module_probe = module_probe
            receipt["step_split_probe"] = {"installed": True,
                                           "dir": str(output_dir / "step_split")}
        elif finite_gate:
            # Whole-run recording: every step is instrumented, nothing aborts, and the module-level
            # probe stays off because a per-module finite check on every step is far more expensive
            # than the gate is worth once the failing module is already known.
            from mvadapter_step_split_probe import StepSplitProbe

            probe = StepSplitProbe(output_dir / "finite_gate", target_step=None,
                                   save_scheduler_inputs=False, abort_on_first_failure=False)
            probe.install(pipe)
            receipt["finite_gate_dir"] = str(output_dir / "finite_gate")

        torch.cuda.reset_peak_memory_stats()
        receipt["memory_before_denoising"] = {**_nvidia_snapshot(), **_torch_memory(torch)}
        result = pipe(
            prompt=receipt["prompt"],
            negative_prompt=None,
            height=int(selected["resolution"]),
            width=int(selected["resolution"]),
            num_inference_steps=int(selected["steps"]),
            guidance_scale=1.0,
            reference_conditioning_scale=float(selected["reference_conditioning_scale"]),
            control_conditioning_scale=float(selected["control_conditioning_scale"]),
            control_image=control,
            reference_image=reference,
            num_images_per_prompt=6,
            generator=generator,
            output_type="pil",
            callback_on_step_end=on_step_end,
        )
        receipt["reference_unet_call_count"] = state["reference_unet_calls"]
        receipt["denoising_unet_call_count"] = state["denoising_unet_calls"]
        if probe is not None:
            probe.uninstall()
            if finite_gate:
                receipt["finite_gate"] = {
                    **probe.finite_gate(expected_steps=int(selected["steps"])),
                    "report": str(probe.write()),
                }
                _heartbeat(heartbeat_path, receipt, "finite_gate_recorded",
                           passed=receipt["finite_gate"]["passed"],
                           failed=receipt["finite_gate"]["failed_checks"])
            else:
                receipt["step_split_probe"] = {**probe.summary(), "report": str(probe.write())}
        images = list(result.images)
        receipt["vae_decode_completed"] = True
        # Final post-processed image tensor, as the last telemetry checkpoint. This is the stage
        # that previously read as six identical blacks, so it is checked explicitly rather than
        # inferred from the fact that files were written.
        if images:
            stacked = torch.from_numpy(
                np.stack([np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                          for image in images])
            )
            telemetry.record("final_postprocessed_image_tensor", stacked, save=True)
            # Recorded before the production QA gate, which raises on a blank image. The repair
            # proof needs the measured numbers either way, not just the exception.
            receipt["decoded_image_gate"] = _decoded_image_gate(images)
            _heartbeat(heartbeat_path, receipt, "decoded_image_gate_recorded",
                       passed=receipt["decoded_image_gate"]["passed"])
        receipt["latent_telemetry"] = telemetry.summary()
        telemetry.write()
        _heartbeat(heartbeat_path, receipt, "vae_decode_completed")
        receipt["gpu_sequence_consumed"] = True
        if len(images) != 6:
            raise RuntimeError(f"MVADAPTER_OUTPUT_COUNT_INVALID:{len(images)}")
        semantic_names = [str(view.get("proven_semantic", VIEW_NAMES[int(view["index"])])) for view in sorted(preflight["camera"]["views"], key=lambda item: int(item["index"]))]
        receipt["view_semantics"] = semantic_names
        for index, image in enumerate(images):
            path = output_dir / f"view_{index}_{semantic_names[index]}.png"
            image.convert("RGB").save(path)
            receipt["output_images"].append({"index": index, "name": path.name, "path": str(path), "sha256": sha256(path)})
        _contact_sheet(images, output_dir / "six_view_contact_sheet.png", semantic_names)
        _heartbeat(heartbeat_path, receipt, "outputs_written", output_count=len(images))
        receipt["contact_sheet"] = str(output_dir / "six_view_contact_sheet.png")
        receipt["qa"] = qa_outputs(images, Path(preflight["controls"]).parent, int(selected["resolution"]), semantic_names)
        receipt["status"] = "PROVEN" if receipt["qa"]["passed"] else "QA_REJECTED"
        if attempt == "primary":
            # The resolution comes from the config, not from the label: a primary attempt is not
            # necessarily 384, and a consumed 256 config that claims "EXECUTED_384" is a false
            # receipt.
            verdict = "PASSED" if receipt["qa"]["passed"] else "REJECTED"
            config_status = f"EXECUTED_{int(selected['resolution'])}_QA_{verdict}"
        else:
            config_status = "EXECUTED_256_OOM_FALLBACK_QA_PASSED" if receipt["qa"]["passed"] else "EXECUTED_256_OOM_FALLBACK_QA_REJECTED"
        _update_config(config_path, config_status)
    except Exception as exc:
        # The step-split probe deliberately aborts the pipeline at the first non-finite checkpoint,
        # so its findings must be written from the failure path, not only the success path.
        if "probe" in locals() and probe is not None:
            probe.uninstall()
            if finite_gate:
                receipt["finite_gate"] = {
                    **probe.finite_gate(expected_steps=int(selected["steps"])),
                    "report": str(probe.write()),
                }
            else:
                receipt["step_split_probe"] = {**probe.summary(), "report": str(probe.write())}
            receipt["classification"] = probe.classification()
        receipt["reference_unet_call_count"] = state.get("reference_unet_calls", 0) if "state" in locals() else 0
        receipt["denoising_unet_call_count"] = state.get("denoising_unet_calls", 0) if "state" in locals() else 0
        genuine_oom = _is_cuda_oom(exc)
        receipt["cuda_oom"] = "YES" if genuine_oom else "NO"
        if "dtype" in str(exc).lower():
            receipt["classification"] = "DTYPE_RUNTIME_REJECTED"
        if genuine_oom and attempt == "primary":
            receipt["status"] = "CUDA_OOM"
            receipt["fallback_to_256"] = "AUTHORIZED_AFTER_CLEANUP"
            _update_config(config_path, "PRIMARY_384_CUDA_OOM_FALLBACK_AUTHORIZED", consumed=False)
        elif genuine_oom and attempt == "oom-fallback":
            receipt["status"] = "HARD_BLOCKER_CUDA_OOM_AT_256"
            receipt["gpu_sequence_consumed"] = True
        elif receipt["denoising_started"]:
            receipt["status"] = "RUNTIME_REJECTED_SEQUENCE_CONSUMED"
            receipt["gpu_sequence_consumed"] = True
            _update_config(config_path, "RUNTIME_REJECTED_SEQUENCE_CONSUMED")
        else:
            receipt["status"] = "CUDA_OOM" if genuine_oom else "REJECTED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["fallback_eligible"] = bool(attempt == "primary" and genuine_oom)
        if not receipt["gpu_sequence_consumed"]:
            receipt["gpu_sequence_consumed"] = bool(receipt["denoising_started"] and not genuine_oom)
    finally:
        for handle in locals().get("guard_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        if pipe is not None:
            try:
                pipe.clear_reference_latents_override()
                pipe.clear_reference_cache_override()
                pipe.unet.forward = original_unet_forward
                pipe.cond_encoder.forward = original_cond_forward
            except Exception:
                pass
        try:
            del pipe
        except Exception as cleanup_exc:
            receipt.setdefault("cleanup_errors", []).append(f"delete_pipeline: {type(cleanup_exc).__name__}: {cleanup_exc}")
        gc.collect()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                receipt["memory_after"] = {**_nvidia_snapshot(), **_torch_memory(torch)}
        except Exception as cleanup_exc:
            # A poisoned CUDA context must not replace the primary generation
            # failure or erase the scheduler-step evidence already recorded.
            receipt.setdefault("cleanup_errors", []).append(f"cuda_cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}")
        try:
            _heartbeat(heartbeat_path, receipt, "cleanup_complete", **receipt.get("memory_after", {}))
        except Exception as cleanup_exc:
            receipt.setdefault("cleanup_errors", []).append(f"heartbeat_cleanup: {type(cleanup_exc).__name__}: {cleanup_exc}")
        receipt["system_after"] = _system_snapshot()
    receipt["output_count"] = len(receipt["output_images"])
    receipt["wall_seconds"] = round(time.time() - receipt.get("_started", time.time()), 3)
    receipt.pop("_started", None)
    receipt_path = output_dir / "inference_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", choices=("primary", "oom-fallback"), required=True)
    parser.add_argument("--primary-receipt", type=Path, default=None)
    parser.add_argument("--step-split-probe", action="store_true",
                        help="split step 1 into its individual operations and stop at the "
                             "first non-finite checkpoint; diagnostic only")
    parser.add_argument("--finite-gate", action="store_true",
                        help="record every step's scaled input, raw UNet output and scheduler "
                             "output without aborting, and report named finite verdicts")
    args = parser.parse_args()
    started = time.time()
    try:
        receipt = execute(args.config, args.output_dir, args.attempt, args.primary_receipt,
                          step_split_probe=args.step_split_probe, finite_gate=args.finite_gate)
    except Exception as exc:
        receipt = {
            "schema": "lowvram3d_mvadapter_sd21_six_view_inference_v1",
            "status": "PREFLIGHT_REJECTED",
            "error": f"{type(exc).__name__}: {exc}",
            "output_count": 0,
            "gpu_sequence_consumed": False,
            "fallback_eligible": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "inference_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    return 0 if receipt.get("status") == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
