"""Fresh-process reference-image/VAE numerical localization probe.

This worker stops after deterministic reference latents. It never runs a
UNet, scheduler, decoder, quality sequence, or production-manifest update.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(torch: Any, name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"name": name, "tensor": False, "type": type(value).__name__}
    record = {
        "name": name,
        "tensor": True,
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
        "numel": int(value.numel()),
    }
    try:
        finite_mask = torch.isfinite(value)
        finite_count = int(finite_mask.sum().item())
        positive_inf = int((torch.isinf(value) & (value > 0)).sum().item())
        negative_inf = int((torch.isinf(value) & (value < 0)).sum().item())
        nan_count = int(torch.isnan(value).sum().item())
        record.update({
            "finite": bool(finite_mask.all()),
            "finite_count": finite_count,
            "nan_count": nan_count,
            "positive_inf_count": positive_inf,
            "negative_inf_count": negative_inf,
            "stats_error": None,
        })
        if finite_count:
            finite_values = value[finite_mask]
            record["minimum_finite_value"] = float(finite_values.min().item())
            record["maximum_finite_value"] = float(finite_values.max().item())
            record["absolute_maximum"] = float(finite_values.abs().max().item())
        else:
            record["minimum_finite_value"] = None
            record["maximum_finite_value"] = None
            record["absolute_maximum"] = None
    except BaseException as exc:
        record.update({
            "finite": None,
            "finite_count": None,
            "nan_count": None,
            "positive_inf_count": None,
            "negative_inf_count": None,
            "minimum_finite_value": None,
            "maximum_finite_value": None,
            "absolute_maximum": None,
            "stats_error": f"{type(exc).__name__}: {exc}",
        })
    return record


class Trace:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, phase: str, **fields: Any) -> None:
        record = {"time": time.time(), "pid": os.getpid(), "phase": phase, **fields}
        try:
            self.handle.write(json.dumps(record, default=str) + "\n")
            self.handle.flush()
        except (OSError, UnicodeError, ValueError, BrokenPipeError):
            pass

    def close(self) -> None:
        try:
            self.handle.flush()
            self.handle.close()
        except (OSError, UnicodeError, ValueError, BrokenPipeError):
            pass


class FirstNonFinite(RuntimeError):
    pass


def _memory(torch: Any) -> dict[str, Any]:
    try:
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 3),
        }
    except BaseException as exc:
        return {"stats_error": f"{type(exc).__name__}: {exc}"}


def _validate_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = config["primary"]
    conditioning = Path(selected["conditioning_reference"])
    if not conditioning.is_file():
        raise RuntimeError(f"REFERENCE_CONDITIONING_MISSING:{conditioning}")
    actual = sha256(conditioning)
    if actual.lower() != selected["conditioning_reference_sha256"].lower():
        raise RuntimeError(f"REFERENCE_CONDITIONING_HASH_MISMATCH:{actual}")
    return {"config": config, "selected": selected, "conditioning": conditioning, "conditioning_sha256": actual}


def _record_module_hook(torch: Any, trace: Trace, records: list[dict[str, Any]], name: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output
        if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
            value = value[0]
        record = tensor_record(torch, name, value)
        records.append(record)
        trace.write("vae_module_output", module=name, record=record, memory=_memory(torch))
        if record.get("tensor") and record.get("finite") is not True:
            raise FirstNonFinite(
                f"FIRST_NONFINITE_MODULE={name};INPUT_DTYPE={record.get('dtype')};OUTPUT_DTYPE={record.get('dtype')};"
                f"OUTPUT_ABS_MAX={record.get('absolute_maximum')}"
            )
    return hook


def run_case(config_path: Path, output_dir: Path, case: str) -> dict[str, Any]:
    if case not in {"real-fp16", "gray-fp16", "real-fp32", "gray-fp32"}:
        raise ValueError(f"REFERENCE_CASE_INVALID:{case}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"REFERENCE_OUTPUT_NOT_EMPTY:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace(output_dir / "vae_trace.jsonl")
    receipt: dict[str, Any] = {
        "schema": "lowvram3d_mvadapter_reference_numeric_diagnostic_v1",
        "case": case,
        "production_manifest_updated": False,
        "scheduler_steps": 0,
        "images": 0,
        "started": time.time(),
    }
    pipe = None
    handles: list[Any] = []
    pending: BaseException | None = None
    try:
        inputs = _validate_config(config_path)
        receipt["conditioning_sha256"] = inputs["conditioning_sha256"]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("REFERENCE_DIAGNOSTIC_CUDA_UNAVAILABLE")
        target = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        upstream = Path(inputs["config"].get("mvadapter_source", r"C:\AI\mvadapter-upstream-inspection"))
        sys.path.append(str(upstream))
        import safetensors.torch
        from lowvram_mvadapter_i2mv_sd21 import build_low_vram_pipeline
        adapter = Path(inputs["config"]["adapter"])
        adapter_state = safetensors.torch.load_file(str(adapter), device="cpu")
        pipe, adapter_report = build_low_vram_pipeline(
            inputs["config"]["base_model"], adapter_state, adapter.name, num_views=6, dtype=torch.float16
        )
        receipt["adapter_report"] = adapter_report
        use_fp32 = case.endswith("fp32")
        dtype = torch.float32 if use_fp32 else torch.float16
        pipe.vae.to(device=target, dtype=dtype)
        receipt["vae_dtype"] = str(dtype).replace("torch.", "")
        source = Image.open(inputs["conditioning"]).convert("RGB")
        image = source if case.startswith("real") else Image.new("RGB", source.size, (128, 128, 128))
        receipt["input_image"] = {"mode": image.mode, "size": list(image.size), "kind": "real" if case.startswith("real") else "neutral_gray"}
        prepared = pipe.image_processor.preprocess(image)
        prepared = prepared.to(device=target, dtype=dtype)
        receipt["preprocessed_reference"] = tensor_record(torch, "preprocessed_reference_image", prepared)
        trace.write("preprocessed_reference", record=receipt["preprocessed_reference"], memory=_memory(torch))
        if receipt["preprocessed_reference"].get("finite") is not True:
            raise FirstNonFinite("FIRST_NONFINITE_STAGE=preprocessed_reference")
        records: list[dict[str, Any]] = []
        for name, module in pipe.vae.encoder.named_modules():
            if name:
                handles.append(module.register_forward_hook(_record_module_hook(torch, trace, records, f"vae.encoder.{name}")))
        handles.append(pipe.vae.quant_conv.register_forward_hook(_record_module_hook(torch, trace, records, "vae.quant_conv")))
        trace.write("vae_encode_started", dtype=receipt["vae_dtype"], memory=_memory(torch))
        with torch.no_grad():
            encoded = pipe.vae.encode(prepared)
        trace.write("vae_encode_completed", memory=_memory(torch))
        distribution = encoded.latent_dist
        mean = tensor_record(torch, "latent_distribution_mean", distribution.mean)
        logvar = tensor_record(torch, "latent_distribution_logvar", distribution.logvar)
        receipt["latent_distribution_mean"] = mean
        receipt["latent_distribution_logvar"] = logvar
        trace.write("latent_distribution", mean=mean, logvar=logvar, memory=_memory(torch))
        if mean.get("finite") is not True or logvar.get("finite") is not True:
            raise FirstNonFinite("FIRST_NONFINITE_STAGE=latent_distribution")
        with torch.no_grad():
            mode = distribution.mode()
        mode_record = tensor_record(torch, "latent_mode", mode)
        receipt["latent_mode"] = mode_record
        trace.write("latent_mode", record=mode_record, memory=_memory(torch))
        if mode_record.get("finite") is not True:
            raise FirstNonFinite("FIRST_NONFINITE_STAGE=latent_mode")
        scaled = mode * float(pipe.vae.config.scaling_factor)
        scaled_record = tensor_record(torch, "final_reference_latents", scaled)
        receipt["scaling_factor"] = float(pipe.vae.config.scaling_factor)
        receipt["final_reference_latents"] = scaled_record
        trace.write("final_reference_latents", scaling_factor=receipt["scaling_factor"], record=scaled_record, memory=_memory(torch))
        if scaled_record.get("finite") is not True:
            raise FirstNonFinite("FIRST_NONFINITE_STAGE=scaling_factor_application")
        receipt["status"] = "PROVEN"
        receipt["classification"] = "REFERENCE_LATENTS_FINITE"
    except BaseException as exc:
        pending = exc
        receipt["status"] = "REJECTED"
        receipt["classification"] = f"{type(exc).__name__}: {exc}"
        receipt["error"] = receipt["classification"]
        trace.write("case_failed", error=receipt["classification"], memory=_memory(__import__("torch")) if "torch" in sys.modules else {})
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        trace.write("cleanup", memory=_memory(__import__("torch")) if "torch" in sys.modules else {})
        trace.close()
        if pipe is not None:
            del pipe
        gc.collect()
        if "torch" in sys.modules and __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    torch_module = __import__("torch") if "torch" in sys.modules else None
    receipt["memory_final"] = _memory(torch_module) if torch_module is not None else {}
    receipt["wall_seconds"] = round(time.time() - receipt["started"], 3)
    receipt.pop("started", None)
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    if pending is not None:
        raise pending
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", required=True, choices=("real-fp16", "gray-fp16", "real-fp32", "gray-fp32"))
    args = parser.parse_args()
    try:
        receipt = run_case(args.config, args.output_dir, args.case)
        print(json.dumps({"status": receipt["status"], "classification": receipt["classification"], "output_dir": str(args.output_dir)}))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "output_dir": str(args.output_dir)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
