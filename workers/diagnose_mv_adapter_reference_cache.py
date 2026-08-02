"""Diagnose MV-Adapter reference-cache propagation without denoising or export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mv_adapter_i2mv_camera_runtime import AZIMUTHS, run_i2mv_pipeline
from run_mv_adapter_fp32_canary import save_json, sha256_file
from run_mv_adapter_fp32_canary_direct import (
    count_custom_mv_attention_processors,
    install_fp32_vae_boundaries,
    install_low_vram_offload,
)


class ReferenceCacheDiagnosticComplete(RuntimeError):
    """Raised before the first denoising UNet computation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-file", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def processor_inventory(pipe: Any) -> list[dict[str, Any]]:
    inventory = []
    for registry_name, processor in pipe.unet.attn_processors.items():
        inventory.append(
            {
                "registry_name": registry_name,
                "processor_name": getattr(processor, "name", None),
                "class": type(processor).__name__,
                "use_mv": bool(getattr(processor, "use_mv", False)),
                "use_ref": bool(getattr(processor, "use_ref", False)),
            }
        )
    return inventory


def main() -> None:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    base_model = Path(args.base_model).resolve()
    adapter_file = Path(args.adapter_file).resolve()
    source_image = Path(args.source_image).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "passed": False,
        "status": "INITIALIZING",
        "diagnostic_only": True,
        "denoising_started": False,
        "image_generation_completed": False,
        "texture_projection_started": False,
        "candidate_promoted": False,
        "official_repo": str(official_repo),
        "base_model": str(base_model),
        "adapter_file": str(adapter_file),
        "source_image": str(source_image),
        "resolution": args.resolution,
        "seed": args.seed,
    }

    try:
        for label, path in (("official repository", official_repo), ("base model", base_model)):
            if not path.is_dir():
                raise RuntimeError(f"{label} directory is missing: {path}")
        for label, path in (("adapter weight", adapter_file), ("source image", source_image)):
            if not path.is_file():
                raise RuntimeError(f"{label} is missing: {path}")

        actual_hash = sha256_file(adapter_file)
        report["adapter_sha256"] = actual_hash
        if actual_hash != args.expected_adapter_sha256.lower():
            raise RuntimeError(
                "adapter hash mismatch: "
                f"expected={args.expected_adapter_sha256.lower()} actual={actual_hash}"
            )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        sys.path.insert(0, str(official_repo))

        import torch
        from diffusers import DDPMScheduler
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pipe = MVAdapterI2MVSDPipeline.from_pretrained(
            str(base_model),
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
            safety_checker=None,
        )
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDPMScheduler,
        )
        pipe.init_custom_adapter(num_views=len(AZIMUTHS))
        pipe.load_custom_adapter(str(adapter_file.parent), weight_name=adapter_file.name)
        pipe.to(dtype=torch.float16)
        pipe.cond_encoder.to(dtype=torch.float16)

        inventory = processor_inventory(pipe)
        expected_reference_names = sorted(
            {
                item["processor_name"]
                for item in inventory
                if item["use_ref"] and item["processor_name"] is not None
            }
        )
        report["processor_inventory"] = inventory
        report["processor_count"] = len(inventory)
        report["custom_processor_count"] = count_custom_mv_attention_processors(pipe)
        report["expected_reference_names"] = expected_reference_names
        report["expected_reference_count"] = len(expected_reference_names)

        report["vae_boundaries"] = install_fp32_vae_boundaries(pipe, torch)
        pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        original_forward = pipe.unet.forward
        call_number = 0

        def diagnostic_forward(*forward_args, **forward_kwargs):
            nonlocal call_number
            call_number += 1
            cross_kwargs = forward_kwargs.get("cross_attention_kwargs") or {}

            if "cache_hidden_states" in cross_kwargs:
                cache = cross_kwargs["cache_hidden_states"]
                report["reference_unet_call_number"] = call_number
                report["reference_cache_object_id"] = id(cache)
                report["reference_cache_keys_before"] = sorted(str(key) for key in cache.keys())
                result = original_forward(*forward_args, **forward_kwargs)
                keys_after = sorted(str(key) for key in cache.keys())
                report["reference_cache_keys_after"] = keys_after
                report["reference_cache_count_after"] = len(keys_after)
                report["reference_cache_tensor_shapes"] = {
                    str(key): list(value.shape)
                    for key, value in cache.items()
                    if isinstance(value, torch.Tensor)
                }
                return result

            if "ref_hidden_states" in cross_kwargs:
                cache = cross_kwargs["ref_hidden_states"]
                actual_names = sorted(str(key) for key in cache.keys())
                missing_names = sorted(set(expected_reference_names) - set(actual_names))
                unexpected_names = sorted(set(actual_names) - set(expected_reference_names))
                report["denoise_unet_call_number"] = call_number
                report["denoise_reference_cache_object_id"] = id(cache)
                report["denoise_reference_keys"] = actual_names
                report["denoise_reference_count"] = len(actual_names)
                report["missing_reference_names"] = missing_names
                report["unexpected_reference_names"] = unexpected_names
                report["first_missing_reference_name"] = missing_names[0] if missing_names else None
                report["denoising_started"] = False
                raise ReferenceCacheDiagnosticComplete("reference cache captured before denoising")

            return original_forward(*forward_args, **forward_kwargs)

        pipe.unet.forward = diagnostic_forward
        report["offload"] = install_low_vram_offload(pipe, torch)

        try:
            run_i2mv_pipeline(
                pipe,
                source_image=source_image,
                text="full body antlered bird shaman, neutral standing pose",
                negative_prompt="black image, blank image, duplicate views",
                height=args.resolution,
                width=args.resolution,
                steps=1,
                seed=args.seed,
                device="cuda",
            )
        except ReferenceCacheDiagnosticComplete:
            pass
        else:
            raise RuntimeError("diagnostic did not intercept the first denoising UNet call")

        cached_after = report.get("reference_cache_keys_after", [])
        denoise_keys = report.get("denoise_reference_keys", [])
        missing_names = report.get("missing_reference_names", [])

        report["cache_survived_pipeline_transformation"] = cached_after == denoise_keys
        report["reference_cache_complete"] = len(missing_names) == 0
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        report["passed"] = True
        report["status"] = (
            "REFERENCE_CACHE_COMPLETE"
            if report["reference_cache_complete"]
            else "REFERENCE_CACHE_INCOMPLETE_PROVEN"
        )
        save_json(output, report)

        print("MV_ADAPTER_REFERENCE_CACHE_DIAGNOSTIC_COMPLETE", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"EXPECTED_REFERENCE_COUNT={len(expected_reference_names)}", flush=True)
        print(f"REFERENCE_CACHE_COUNT_AFTER={len(cached_after)}", flush=True)
        print(f"DENOISE_REFERENCE_COUNT={len(denoise_keys)}", flush=True)
        print(f"MISSING_REFERENCE_COUNT={len(missing_names)}", flush=True)
        print(f"FIRST_MISSING_REFERENCE_NAME={report.get('first_missing_reference_name')}", flush=True)
        print(f"CACHE_SURVIVED_PIPELINE_TRANSFORMATION={report['cache_survived_pipeline_transformation']}", flush=True)
        print("DENOISING_STARTED=false", flush=True)
        print("TEXTURE_PROJECTION_STARTED=false", flush=True)
        print(f"REPORT={output}", flush=True)

    except Exception as error:
        report["passed"] = False
        report["status"] = "DIAGNOSTIC_FAILED"
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()
        try:
            import torch

            if torch.cuda.is_available():
                report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        save_json(output, report)
        print("MV_ADAPTER_REFERENCE_CACHE_DIAGNOSTIC_FAILED", flush=True)
        print(f"EXCEPTION={repr(error)}", flush=True)
        print(f"REPORT={output}", flush=True)
        raise


if __name__ == "__main__":
    main()
