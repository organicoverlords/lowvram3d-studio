"""Corrected MV-Adapter SD2.1 I2MV canary with an FP32 condition encoder.

Identical to the proven direct canary in every respect except one: the dynamic
``T2IAdapter`` condition encoder runs in FP32 and its residuals are validated
before being cast back to the UNet latent dtype. The UNet, FP32 VAE boundaries,
reference-cache relay, scheduler, weights, seed, camera order, view count and
prompts are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import traceback
import warnings
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mv_adapter_condition_encoder_fp32 import (
    ConditionEncoderPrecisionError,
    install_fp32_condition_encoder_boundary,
    prepare_fp32_condition_encoder,
)
from mv_adapter_i2mv_camera_runtime import AZIMUTHS, run_i2mv_pipeline
from run_mv_adapter_fp32_canary import (
    build_contact_sheet,
    inspect_image,
    save_json,
    sha256_file,
    validate_inspections,
)
from run_mv_adapter_fp32_canary_direct import (
    count_custom_mv_attention_processors,
    install_fp32_vae_boundaries,
    install_low_vram_offload,
)


PROMPT = (
    "full body antlered bird shaman, weathered layered cloth, "
    "wooden staff, hanging ornaments, neutral standing pose, high quality"
)
NEGATIVE_PROMPT = (
    "black image, blank image, empty image, duplicate views, "
    "mirrored front view, watermark, blurry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-file", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    # Locked to the proven failing configuration.
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def condition_encoder_checkpoint_coverage(
    cond_encoder: Any, adapter_file: Path
) -> dict[str, Any]:
    """Record how much of the condition encoder the adapter checkpoint supplies.

    ``_load_custom_adapter`` loads the checkpoint with ``strict=False``, so an
    architecture or key-naming mismatch would silently leave the encoder at its
    random initialisation. That distinction decides between
    ``BLOCKED_CONDITION_ENCODER_WEIGHTS`` and a precision-only fault, so it is
    recorded as evidence rather than inferred.
    """

    from safetensors import safe_open

    with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())

    encoder_keys = set(cond_encoder.state_dict().keys())
    matched = sorted(encoder_keys & checkpoint_keys)
    missing = sorted(encoder_keys - checkpoint_keys)
    return {
        "condition_encoder_key_count": len(encoder_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "matched_key_count": len(matched),
        "missing_key_count": len(missing),
        "coverage_ratio": (len(matched) / len(encoder_keys)) if encoder_keys else None,
        "example_missing_keys": missing[:8],
    }


def main() -> int:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    base_model = Path(args.base_model).resolve()
    adapter_file = Path(args.adapter_file).resolve()
    source_image = Path(args.source_image).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mv_adapter_cond_fp32_canary_report.json"

    report: dict[str, Any] = {
        "passed": False,
        "status": "INITIALIZING",
        "correction": "condition_encoder_fp32_only",
        "runtime": "direct_i2mv_without_mvadapter_utils",
        "texture_projection_started": False,
        "candidate_promoted": False,
        "prior_black_output_directory_modified": False,
        "official_repo": str(official_repo),
        "base_model": str(base_model),
        "adapter_file": str(adapter_file),
        "source_image": str(source_image),
        "output_root": str(output_root),
        "settings": {
            "num_views": len(AZIMUTHS),
            "camera_azimuths": list(AZIMUTHS),
            "resolution": args.resolution,
            "steps": args.steps,
            "seed": args.seed,
            "guidance_scale": 3.0,
            "scheduler": "ddpm_shift_snr_interpolated_8.0",
            "pipeline_dtype": "float16",
            "vae_dtype": "float32",
            "condition_encoder_dtype": "float32",
            "control_normalisation": "official_do_normalize_false_unchanged",
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
        },
    }

    try:
        for label, path in (
            ("official repository", official_repo),
            ("base model", base_model),
        ):
            if not path.is_dir():
                raise RuntimeError(f"{label} directory is missing: {path}")
        for label, path in (
            ("adapter weight", adapter_file),
            ("source image", source_image),
        ):
            if not path.is_file():
                raise RuntimeError(f"{label} is missing: {path}")

        actual_adapter_hash = sha256_file(adapter_file)
        actual_source_hash = sha256_file(source_image)
        report["adapter_sha256"] = actual_adapter_hash
        report["source_sha256"] = actual_source_hash
        if actual_adapter_hash != args.expected_adapter_sha256.lower():
            raise RuntimeError(
                "adapter hash mismatch: "
                f"expected={args.expected_adapter_sha256.lower()} "
                f"actual={actual_adapter_hash}"
            )
        if actual_source_hash != args.expected_source_sha256.lower():
            raise RuntimeError(
                "source-image hash mismatch: "
                f"expected={args.expected_source_sha256.lower()} "
                f"actual={actual_source_hash}"
            )

        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128",
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        sys.path.insert(0, str(official_repo))

        import torch
        from diffusers import DDPMScheduler
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import (
            MVAdapterI2MVSDPipeline,
        )
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the MV-Adapter environment")

        gpu = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_vram_bytes": int(gpu.total_memory),
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print("CONDITION_ENCODER_FP32_CORRECTION=true", flush=True)
        print("LOADING_SD21_PIPELINE_ON_CPU", flush=True)

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

        report["condition_encoder_checkpoint_coverage"] = (
            condition_encoder_checkpoint_coverage(pipe.cond_encoder, adapter_file)
        )
        coverage = report["condition_encoder_checkpoint_coverage"]
        print(
            "CONDITION_ENCODER_CHECKPOINT_COVERAGE="
            f"{coverage['matched_key_count']}/{coverage['condition_encoder_key_count']}",
            flush=True,
        )

        custom_attention_count = count_custom_mv_attention_processors(pipe)
        if custom_attention_count <= 0:
            raise RuntimeError("custom MV attention processors were not installed")
        report["custom_mv_attention_processor_count"] = custom_attention_count

        unet_dtype = next(pipe.unet.parameters()).dtype
        if unet_dtype != torch.float16:
            raise RuntimeError(f"UNet dtype is not FP16: {unet_dtype}")
        report["unet_dtype"] = str(unet_dtype)

        # Cast the condition encoder to FP32 BEFORE the offload hook exists, so
        # the hook's weights map holds the FP32 tensors it restores per forward.
        report["condition_encoder_fp32_preparation"] = prepare_fp32_condition_encoder(
            pipe, torch
        )
        print(
            "CONDITION_ENCODER_WEIGHTS_FINITE="
            f"{report['condition_encoder_fp32_preparation']['weights_finite_before_cast']}",
            flush=True,
        )
        print(
            "CONDITION_ENCODER_DTYPES_AFTER_CAST="
            f"{report['condition_encoder_fp32_preparation']['floating_dtypes_after']}",
            flush=True,
        )

        report["vae_dtype_boundaries"] = install_fp32_vae_boundaries(pipe, torch)
        if hasattr(pipe.vae, "config"):
            try:
                pipe.vae.config.force_upcast = True
            except Exception:
                pass
        pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        report["offload"] = install_low_vram_offload(pipe, torch)

        cond_encoder_dtype = next(pipe.cond_encoder.parameters()).dtype
        if cond_encoder_dtype != torch.float32:
            raise RuntimeError(
                f"condition encoder is not FP32 after offload: {cond_encoder_dtype}"
            )
        report["condition_encoder_dtype_after_offload"] = str(cond_encoder_dtype)

        boundary_state = install_fp32_condition_encoder_boundary(
            pipe, torch, target_dtype=torch.float16
        )
        report["condition_encoder_boundary_placement"] = boundary_state["placement"]
        if boundary_state["placement"] != "inside_accelerate_hook":
            raise RuntimeError(
                "FP32 condition-encoder boundary did not land inside the "
                f"Accelerate offload hook: {boundary_state['placement']}"
            )

        print(f"OFFLOAD_MODE={report['offload']['mode']}", flush=True)
        print(f"UNET_HOOK={report['offload']['unet_hook']}", flush=True)
        print(
            f"CONDITION_ENCODER_HOOK={report['offload']['condition_encoder_hook']}",
            flush=True,
        )
        print(
            f"CONDITION_ENCODER_BOUNDARY={boundary_state['placement']}", flush=True
        )
        print(f"CONDITION_ENCODER_DTYPE={cond_encoder_dtype}", flush=True)
        print(f"CUSTOM_MV_ATTENTION_PROCESSORS={custom_attention_count}", flush=True)
        print(f"VAE_DTYPE={pipe.vae.dtype}", flush=True)
        print("STARTING_CORRECTED_COND_FP32_CANARY", flush=True)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with torch.inference_mode():
                images, reference = run_i2mv_pipeline(
                    pipe,
                    source_image=source_image,
                    text=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=args.resolution,
                    width=args.resolution,
                    steps=args.steps,
                    seed=args.seed,
                    device="cuda",
                )
            captured_warnings = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in records
            ]
        report["warnings"] = captured_warnings

        report["condition_encoder_runtime"] = {
            "call_count": boundary_state["call_count"],
            "encoder_input_dtype": boundary_state["encoder_input_dtype"],
            "encoder_compute_dtype": boundary_state["encoder_compute_dtype"],
            "resolved_target_dtype": boundary_state["resolved_target_dtype"],
            "input_statistics": boundary_state["input_statistics"],
            "adapter_state_records": boundary_state["adapter_state_records"],
            "converted_dtypes": boundary_state["converted_dtypes"],
        }

        if boundary_state["call_count"] < 1:
            raise RuntimeError("the FP32 condition-encoder boundary never ran")
        if not boundary_state["adapter_state_records"]:
            raise RuntimeError("no adapter states were recorded")

        def _format(value: Any) -> str:
            return "None" if value is None else f"{value:.6g}"

        for record in boundary_state["adapter_state_records"]:
            statistics = record["statistics"]
            print(
                f"COND_FP32 {record['label']} shape={statistics['shape']} "
                f"finite={statistics['finite_count']}/"
                f"{statistics['finite_count'] + statistics['nonfinite_count']} "
                f"min={_format(statistics['minimum'])} "
                f"max={_format(statistics['maximum'])} "
                f"mean={_format(statistics['mean'])} "
                f"std={_format(statistics['standard_deviation'])} "
                f"abs_max={_format(statistics['absolute_maximum'])} "
                f"cast={record['cast_dtype']}",
                flush=True,
            )

        sliced_processor_warnings = [
            item
            for item in captured_warnings
            if "not expected by SlicedAttnProcessor" in item["message"]
        ]
        if sliced_processor_warnings:
            raise RuntimeError("custom MV attention was replaced by SlicedAttnProcessor")

        if len(images) != len(AZIMUTHS):
            raise RuntimeError(f"expected six generated views, received {len(images)}")

        views_dir = output_root / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        inspections = []
        for index, image in enumerate(images):
            destination = views_dir / f"view_{index:02d}_azimuth_{AZIMUTHS[index]:03d}.png"
            image.convert("RGB").save(destination)
            entry = inspect_image(image, index)
            entry["path"] = str(destination)
            inspections.append(entry)
            print(
                f"VIEW_{index} std={entry['standard_deviation']:.6f} "
                f"range={entry['dynamic_range']} unique={entry['sampled_unique_colours']} "
                f"hash={entry['pixel_sha256']}",
                flush=True,
            )

        reference_path = output_root / "reference_preprocessed.png"
        reference.convert("RGB").save(reference_path)
        contact_sheet = output_root / "mv-adapter-cond-fp32-canary-contact-sheet.png"
        build_contact_sheet(images, contact_sheet)

        failures = validate_inspections(inspections)
        if any(
            "invalid value encountered in cast" in item["message"].lower()
            for item in captured_warnings
        ):
            failures.append("NONFINITE_POSTPROCESS_WARNING")
        failures = sorted(set(failures))

        report["views"] = inspections
        report["reference_preprocessed"] = str(reference_path)
        report["contact_sheet"] = str(contact_sheet)
        report["failure_codes"] = failures
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

        if failures:
            report["status"] = "REJECTED_NUMERICALLY_INVALID_OUTPUT"
            save_json(report_path, report)
            raise RuntimeError("canary validation failed: " + ", ".join(failures))

        report["passed"] = True
        report["status"] = "VALID_NONBLACK_CANARY_REQUIRES_VISUAL_REVIEW"
        save_json(report_path, report)
        print("MV_ADAPTER_COND_FP32_CANARY_PASSED", flush=True)
        print("BLACK_OUTPUTS=false", flush=True)
        print("PIXEL_IDENTICAL_OUTPUTS=false", flush=True)
        print(f"PEAK_CUDA_MEMORY_BYTES={report['peak_cuda_memory_bytes']}", flush=True)
        print("TEXTURE_PROJECTION_STARTED=false", flush=True)
        print("CANDIDATE_PROMOTED=false", flush=True)
        print(f"CONTACT_SHEET={contact_sheet}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        return 0

    except Exception as error:
        report["passed"] = False
        if isinstance(error, ConditionEncoderPrecisionError):
            report["status"] = "CONDITION_ENCODER_FP32_OUTPUT_REJECTED"
            report["condition_encoder_blocked"] = True
        elif report.get("status") in (None, "INITIALIZING"):
            report["status"] = "CANARY_EXECUTION_FAILED"
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()

        boundary = locals().get("boundary_state")
        if isinstance(boundary, dict):
            report.setdefault(
                "condition_encoder_runtime",
                {
                    "call_count": boundary["call_count"],
                    "encoder_input_dtype": boundary["encoder_input_dtype"],
                    "encoder_compute_dtype": boundary["encoder_compute_dtype"],
                    "input_statistics": boundary["input_statistics"],
                    "adapter_state_records": boundary["adapter_state_records"],
                },
            )
        try:
            import torch

            if torch.cuda.is_available():
                report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        save_json(report_path, report)
        print("MV_ADAPTER_COND_FP32_CANARY_FAILED", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"EXCEPTION={repr(error)}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
