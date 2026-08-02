"""Run MV-Adapter SD2.1 without importing optional raster/texturing stacks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mv_adapter_i2mv_camera_runtime import AZIMUTHS, run_i2mv_pipeline
from run_mv_adapter_fp32_canary import (
    build_contact_sheet,
    inspect_image,
    save_json,
    sha256_file,
    validate_inspections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-file", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def install_fp32_vae_boundaries(pipe, torch_module) -> dict[str, str]:
    """Keep VAE math in FP32 while returning denoising latents in caller dtype.

    MV-Adapter's SD2.1 pipeline converts the reference image to the prompt/UNet
    dtype before calling ``vae.encode`` and passes denoising latents directly to
    ``vae.decode``.  When the VAE is intentionally kept in FP32, both boundaries
    otherwise receive FP16 tensors and fail with a convolution dtype mismatch.

    The wrappers are local to the canary pipeline instance.  Accelerate's model
    offload hooks remain active because the original bound encode/decode methods
    are still called.  The returned reference latents are converted back to the
    dtype requested by the pipeline before they enter the FP16 UNet.
    """

    vae = pipe.vae
    vae.to(dtype=torch_module.float32)

    original_encode = vae.encode
    original_decode = vae.decode
    original_prepare_image_latents = pipe.prepare_image_latents

    def encode_fp32(sample, *args, **kwargs):
        if not isinstance(sample, torch_module.Tensor):
            raise TypeError(f"VAE encode expected a tensor, received {type(sample)!r}")
        sample = sample.to(dtype=torch_module.float32)
        return original_encode(sample, *args, **kwargs)

    def decode_fp32(latents, *args, **kwargs):
        if not isinstance(latents, torch_module.Tensor):
            raise TypeError(f"VAE decode expected a tensor, received {type(latents)!r}")
        latents = latents.to(dtype=torch_module.float32)
        return original_decode(latents, *args, **kwargs)

    def prepare_image_latents_with_boundary(*args, **kwargs):
        requested_dtype = kwargs.get("dtype")
        if requested_dtype is None and len(args) >= 5:
            requested_dtype = args[4]

        latents = original_prepare_image_latents(*args, **kwargs)

        if requested_dtype is not None:
            latents = latents.to(dtype=requested_dtype)

        return latents

    vae.encode = encode_fp32
    vae.decode = decode_fp32
    pipe.prepare_image_latents = prepare_image_latents_with_boundary

    return {
        "vae_parameter_dtype": str(vae.dtype),
        "vae_encode_input_dtype": "torch.float32",
        "vae_decode_input_dtype": "torch.float32",
        "reference_latent_return_dtype": "pipeline_requested_dtype",
    }


def main() -> None:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    base_model = Path(args.base_model).resolve()
    adapter_file = Path(args.adapter_file).resolve()
    source_image = Path(args.source_image).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mv_adapter_fp32_canary_report.json"

    report = {
        "passed": False,
        "status": "INITIALIZING",
        "runtime": "direct_i2mv_without_mvadapter_utils",
        "optional_raster_imported": False,
        "optional_triton_imported": False,
        "texture_projection_started": False,
        "prior_jobs_modified": False,
        "official_repo": str(official_repo),
        "base_model": str(base_model),
        "adapter_file": str(adapter_file),
        "source_image": str(source_image),
        "output_root": str(output_root),
        "settings": {
            "num_views": len(AZIMUTHS),
            "resolution": args.resolution,
            "steps": args.steps,
            "seed": args.seed,
            "scheduler": "ddpm",
            "pipeline_dtype": "float16",
            "non_vae_module_dtype": "float16",
            "vae_dtype": "float32",
            "vae_encode_input_dtype": "float32",
            "vae_decode_input_dtype": "float32",
            "reference_latent_return_dtype": "float16",
            "vae_slicing": True,
            "vae_tiling": True,
            "attention_slicing": "max",
        },
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
        report["adapter_bytes"] = adapter_file.stat().st_size
        if actual_hash != args.expected_adapter_sha256.lower():
            raise RuntimeError(
                "adapter hash mismatch: "
                f"expected={args.expected_adapter_sha256.lower()} actual={actual_hash}"
            )

        from safetensors import safe_open

        with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not keys:
            raise RuntimeError("adapter safetensors contains no tensors")
        report["adapter_tensor_count"] = len(keys)

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

        if "nvdiffrast" in sys.modules or "triton" in sys.modules:
            raise RuntimeError("optional raster/texturing module was imported unexpectedly")
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

        print("DIRECT_I2MV_RUNTIME=true", flush=True)
        print("OPTIONAL_NVDIFFRAST_IMPORTED=false", flush=True)
        print("OPTIONAL_TRITON_IMPORTED=false", flush=True)
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

        # The custom adapter creates new modules after from_pretrained, so cast
        # the complete non-VAE pipeline to the intended FP16 inference dtype
        # before restoring the VAE to its explicit FP32 numerical boundary.
        pipe.to(dtype=torch.float16)
        if hasattr(pipe, "cond_encoder"):
            pipe.cond_encoder.to(dtype=torch.float16)

        unet_dtype = next(pipe.unet.parameters()).dtype
        cond_encoder_dtype = next(pipe.cond_encoder.parameters()).dtype
        if unet_dtype != torch.float16:
            raise RuntimeError(f"UNet dtype is not FP16: {unet_dtype}")
        if cond_encoder_dtype != torch.float16:
            raise RuntimeError(
                f"condition encoder dtype is not FP16: {cond_encoder_dtype}"
            )

        report["module_dtypes_before_offload"] = {
            "unet": str(unet_dtype),
            "condition_encoder": str(cond_encoder_dtype),
        }
        report["vae_dtype_boundaries"] = install_fp32_vae_boundaries(pipe, torch)
        if hasattr(pipe.vae, "config"):
            try:
                pipe.vae.config.force_upcast = True
            except Exception:
                pass
        pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")

        try:
            pipe.enable_model_cpu_offload(gpu_id=0)
            offload_mode = "model_cpu_offload"
        except Exception as model_error:
            report["model_cpu_offload_error"] = repr(model_error)
            pipe.enable_sequential_cpu_offload(gpu_id=0)
            offload_mode = "sequential_cpu_offload"
        report["settings"]["resolved_offload_mode"] = offload_mode

        print(f"OFFLOAD_MODE={offload_mode}", flush=True)
        print(f"UNET_DTYPE={unet_dtype}", flush=True)
        print(f"CONDITION_ENCODER_DTYPE={cond_encoder_dtype}", flush=True)
        print(f"VAE_DTYPE={pipe.vae.dtype}", flush=True)
        print("VAE_ENCODE_INPUT_DTYPE=torch.float32", flush=True)
        print("VAE_DECODE_INPUT_DTYPE=torch.float32", flush=True)
        print("REFERENCE_LATENT_RETURN_DTYPE=torch.float16", flush=True)
        print("STARTING_BOUNDED_DIRECT_CANARY", flush=True)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with torch.inference_mode():
                images, reference = run_i2mv_pipeline(
                    pipe,
                    source_image=source_image,
                    text=(
                        "full body antlered bird shaman, weathered layered cloth, "
                        "wooden staff, hanging ornaments, neutral standing pose, high quality"
                    ),
                    negative_prompt=(
                        "black image, blank image, empty image, duplicate views, "
                        "mirrored front view, watermark, blurry"
                    ),
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
        contact_sheet = output_root / "mv-adapter-fp32-canary-contact-sheet.png"
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
        print("MV_ADAPTER_DIRECT_FP32_CANARY_PASSED", flush=True)
        print("BLACK_OUTPUTS=false", flush=True)
        print("PIXEL_IDENTICAL_OUTPUTS=false", flush=True)
        print(f"PEAK_CUDA_MEMORY_BYTES={report['peak_cuda_memory_bytes']}", flush=True)
        print(f"CONTACT_SHEET={contact_sheet}", flush=True)
        print(f"REPORT={report_path}", flush=True)

    except Exception as error:
        report["passed"] = False
        if report.get("status") == "INITIALIZING":
            report["status"] = "CANARY_EXECUTION_FAILED"
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()
        try:
            import torch

            if torch.cuda.is_available():
                report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        save_json(report_path, report)
        print("MV_ADAPTER_DIRECT_FP32_CANARY_FAILED", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"EXCEPTION={repr(error)}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
