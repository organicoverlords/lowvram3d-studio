"""Run a bounded MV-Adapter SD2.1 image-to-multiview numerical canary.

This worker is intentionally separate from the texture pipeline.  It proves only
that the local MV-Adapter installation can produce six finite, non-constant,
non-identical images on constrained hardware.  It never writes into a prior job,
never promotes a texture, and rejects black/blank/duplicate outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import warnings
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

AZIMUTHS = (0, 45, 90, 180, 270, 315)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(image: Image.Image, index: int) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    sampled = rgb.reshape(-1, 3)
    if len(sampled) > 150_000:
        positions = np.linspace(0, len(sampled) - 1, 150_000, dtype=np.int64)
        sampled = sampled[positions]
    return {
        "index": index,
        "width": image.width,
        "height": image.height,
        "minimum": int(rgb.min()),
        "maximum": int(rgb.max()),
        "mean": float(rgb.mean()),
        "standard_deviation": float(rgb.std()),
        "dynamic_range": int(rgb.max()) - int(rgb.min()),
        "sampled_unique_colours": int(len(np.unique(sampled, axis=0))),
        "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }


def validate_inspections(inspections: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for entry in inspections:
        index = int(entry["index"])
        if float(entry["standard_deviation"]) < 1.0:
            failures.append(f"VIEW_{index}_NEAR_CONSTANT")
        if int(entry["dynamic_range"]) < 12:
            failures.append(f"VIEW_{index}_LOW_DYNAMIC_RANGE")
        if int(entry["sampled_unique_colours"]) < 64:
            failures.append(f"VIEW_{index}_INSUFFICIENT_COLOUR_DIVERSITY")
    hashes = [str(entry["pixel_sha256"]) for entry in inspections]
    if len(set(hashes)) != len(hashes):
        failures.append("PIXEL_IDENTICAL_OUTPUT_VIEWS")
    return sorted(set(failures))


def build_contact_sheet(images: list[Image.Image], destination: Path) -> None:
    tile = 384
    heading = 44
    columns = 3
    rows = 2
    sheet = Image.new(
        "RGB",
        (columns * tile, rows * (tile + heading)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        preview = image.convert("RGB")
        preview.thumbnail((tile, tile))
        row = index // columns
        column = index % columns
        x = column * tile + (tile - preview.width) // 2
        y = row * (tile + heading) + heading
        sheet.paste(preview, (x, y))
        draw.text(
            (column * tile + 10, row * (tile + heading) + 13),
            f"view {index} / azimuth {AZIMUTHS[index]}",
            fill=(245, 245, 245),
        )
    sheet.save(destination)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    base_model = Path(args.base_model).resolve()
    adapter_file = Path(args.adapter_file).resolve()
    source_image = Path(args.source_image).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mv_adapter_fp32_canary_report.json"

    report: dict[str, Any] = {
        "passed": False,
        "status": "INITIALIZING",
        "official_repo": str(official_repo),
        "base_model": str(base_model),
        "adapter_file": str(adapter_file),
        "source_image": str(source_image),
        "output_root": str(output_root),
        "texture_projection_started": False,
        "prior_jobs_modified": False,
        "settings": {
            "num_views": 6,
            "resolution": args.resolution,
            "steps": args.steps,
            "seed": args.seed,
            "scheduler": "ddpm",
            "pipeline_dtype": "float16",
            "vae_dtype": "float32",
            "vae_slicing": True,
            "vae_tiling": True,
            "attention_slicing": "max",
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
        report["adapter_sha256"] = actual_adapter_hash
        report["adapter_bytes"] = adapter_file.stat().st_size
        if actual_adapter_hash != args.expected_adapter_sha256.lower():
            raise RuntimeError(
                "adapter hash mismatch: "
                f"expected={args.expected_adapter_sha256.lower()} "
                f"actual={actual_adapter_hash}"
            )

        from safetensors import safe_open

        with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
            tensor_keys = list(handle.keys())
        if not tensor_keys:
            raise RuntimeError("adapter safetensors contains no tensors")
        report["adapter_tensor_count"] = len(tensor_keys)
        report["adapter_first_keys"] = tensor_keys[:20]

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
        from scripts import inference_i2mv_sd as official_i2mv

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
        pipe.init_custom_adapter(num_views=6)
        pipe.load_custom_adapter(
            str(adapter_file.parent),
            weight_name=adapter_file.name,
        )

        pipe.vae.to(dtype=torch.float32)
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
        print(f"VAE_DTYPE={pipe.vae.dtype}", flush=True)
        print("STARTING_BOUNDED_CANARY", flush=True)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with torch.inference_mode():
                images, reference = official_i2mv.run_pipeline(
                    pipe,
                    num_views=6,
                    text=(
                        "full body antlered bird shaman, weathered layered "
                        "cloth, wooden staff, hanging ornaments, neutral "
                        "standing pose, high quality"
                    ),
                    image=str(source_image),
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.steps,
                    guidance_scale=3.0,
                    seed=args.seed,
                    remove_bg_fn=None,
                    reference_conditioning_scale=1.0,
                    negative_prompt=(
                        "black image, blank image, empty image, duplicate "
                        "views, mirrored front view, watermark, blurry"
                    ),
                    lora_scale=1.0,
                    device="cuda",
                )
            captured_warnings = [
                {
                    "category": record.category.__name__,
                    "message": str(record.message),
                }
                for record in records
            ]
        report["warnings"] = captured_warnings

        if len(images) != 6:
            raise RuntimeError(f"expected six generated views, received {len(images)}")

        views_dir = output_root / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        inspections: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            destination = views_dir / f"view_{index:02d}_azimuth_{AZIMUTHS[index]:03d}.png"
            image.convert("RGB").save(destination)
            inspection = inspect_image(image, index)
            inspection["path"] = str(destination)
            inspections.append(inspection)
            print(
                f"VIEW_{index} std={inspection['standard_deviation']:.6f} "
                f"range={inspection['dynamic_range']} "
                f"unique={inspection['sampled_unique_colours']} "
                f"hash={inspection['pixel_sha256']}",
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
        print("MV_ADAPTER_FP32_VAE_CANARY_PASSED", flush=True)
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
        print("MV_ADAPTER_FP32_VAE_CANARY_FAILED", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"EXCEPTION={repr(error)}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        raise


if __name__ == "__main__":
    main()
