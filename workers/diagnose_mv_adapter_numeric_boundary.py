"""Locate the first nonfinite tensor boundary in the SD2.1 I2MV route.

This reproduces the proven failing run exactly - same weights, adapter,
scheduler, seed, resolution, step count, six-view camera order, reference-cache
relay, custom MV attention, condition-encoder CPU offload and FP32 VAE
boundaries - and adds read-only statistics probes at every tensor boundary
between the reference latents and the decoded image.

Nothing here changes numerics: every probe records and re-emits the tensor it
was given. The run fails closed at the first nonfinite tensor, after the report
has been written.

Exit codes:
    0  a verdict was produced (nonfinite boundary located, or all probes finite)
    1  the diagnostic itself failed and produced no verdict
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
import warnings
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mv_adapter_i2mv_camera_runtime import (
    AZIMUTHS,
    install_reference_cache_relay,
    run_i2mv_pipeline,
)
from mv_adapter_numeric_probe import (
    FirstNonfiniteTensor,
    NumericProbe,
)
from run_mv_adapter_fp32_canary import inspect_image, save_json, sha256_file
from run_mv_adapter_fp32_canary_direct import (
    count_custom_mv_attention_processors,
    install_fp32_vae_boundaries,
    install_low_vram_offload,
)


REQUIRED_BOUNDARIES = (
    "reference_latents",
    "control_image_prepared",
    "adapter_state",
    "initial_noise_latents",
    "unet_noise_pred",
    "scheduler_latents",
    "final_pre_decode_latents",
    "vae_decode_input",
    "vae_decode_output",
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
    # Locked to the proven failing configuration.
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def install_numeric_probes(pipe: Any, probe: NumericProbe) -> dict[str, Any]:
    """Wrap every tensor boundary of interest without altering any value.

    Must be called after the FP32 VAE boundaries, the low-VRAM offload and the
    reference-cache relay are installed, so the probes observe exactly the
    tensors the proven failing run produced.
    """

    import torch

    original_prepare_latents = pipe.prepare_latents
    original_prepare_image_latents = pipe.prepare_image_latents
    original_prepare_control_image = pipe.prepare_control_image
    original_scheduler_step = pipe.scheduler.step
    original_vae_decode = pipe.vae.decode

    counters = {"unet_denoise": 0, "scheduler": 0, "decode": 0, "cond_encoder": 0}
    last_scheduler_output: dict[str, Any] = {}

    def prepare_latents_probe(*args, **kwargs):
        latents = original_prepare_latents(*args, **kwargs)
        probe.record("initial_noise_latents", latents)
        return latents

    def prepare_image_latents_probe(*args, **kwargs):
        latents = original_prepare_image_latents(*args, **kwargs)
        probe.record("reference_latents", latents)
        return latents

    def prepare_control_image_probe(*args, **kwargs):
        control = original_prepare_control_image(*args, **kwargs)
        probe.record("control_image_prepared", control)
        return control

    # Condition encoder: Accelerate's cpu_offload wrapper sits around forward,
    # so probe inside it exactly as the reference-cache relay does for the UNet.
    cond_encoder = pipe.cond_encoder
    cond_hook = getattr(cond_encoder, "_hf_hook", None)
    cond_old_forward = getattr(cond_encoder, "_old_forward", None)
    if cond_hook is not None and callable(cond_old_forward):
        cond_inner_forward = cond_old_forward
        cond_placement = "inside_accelerate_hook"
    else:
        cond_inner_forward = cond_encoder.forward
        cond_placement = "direct_forward"

    def cond_encoder_probe(*args, **kwargs):
        counters["cond_encoder"] += 1
        call_index = counters["cond_encoder"] - 1
        if args and isinstance(args[0], torch.Tensor):
            probe.record(
                "condition_encoder_input", args[0], cond_encoder_call=call_index
            )
        states = cond_inner_forward(*args, **kwargs)
        sequence = states if isinstance(states, (list, tuple)) else [states]
        for index, state in enumerate(sequence):
            if isinstance(state, torch.Tensor):
                probe.record(
                    f"adapter_state_{index:02d}",
                    state,
                    cond_encoder_call=call_index,
                    adapter_state_index=index,
                )
        return states

    # UNet: the relay already owns `_old_forward` inside the Accelerate hook.
    # Layer the probe outside the relay so the relay's cache injection is
    # preserved untouched and the probe sees the final noise prediction.
    unet = pipe.unet
    unet_hook = getattr(unet, "_hf_hook", None)
    unet_old_forward = getattr(unet, "_old_forward", None)
    if unet_hook is not None and callable(unet_old_forward):
        unet_inner_forward = unet_old_forward
        unet_placement = "outside_relay_inside_accelerate_hook"
    else:
        unet_inner_forward = unet.forward
        unet_placement = "outside_relay_direct_forward"

    def unet_probe(*args, **kwargs):
        cross_kwargs = kwargs.get("cross_attention_kwargs")
        is_reference_pass = (
            isinstance(cross_kwargs, dict) and "cache_hidden_states" in cross_kwargs
        )
        result = unet_inner_forward(*args, **kwargs)

        sample = result[0] if isinstance(result, (list, tuple)) else getattr(result, "sample", None)
        if not isinstance(sample, torch.Tensor):
            return result

        if is_reference_pass:
            probe.record("reference_unet_output", sample)
        else:
            step_index = counters["unet_denoise"]
            counters["unet_denoise"] += 1
            probe.record(
                f"unet_noise_pred_step_{step_index:02d}", sample, step=step_index
            )
        return result

    def scheduler_step_probe(*args, **kwargs):
        output = original_scheduler_step(*args, **kwargs)
        step_index = counters["scheduler"]
        counters["scheduler"] += 1

        if isinstance(output, (list, tuple)):
            prev_sample = output[0]
        else:
            prev_sample = getattr(output, "prev_sample", None)

        if isinstance(prev_sample, torch.Tensor):
            last_scheduler_output["latents"] = prev_sample
            probe.record(
                f"scheduler_latents_step_{step_index:02d}", prev_sample, step=step_index
            )
        return output

    def vae_decode_probe(latents, *args, **kwargs):
        counters["decode"] += 1
        if counters["decode"] == 1 and isinstance(
            last_scheduler_output.get("latents"), torch.Tensor
        ):
            probe.record("final_pre_decode_latents", last_scheduler_output["latents"])

        if isinstance(latents, torch.Tensor):
            probe.record("vae_decode_input", latents, decode_call=counters["decode"] - 1)

        decoded = original_vae_decode(latents, *args, **kwargs)
        sample = (
            decoded[0]
            if isinstance(decoded, (list, tuple))
            else getattr(decoded, "sample", None)
        )
        if isinstance(sample, torch.Tensor):
            probe.record(
                "vae_decode_output", sample, decode_call=counters["decode"] - 1
            )
        return decoded

    pipe.prepare_latents = prepare_latents_probe
    pipe.prepare_image_latents = prepare_image_latents_probe
    pipe.prepare_control_image = prepare_control_image_probe
    pipe.scheduler.step = scheduler_step_probe
    pipe.vae.decode = vae_decode_probe

    if cond_placement == "inside_accelerate_hook":
        cond_encoder._old_forward = cond_encoder_probe
    else:
        cond_encoder.forward = cond_encoder_probe

    if unet_placement == "outside_relay_inside_accelerate_hook":
        unet._old_forward = unet_probe
    else:
        unet.forward = unet_probe

    return {
        "condition_encoder_probe_placement": cond_placement,
        "unet_probe_placement": unet_placement,
        "probes_installed": [
            "prepare_latents",
            "prepare_image_latents",
            "prepare_control_image",
            "cond_encoder_forward",
            "unet_forward",
            "scheduler_step",
            "vae_decode",
        ],
    }


def missing_required_boundaries(labels: list[str]) -> list[str]:
    return [
        required
        for required in REQUIRED_BOUNDARIES
        if not any(label.startswith(required) for label in labels)
    ]


def main() -> int:
    args = parse_args()
    official_repo = Path(args.official_repo).resolve()
    base_model = Path(args.base_model).resolve()
    adapter_file = Path(args.adapter_file).resolve()
    source_image = Path(args.source_image).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "mv_adapter_numeric_boundary_report.json"

    probe = NumericProbe()
    report: dict[str, Any] = {
        "diagnostic_complete": False,
        "status": "INITIALIZING",
        "diagnostic": "first_nonfinite_tensor_boundary",
        "diagnostic_only": True,
        "texture_projection_started": False,
        "prior_black_output_directory_modified": False,
        "candidate_promoted": False,
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
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
        },
        "required_boundaries": list(REQUIRED_BOUNDARIES),
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

        actual_hash = sha256_file(adapter_file)
        report["adapter_sha256"] = actual_hash
        if actual_hash != args.expected_adapter_sha256.lower():
            raise RuntimeError(
                "adapter hash mismatch: "
                f"expected={args.expected_adapter_sha256.lower()} actual={actual_hash}"
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

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print("NUMERIC_BOUNDARY_DIAGNOSTIC=true", flush=True)
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
        pipe.cond_encoder.to(dtype=torch.float16)

        custom_attention_count = count_custom_mv_attention_processors(pipe)
        if custom_attention_count <= 0:
            raise RuntimeError("custom MV attention processors were not installed")
        report["custom_mv_attention_processor_count"] = custom_attention_count

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

        relay_state = install_reference_cache_relay(pipe)
        report["reference_cache_relay_placement"] = relay_state["placement"]
        report["reference_cache_relay_expected_count"] = relay_state[
            "expected_reference_count"
        ]

        report["probe_installation"] = install_numeric_probes(pipe, probe)
        print(
            f"UNET_PROBE_PLACEMENT={report['probe_installation']['unet_probe_placement']}",
            flush=True,
        )
        print(
            "CONDITION_ENCODER_PROBE_PLACEMENT="
            f"{report['probe_installation']['condition_encoder_probe_placement']}",
            flush=True,
        )
        print("STARTING_BOUNDED_NUMERIC_BOUNDARY_DIAGNOSTIC", flush=True)

        images: list[Any] = []
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            try:
                with torch.inference_mode():
                    images, _reference = run_i2mv_pipeline(
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
            except FirstNonfiniteTensor as gate:
                print(f"FIRST_NONFINITE_TENSOR={gate.label}", flush=True)
            report["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in records
            ]

        summary = probe.summary()
        report.update(summary)
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

        for entry in summary["records"]:
            statistics = entry["statistics"]
            print(
                f"PROBE {entry['order']:02d} {entry['label']} "
                f"shape={statistics['shape']} dtype={statistics['dtype']} "
                f"finite={statistics['finite_count']}/{statistics['finite_count'] + statistics['nonfinite_count']} "
                f"abs_max={statistics['absolute_maximum']} std={statistics['standard_deviation']}",
                flush=True,
            )

        if summary["nonfinite_boundary_found"]:
            report["status"] = "FIRST_NONFINITE_TENSOR_LOCATED"
            report["missing_required_boundaries"] = []
            report["boundaries_not_reached_due_to_fail_closed"] = (
                missing_required_boundaries(summary["probed_labels"])
            )
        else:
            missing = missing_required_boundaries(summary["probed_labels"])
            report["missing_required_boundaries"] = missing
            if missing:
                report["status"] = "DIAGNOSTIC_INCOMPLETE_BOUNDARY_NOT_PROBED"
                save_json(report_path, report)
                raise RuntimeError(
                    "required boundaries were never probed: " + ", ".join(missing)
                )
            report["status"] = "ALL_PROBED_TENSORS_FINITE"

            views_dir = output_root / "views"
            views_dir.mkdir(parents=True, exist_ok=True)
            inspections = []
            for index, image in enumerate(images):
                destination = (
                    views_dir / f"view_{index:02d}_azimuth_{AZIMUTHS[index]:03d}.png"
                )
                image.convert("RGB").save(destination)
                entry = inspect_image(image, index)
                entry["path"] = str(destination)
                inspections.append(entry)
            report["views"] = inspections

        report["diagnostic_complete"] = True
        save_json(report_path, report)

        print("MV_ADAPTER_NUMERIC_BOUNDARY_DIAGNOSTIC_COMPLETE", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"NONFINITE_BOUNDARY_FOUND={summary['nonfinite_boundary_found']}", flush=True)
        print(f"FIRST_NONFINITE_LABEL={summary['first_nonfinite_label']}", flush=True)
        print(f"BOUNDARY_CATEGORY={summary['boundary_category']}", flush=True)
        print(f"DECISION={summary['decision']}", flush=True)
        print(f"PROBE_RECORD_COUNT={summary['probe_record_count']}", flush=True)
        print(f"PEAK_CUDA_MEMORY_BYTES={report['peak_cuda_memory_bytes']}", flush=True)
        print("TEXTURE_PROJECTION_STARTED=false", flush=True)
        print(f"REPORT={report_path}", flush=True)
        return 0

    except Exception as error:
        if report.get("status") in (None, "INITIALIZING"):
            report["status"] = "DIAGNOSTIC_FAILED"
        report["diagnostic_complete"] = False
        report["exception"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report.setdefault("records", probe.records)
        report.setdefault("first_nonfinite_label", probe.first_nonfinite_label)
        try:
            import torch

            if torch.cuda.is_available():
                report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        save_json(report_path, report)
        print("MV_ADAPTER_NUMERIC_BOUNDARY_DIAGNOSTIC_FAILED", flush=True)
        print(f"STATUS={report['status']}", flush=True)
        print(f"EXCEPTION={repr(error)}", flush=True)
        print(f"REPORT={report_path}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
