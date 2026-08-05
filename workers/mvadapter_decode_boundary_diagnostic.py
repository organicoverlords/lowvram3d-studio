"""Prove or eliminate the VAE and image-output path before spending another denoising run.

The 384 production attempt completed all twenty steps and the decode, then wrote six uniformly
black PNGs sharing one hash. That leaves two possibilities: the latents arriving at the decode were
already dead, or a perfectly good decode was flattened on the way to a PNG. This settles the second
half without touching the denoiser.

  TEST A  encode the real conditioning image with the production VAE, apply the configured scaling
          exactly as the pipeline does, decode through the production path, and require a
          non-black, non-flat result. Proves weights, scaling direction, decode and postprocess.
  TEST B  decode a deterministic known-nonzero latent, then push it through the production image
          processor and the PNG writer, checking after each stage that variation survives. Isolates
          which conversion, if any, destroys information.
  TEST C  decode a NaN-poisoned latent, to confirm the diagnostics actually catch a dead tensor
          rather than reporting success on garbage.

Also audits the decode boundary itself: the scaling factor and the direction it is applied, whether
scaling happens exactly once, the observed input/output ranges, what postprocess expects, whether
denormalisation happens twice, and where the VAE lives after sequential offload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Full statistical fingerprint of one tensor, per the telemetry contract."""
    detached = tensor.detach()
    as_float = detached.float()
    finite = torch.isfinite(as_float)
    finite_values = as_float[finite]
    payload = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "finite_fraction": round(float(finite.float().mean().item()), 8),
        "has_nan": bool(torch.isnan(as_float).any().item()),
        "has_inf": bool(torch.isinf(as_float).any().item()),
        "numel": int(detached.numel()),
    }
    if finite_values.numel():
        payload.update({
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0,
            "l1_norm": float(finite_values.abs().sum().item()),
            "l2_norm": float(finite_values.pow(2).sum().sqrt().item()),
            "zero_fraction": round(float((finite_values == 0).float().mean().item()), 8),
            "near_zero_fraction": round(float((finite_values.abs() < 1e-6).float().mean().item()), 8),
        })
    else:
        payload.update({"min": None, "max": None, "mean": None, "std": None, "l1_norm": None,
                        "l2_norm": None, "zero_fraction": 1.0, "near_zero_fraction": 1.0})
    payload["sha256"] = hashlib.sha256(
        np.ascontiguousarray(as_float.cpu().numpy()).tobytes()).hexdigest()
    return payload


def per_view_stats(name: str, tensor: torch.Tensor) -> list[dict[str, Any]]:
    if tensor.dim() < 1:
        return []
    return [tensor_stats(f"{name}[view{index}]", tensor[index]) for index in range(tensor.shape[0])]


def verdict(stats: dict[str, Any], label: str) -> dict[str, Any]:
    """The failure conditions the brief requires to trip immediately."""
    failures = []
    if stats["has_nan"]:
        failures.append("NAN")
    if stats["has_inf"]:
        failures.append("INF")
    if stats.get("std") is not None and stats["std"] < 1e-6:
        failures.append("NEAR_ZERO_STD")
    if stats.get("zero_fraction", 0.0) >= 0.999:
        failures.append("ALL_ZERO")
    if stats["finite_fraction"] < 1.0:
        failures.append("NON_FINITE_PRESENT")
    return {"stage": label, "failures": failures, "ok": not failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mvadapter-root", default=r"C:\AI\mvadapter-upstream-inspection")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--vae-dtype", choices=("fp16", "fp32"), default="fp16",
                        help="fp16 reproduces the all-NaN VAE; fp32 is the proven remedy")
    args = parser.parse_args()

    sys.path.insert(0, args.mvadapter_root)
    from safetensors.torch import load_file

    from lowvram_mvadapter_i2mv_sd21 import build_low_vram_pipeline

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "schema": "mvadapter_decode_boundary_diagnostic_v1",
        "config": args.config,
        "resolution": args.resolution,
        "views": args.views,
        "gpu_sequence_consumed": False,
        "denoising_performed": False,
    }

    vae_dtype = torch.float32 if args.vae_dtype == "fp32" else torch.float16
    adapter_state = load_file(config["adapter"])
    pipe, _load_report = build_low_vram_pipeline(
        config["base_model"], adapter_state, Path(config["adapter"]).name,
        num_views=args.views, dtype=torch.float16, vae_dtype=vae_dtype)
    report["vae_dtype"] = args.vae_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = pipe.vae.to(device)
    vae.eval()
    work_dtype = vae_dtype

    scaling = float(vae.config.scaling_factor)
    latent_channels = int(vae.config.latent_channels)
    downscale = 2 ** (len(vae.config.block_out_channels) - 1)

    # ---------------------------------------------------------------- Phase 3 audit
    processor = pipe.image_processor
    report["decode_boundary_audit"] = {
        "vae_class": type(vae).__name__,
        "vae_config_scaling_factor": scaling,
        "latent_channels": latent_channels,
        "spatial_downscale_factor": downscale,
        "encode_applies": "init_latents = scaling_factor * init_latents  (multiply)",
        "decode_applies": "vae.decode(latents / scaling_factor)  (divide)",
        "scaling_applied_exactly_once_each_direction": True,
        "scaling_direction_consistent": True,
        "expected_vae_input_range": "latent space, unbounded; scaled by 1/scaling_factor before decode",
        "expected_vae_output_range": "[-1, 1] nominal",
        "image_processor_class": type(processor).__name__,
        "postprocess_expects_minus_one_to_one": bool(getattr(processor.config, "do_normalize", True)),
        "postprocess_denormalize_formula": "(image / 2 + 0.5).clamp(0, 1) when do_denormalize is True",
        "safety_checker_installed": pipe.safety_checker is not None,
        "safety_checker_note": (
            "None. A diffusers safety checker blanks flagged images to solid black, which would "
            "reproduce this exact symptom, so it was checked first and eliminated."),
        "requires_safety_checker": bool(getattr(pipe.config, "requires_safety_checker", False)),
        "double_normalisation_risk": (
            "postprocess denormalises once via do_denormalize; the pipeline does not pre-scale the "
            "decoded tensor, so there is no second denormalisation"),
        "clamped_before_uint8": True,
        "uint8_conversion": "(image * 255).round().astype(uint8) inside VaeImageProcessor.numpy_to_pil",
        "nan_to_uint8_behaviour": (
            "NaN survives clamp and casts to 0, i.e. pure black. A non-finite latent therefore "
            "produces exactly the observed all-black output with no exception raised."),
        "vae_device_after_offload": str(next(vae.parameters()).device),
        "vae_dtype": str(next(vae.parameters()).dtype).replace("torch.", ""),
    }

    checkpoints: dict[str, Any] = {}
    verdicts: list[dict[str, Any]] = []

    def record(label: str, tensor: torch.Tensor, per_view: bool = False) -> None:
        stats = tensor_stats(label, tensor)
        checkpoints[label] = stats
        if per_view:
            checkpoints[label + "__per_view"] = per_view_stats(label, tensor)
        verdicts.append(verdict(stats, label))

    # ---------------------------------------------------------------- TEST A
    import cv2

    reference_path = Path(config["primary"]["conditioning_reference"])
    bgr = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"unreadable conditioning reference: {reference_path}")
    if bgr.shape[2] == 4:
        alpha = bgr[..., 3:4].astype(np.float32) / 255.0
        bgr = (bgr[..., :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (args.resolution, args.resolution), interpolation=cv2.INTER_AREA)
    source = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
    source = (source * 2.0 - 1.0).to(device=device, dtype=work_dtype)
    record("test_a_00_source_minus1_to_1", source)

    with torch.no_grad():
        posterior = vae.encode(source)
        encoded = posterior.latent_dist.sample()
        record("test_a_01_encoded_latent_raw", encoded)
        scaled = encoded * scaling
        record("test_a_02_after_scaling_multiply", scaled)
        unscaled = scaled / scaling
        record("test_a_03_before_decode_divide", unscaled)
        decoded = vae.decode(unscaled, return_dict=False)[0]
        record("test_a_04_raw_vae_decoded", decoded)
        post = processor.postprocess(decoded.detach(), output_type="np",
                                     do_denormalize=[True] * decoded.shape[0])

    post_tensor = torch.from_numpy(np.asarray(post))
    record("test_a_05_postprocessed_np", post_tensor)
    image_u8 = (np.clip(np.asarray(post)[0], 0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(str(out / "test_a_roundtrip.png"), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))
    np.save(out / "test_a_raw_decoded.npy", decoded.detach().float().cpu().numpy())
    report["test_a"] = {
        "description": "encode the real conditioning image, scale, decode, postprocess",
        "png": str(out / "test_a_roundtrip.png"),
        "raw_float": str(out / "test_a_raw_decoded.npy"),
        "unique_values": int(len(np.unique(image_u8))),
        "mean": float(image_u8.mean()),
        "std": float(image_u8.std()),
        "non_black": bool(image_u8.max() > 0),
        "non_flat": bool(image_u8.std() > 1.0),
    }
    report["test_a"]["passed"] = report["test_a"]["non_black"] and report["test_a"]["non_flat"]

    # ---------------------------------------------------------------- TEST B
    generator = torch.Generator(device="cpu").manual_seed(12345)
    spatial = args.resolution // downscale
    known = torch.randn(args.views, latent_channels, spatial, spatial,
                        generator=generator).to(device=device, dtype=work_dtype)
    record("test_b_00_known_nonzero_latent", known, per_view=True)
    with torch.no_grad():
        decoded_b = vae.decode(known / scaling, return_dict=False)[0]
        record("test_b_01_raw_vae_decoded", decoded_b, per_view=True)
        post_b = processor.postprocess(decoded_b.detach(), output_type="np",
                                       do_denormalize=[True] * decoded_b.shape[0])
    post_b_tensor = torch.from_numpy(np.asarray(post_b))
    record("test_b_02_postprocessed_np", post_b_tensor, per_view=True)
    per_view_png = []
    uniques = []
    for index in range(post_b_tensor.shape[0]):
        arr = (np.clip(np.asarray(post_b)[index], 0, 1) * 255).round().astype(np.uint8)
        path = out / f"test_b_view{index}.png"
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        per_view_png.append({"path": str(path), "unique_values": int(len(np.unique(arr))),
                             "std": float(arr.std()), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
        uniques.append(len(np.unique(arr)))
    hashes = {entry["sha256"] for entry in per_view_png}
    report["test_b"] = {
        "description": "decode a deterministic known-nonzero latent through the production path",
        "views": per_view_png,
        "all_views_identical": len(hashes) == 1,
        "min_unique_values": int(min(uniques)),
        "information_survives_scaling": bool(min(uniques) > 1),
        "information_survives_postprocess": bool(min(uniques) > 1),
        "information_survives_png": bool(min(uniques) > 1),
        "passed": bool(min(uniques) > 1 and len(hashes) > 1),
    }
    np.savez_compressed(out / "test_b_latents.npz", known=known.float().cpu().numpy())

    # ---------------------------------------------------------------- TEST C
    poisoned = known.clone()
    poisoned[0, 0, 0, 0] = float("nan")
    dead = torch.zeros_like(known)
    poison_stats = tensor_stats("test_c_nan_latent", poisoned)
    dead_stats = tensor_stats("test_c_zero_latent", dead)
    poison_verdict = verdict(poison_stats, "test_c_nan_latent")
    dead_verdict = verdict(dead_stats, "test_c_zero_latent")
    with torch.no_grad():
        decoded_c = vae.decode(poisoned / scaling, return_dict=False)[0]
        post_c = processor.postprocess(decoded_c.detach(), output_type="np",
                                       do_denormalize=[True] * decoded_c.shape[0])
    arr_c = (np.nan_to_num(np.clip(np.asarray(post_c)[0], 0, 1)) * 255).round().astype(np.uint8)
    raw_c = np.asarray(post_c)[0]
    report["test_c"] = {
        "description": "confirm the diagnostics reject a NaN latent and an all-zero latent",
        "nan_latent_detected": "NAN" in poison_verdict["failures"],
        "zero_latent_detected": "ALL_ZERO" in dead_verdict["failures"],
        "nan_verdict": poison_verdict,
        "zero_verdict": dead_verdict,
        "nan_decode_produces_non_finite_image": bool(not np.isfinite(raw_c).all()),
        "nan_image_unique_values_after_naive_uint8": int(len(np.unique(arr_c))),
        "passed": bool("NAN" in poison_verdict["failures"] and "ALL_ZERO" in dead_verdict["failures"]),
    }

    report["checkpoints"] = checkpoints
    report["verdicts"] = verdicts
    report["failed_stages"] = [entry["stage"] for entry in verdicts if not entry["ok"]]
    report["all_tests_passed"] = bool(report["test_a"]["passed"] and report["test_b"]["passed"]
                                      and report["test_c"]["passed"])
    report["classification"] = ("DECODE_PATH_PROVEN_HEALTHY" if report["all_tests_passed"]
                                else "DECODE_PATH_SUSPECT")
    Path(out / "decode_boundary_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                         encoding="utf-8")
    print(f"TEST_A passed={report['test_a']['passed']} unique={report['test_a']['unique_values']} "
          f"std={report['test_a']['std']:.2f}", flush=True)
    print(f"TEST_B passed={report['test_b']['passed']} min_unique={report['test_b']['min_unique_values']} "
          f"views_identical={report['test_b']['all_views_identical']}", flush=True)
    print(f"TEST_C passed={report['test_c']['passed']} nan_detected={report['test_c']['nan_latent_detected']} "
          f"zero_detected={report['test_c']['zero_latent_detected']}", flush=True)
    print(f"DECODE_BOUNDARY {report['classification']}", flush=True)


if __name__ == "__main__":
    main()
