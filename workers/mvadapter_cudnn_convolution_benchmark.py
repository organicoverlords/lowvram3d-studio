"""Cost of running the MV-Adapter UNet's convolutions without cuDNN.

The repair disables cuDNN for the UNet, so the obvious question is what that costs. It cannot be
answered end to end on this machine: the cuDNN-enabled FP16 path returns NaN, so there is no
correct baseline run to compare a wall time against. What can be measured is the convolution
itself, at the two latent sizes production actually uses, which is where the whole cost lives.

Run with the MV-Adapter environment's interpreter; it needs only torch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

#: 256px and 384px images become 32x32 and 48x48 latents through the VAE's 8x downsample.
LATENT_SIZES = ((256, 32), (384, 48))
CHANNELS = 320
WARMUP = 5
ITERATIONS = 50


def _time(convolution: torch.nn.Module, sample: torch.Tensor, cudnn: bool,
          iterations: int = ITERATIONS) -> dict[str, Any]:
    backend = torch.backends.cudnn
    with backend.flags(enabled=cudnn, benchmark=backend.benchmark,
                       deterministic=backend.deterministic, allow_tf32=backend.allow_tf32):
        with torch.no_grad():
            for _ in range(WARMUP):
                output = convolution(sample)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                output = convolution(sample)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
    return {
        "cudnn": cudnn,
        "iterations": iterations,
        "total_seconds": round(elapsed, 6),
        "milliseconds_per_call": round(elapsed / iterations * 1000.0, 4),
        "finite_fraction": round(float(torch.isfinite(output.float()).float().mean().item()), 8),
    }


def benchmark(views: int = 6) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("MVADAPTER_CONVOLUTION_BENCHMARK_REQUIRES_CUDA")
    backend = torch.backends.cudnn
    results = []
    for resolution, latent in LATENT_SIZES:
        convolution = torch.nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1).to(
            device="cuda", dtype=torch.float16)
        sample = torch.randn((views, CHANNELS, latent, latent), device="cuda", dtype=torch.float16)
        enabled = _time(convolution, sample, cudnn=True)
        disabled = _time(convolution, sample, cudnn=False)
        results.append({
            "resolution": resolution,
            "latent": latent,
            "batch": views,
            "cudnn_enabled": enabled,
            "cudnn_disabled": disabled,
            "slowdown_factor": round(
                disabled["milliseconds_per_call"] / enabled["milliseconds_per_call"], 4),
        })
        del convolution, sample
        torch.cuda.empty_cache()
    return {
        "schema": "mvadapter_cudnn_convolution_benchmark_v1",
        "module": "down_blocks.0.resnets.0.conv1",
        "dtype": "float16",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": backend.version(),
        "note": ("No end-to-end cuDNN-enabled baseline exists on this machine: that path returns "
                 "non-finite output, so there is no correct run to time it against."),
        "measurements": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = benchmark()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
