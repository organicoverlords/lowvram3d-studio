"""Run one isolated MoGe-2 inference and write a proof receipt.

Torch, OpenCV and MoGe are imported only inside main so normal pipeline imports
do not reserve GPU memory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from lowvram3d.image_world.image_decode import decode_image_bgr
from lowvram3d.image_world.moge_probe import (
    MogeProbeReport,
    MogeProbeSettings,
    save_moge_maps,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--input-long-edge", type=int, default=768)
    parser.add_argument("--max-gpu-memory-mb", type=int, default=5600)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--lock-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = MogeProbeSettings(
        model=args.model,
        num_tokens=args.num_tokens,
        input_long_edge=args.input_long_edge,
        fp16=not args.fp32,
        max_gpu_memory_mb=args.max_gpu_memory_mb,
        allow_download=args.allow_download,
        allow_cpu=args.allow_cpu,
    )
    settings.validate()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "moge-probe-report.json"
    errors: list[str] = []
    started = time.perf_counter()
    source_hash = sha256_file(source) if source.is_file() else "0" * 64
    peak_allocated = None
    peak_reserved = None
    versions = {"python": platform.python_version(), "platform": platform.platform()}

    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        lock_path = Path(args.lock_path) if args.lock_path else output.parent / ".heavy-gpu.lock"
        with exclusive_lock(lock_path):
            import cv2
            import torch
            from moge.model.v2 import MoGeModel

            versions.update({"torch": torch.__version__, "opencv": cv2.__version__})
            cuda_available = torch.cuda.is_available()
            if not cuda_available and not settings.allow_cpu:
                raise RuntimeError("CUDA is unavailable and --allow-cpu was not supplied")
            device = torch.device("cuda" if cuda_available else "cpu")
            if cuda_available:
                versions["cuda_runtime"] = str(torch.version.cuda)
                versions["gpu"] = torch.cuda.get_device_name(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            decoded = decode_image_bgr(source, cv2)
            versions["image_decoder"] = decoded.decoder
            versions["image_recovered"] = str(decoded.recovered).lower()
            image = prepare_rgb(decoded.bgr_or_bgra, settings.input_long_edge, cv2)
            tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(
                device=device,
                dtype=torch.float32,
            ) / 255.0

            model = MoGeModel.from_pretrained(
                settings.model,
                local_files_only=not settings.allow_download,
            ).to(device)
            model.eval()
            if settings.fp16 and device.type == "cuda":
                model.half()
                tensor = tensor.half()

            infer_kwargs = {}
            if "num_tokens" in inspect.signature(model.infer).parameters:
                infer_kwargs["num_tokens"] = settings.num_tokens
            with torch.inference_mode():
                result = model.infer(tensor, **infer_kwargs)
            if cuda_available:
                torch.cuda.synchronize(device)
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            summary = save_moge_maps(result, output / "geometry")

            if peak_reserved is not None and peak_reserved > settings.max_gpu_memory_mb:
                errors.append(
                    f"peak reserved GPU memory {peak_reserved:.1f} MB exceeded "
                    f"ceiling {settings.max_gpu_memory_mb} MB"
                )
            status = "PASS" if not errors else "FAILED_GPU_MEMORY_CEILING"
            report = MogeProbeReport(
                status=status,
                source_sha256=source_hash,
                settings=settings,
                output=summary,
                wall_time_seconds=time.perf_counter() - started,
                peak_gpu_allocated_mb=peak_allocated,
                peak_gpu_reserved_mb=peak_reserved,
                versions=versions,
                errors=tuple(errors),
            )
            report_path.write_text(report.to_json(), encoding="utf-8")
            return 0 if not errors else 3
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.append(traceback.format_exc())
        report = MogeProbeReport(
            status="FAILED",
            source_sha256=source_hash,
            settings=settings,
            output=None,
            wall_time_seconds=time.perf_counter() - started,
            peak_gpu_allocated_mb=peak_allocated,
            peak_gpu_reserved_mb=peak_reserved,
            versions=versions,
            errors=tuple(errors),
        )
        report_path.write_text(report.to_json(), encoding="utf-8")
        return 2


def prepare_rgb(image: np.ndarray, long_edge: int, cv2_module) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("input image must have three or four channels")
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image = (bgr * alpha + 127.0 * (1.0 - alpha)).astype(np.uint8)
    current_long = max(image.shape[:2])
    if current_long > long_edge:
        scale = long_edge / current_long
        image = cv2_module.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2_module.INTER_AREA,
        )
    return cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"heavy GPU lock already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": time.time()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
