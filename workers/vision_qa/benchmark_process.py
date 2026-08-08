#!/usr/bin/env python3
"""Benchmark any isolated model command and emit a fail-closed receipt.

Peak VRAM is sampled from nvidia-smi total GPU memory usage. The receipt records both
absolute peak and baseline-adjusted delta because the Windows display already uses VRAM.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sample-seconds", type=float, default=0.25)
    parser.add_argument("--vram-ceiling-mib", type=int, default=5600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def gpu_memory_mib() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        return None
    values = [int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")

    baseline = gpu_memory_mib()
    samples: list[int] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(args.sample_seconds):
            value = gpu_memory_mib()
            if value is not None:
                samples.append(value)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
        exit_code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    finally:
        stop.set()
        sampler.join(timeout=2)
    duration = time.perf_counter() - started
    final = gpu_memory_mib()
    for value in (baseline, final):
        if value is not None:
            samples.append(value)
    peak = max(samples) if samples else None
    delta = None if baseline is None or peak is None else max(0, peak - baseline)
    under_ceiling = peak is not None and peak <= args.vram_ceiling_mib
    classification = "PROVEN_PROCESS_ONLY" if exit_code == 0 and under_ceiling else "NOT_PROVEN"
    receipt = {
        "schema": "vision_model_process_benchmark_v1",
        "classification": classification,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "gpu": {
            "baseline_memory_mib": baseline,
            "peak_memory_mib": peak,
            "peak_delta_mib": delta,
            "ceiling_mib": args.vram_ceiling_mib,
            "under_ceiling": under_ceiling,
            "sample_count": len(samples),
        },
        "stdout_tail": stdout[-8000:],
        "stderr_tail": stderr[-8000:],
        "limitations": [
            "Process success and VRAM fit do not prove output quality.",
            "Total GPU memory includes display and unrelated processes.",
            "A separate output validator and visual benchmark are required.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 0 if classification == "PROVEN_PROCESS_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
