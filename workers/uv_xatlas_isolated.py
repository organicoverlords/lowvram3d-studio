"""Bounded parent for isolated, checkpointed xatlas presets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PRESET_ORDER = ("A", "B", "C")


def choose_next(attempted: list[str]) -> str | None:
    return next((name for name in PRESET_ORDER if name not in attempted), None)


def run_one(python: str, child: Path, source: Path, directory: Path, preset: str,
            resolution: int, padding: int, timeout: float) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"preset_{preset}_valid.glb"
    checkpoint = directory / f"preset_{preset}_checkpoint.glb"
    report = directory / f"preset_{preset}.json"
    command = [python, str(child), "--input", str(source), "--output", str(output),
               "--checkpoint", str(checkpoint), "--report", str(report), "--preset", preset,
               "--resolution", str(resolution), "--padding", str(padding)]
    started = time.monotonic()
    try:
        env = dict(os.environ)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, _ = proc.communicate(timeout=10)
            child_report = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
            # This return belongs to the timeout handler. At the outer indent it ran on every
            # call, which made the two statements below unreachable and left `child_report`
            # unbound on the success path - an UnboundLocalError the outer handler then reported
            # as "failed". The xatlas route could therefore never succeed, and the UV_OVERLAP
            # repair recipe that selects it could never have worked.
            return {"preset": preset, "status": "timed_out",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "command": command,
                    "last_operation": child_report.get("last_operation"),
                    "candidate_written": bool(child_report.get("candidate_written")),
                    "report": str(report), "output": str(output),
                    "last_output": (stdout or "")[-500:]}
        child_report = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        return {"preset": preset, "status": child_report.get("status", "failed"),
                "command": command,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "last_operation": child_report.get("last_operation"),
                "candidate_written": bool(child_report.get("candidate_written")),
                "child_returncode": proc.returncode, "report": str(report),
                "output": str(output), "last_output": (stdout or "")[-500:]}
    except Exception as exc:
        return {"preset": preset, "status": "failed", "error": repr(exc), "command": command,
                "elapsed_seconds": round(time.monotonic() - started, 3), "report": str(report)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    source = Path(args.input)
    root = Path(args.report).parent
    child = Path(__file__).with_name("uv_xatlas_candidate.py")
    receipt = {"status": "running", "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "resolution": args.resolution, "padding": args.padding, "timeout_seconds": args.timeout,
               "preset_policy": "A first; B only after A invalid; C only after A and B invalid",
               "presets": []}
    attempted: list[str] = []
    for _ in PRESET_ORDER:
        preset = choose_next(attempted)
        if preset is None:
            break
        attempted.append(preset)
        result = run_one(args.python, child, source, root, preset, args.resolution, args.padding, args.timeout)
        receipt["presets"].append(result)
        if result.get("status") == "passed" and Path(result.get("output", "")).is_file():
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result["output"], destination)
            receipt.update({"status": "passed", "selected": preset, "output": str(destination),
                            "last_operation": "promote_valid_candidate"})
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            return 0
    receipt.update({"status": "failed", "selected": None, "last_operation": "no_valid_preset"})
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
