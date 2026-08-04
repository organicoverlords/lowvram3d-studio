"""Optional native bridge for the exact UV overlap detector.

The helper is deliberately a subprocess rather than a replacement algorithm.  It receives the
same float64 triangles and returns the same report fields, so the Python implementation remains a
portable fallback and the acceptance semantics stay unchanged.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path


def native_executable() -> Path | None:
    configured = os.environ.get("LOWVRAM3D_UV_OVERLAP_NATIVE")
    candidates = ([Path(configured)] if configured else []) + [
        Path(__file__).resolve().parents[2] / "tools" / "native" / "uv_overlap_native.exe",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def detect_native(triangles, atlas_resolution: int, timeout_seconds: float,
                  max_candidate_pairs: int, collect_pairs: bool):
    executable = native_executable()
    if executable is None:
        return None
    from .uv_overlap import UvOverlapReport

    report = UvOverlapReport()
    with tempfile.TemporaryDirectory(prefix="lowvram3d_uv_native_") as temp_dir:
        root = Path(temp_dir)
        input_path = root / "triangles.bin"
        output_path = root / "report.txt"
        flat = triangles.astype("<f8", copy=False)
        with input_path.open("wb") as handle:
            handle.write(struct.pack("<Q", int(len(flat))))
            handle.write(flat.tobytes(order="C"))
        completed = subprocess.run(
            [str(executable), str(input_path), str(atlas_resolution), str(timeout_seconds),
             str(max_candidate_pairs), "1" if collect_pairs else "0", str(output_path)],
            capture_output=True, text=True,
            timeout=max(30.0, float(timeout_seconds) + 30.0),
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            return None
        values: dict[str, str] = {}
        pairs: list[tuple[int, int]] = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("positive_pair="):
                first, second = line.split("=", 1)[1].split(",")
                pairs.append((int(first), int(second)))
            elif "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        for key in (
            "candidate_pair_count", "tested_pair_count", "positive_overlap_pair_count",
            "degenerate_uv_triangle_count", "out_of_bounds_triangle_count",
            "ignored_noise_intersection_count",
        ):
            setattr(report, key, int(values.get(key, "0")))
        for key in (
            "positive_overlap_total_area_uv", "positive_overlap_max_area_uv",
            "positive_overlap_total_texels_equivalent",
        ):
            setattr(report, key, float(values.get(key, "0")))
        report.timed_out = values.get("timed_out", "1") == "1"
        report.success = values.get("success", "0") == "1"
        if not report.success:
            report.errors.append("native overlap detector failed or timed out")
        report.positive_overlap_pairs = pairs
    return report
