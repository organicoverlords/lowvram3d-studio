"""Run exactly one xatlas preset and checkpoint its raw candidate.

This is intentionally a child-process boundary.  The parent can terminate a slow or wedged
unwrap without losing the durable candidate/checkpoint receipt or accidentally promoting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import xatlas

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers"))

from lowvram3d.uv_overlap import positive_area_uv_overlaps  # noqa: E402
from lowvram3d.uv_quality import (  # noqa: E402
    MAX_CHART_COUNT, MAX_STRETCH_P95, MAX_TINY_CHART_SURFACE_PERCENT,
    MIN_ATLAS_UTILIZATION, PRESETS, UvCandidateMetrics, area_weighted_percentile,
    conformal_stretch,
)
from uv_exact_validate import atlas_utilisation  # noqa: E402
from uv_xatlas_route import (  # noqa: E402
    chart_count_from_topology, load_indexed, tiny_chart_surface_percent, weld, write_glb,
)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preset", choices=[p.name for p in PRESETS], required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--overlap-timeout", type=float, default=600.0)
    args = parser.parse_args()
    started = time.monotonic()
    source = Path(args.input)
    checkpoint = Path(args.checkpoint)
    report_path = Path(args.report)
    preset = next(p for p in PRESETS if p.name == args.preset)
    report = {
        "status": "running", "preset": preset.name, "input": str(source),
        "resolution": args.resolution, "padding": args.padding,
        "bruteForce": False, "last_operation": "load_input", "candidate_written": False,
    }

    def write_report() -> None:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    write_report()

    def heartbeat() -> None:
        while report.get("status") == "running":
            time.sleep(5.0)
            if report.get("status") == "running":
                write_report()

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        positions_raw, normals_raw, indices_raw = load_indexed(source)
        positions, indices, first = weld(positions_raw, indices_raw)
        normals = normals_raw[first] if normals_raw is not None else None
        report["input_geometry"] = {"vertices": int(len(positions)), "triangles": int(len(indices))}
        report["last_operation"] = "atlas_add_mesh"
        atlas = xatlas.Atlas()
        atlas.add_mesh(positions, indices.astype(np.uint32))
        chart_options = xatlas.ChartOptions()
        chart_options.max_cost = preset.max_cost
        chart_options.max_iterations = preset.max_iterations
        pack_options = xatlas.PackOptions()
        pack_options.resolution = args.resolution
        pack_options.padding = args.padding
        pack_options.bruteForce = False
        report["last_operation"] = "atlas_generate"
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, out_indices, out_uv = atlas[0]
        out_uv = np.asarray(out_uv, np.float64)
        if out_uv.size and out_uv.max() > 1.5:
            out_uv = out_uv / np.array([atlas.width, atlas.height], np.float64)
        out_indices = np.asarray(out_indices, np.int64).reshape(-1, 3)
        out_positions = positions[vmapping]
        out_normals = normals[vmapping] if normals is not None else None
        report["last_operation"] = "checkpoint_raw_candidate"
        write_glb(checkpoint, out_positions, out_normals, out_uv, out_indices)
        np.savez_compressed(
            checkpoint.with_suffix(".arrays.npz"), positions=out_positions, normals=out_normals,
            uv=out_uv, indices=out_indices,
        )
        report["candidate_written"] = True
        report["candidate_hashes"] = {
            "positions": digest(out_positions), "normals": digest(out_normals) if out_normals is not None else None,
            "uv": digest(out_uv), "indices": digest(out_indices),
        }
        report["candidate_geometry"] = {"vertices": int(len(out_positions)), "triangles": int(len(out_indices))}
        write_report()

        report["last_operation"] = "exact_uv_overlap"
        uv_triangles = out_uv[out_indices]
        pos_triangles = out_positions[out_indices]
        overlap = positive_area_uv_overlaps(
            uv_triangles, args.resolution, timeout_seconds=args.overlap_timeout
        )
        stretch, area3d = conformal_stretch(pos_triangles, uv_triangles)
        stretch_p95 = area_weighted_percentile(stretch, area3d, 95.0)
        raw_util = getattr(atlas, "utilization", None)
        if isinstance(raw_util, (list, tuple, np.ndarray)) and len(raw_util):
            utilization = float(raw_util[0])
        elif isinstance(raw_util, (float, int)):
            utilization = float(raw_util)
        else:
            utilization = atlas_utilisation(uv_triangles, args.resolution)
        charts = int(atlas.get_mesh_chart_count(0)) if hasattr(atlas, "get_mesh_chart_count") else chart_count_from_topology(out_indices, len(out_positions))
        tiny = tiny_chart_surface_percent(out_indices, len(out_positions), out_uv, out_positions, args.resolution)
        errors: list[str] = []
        if int(atlas.atlas_count) != 1: errors.append("ATLAS_COUNT")
        if not overlap.success: errors.append("OVERLAP_VALIDATION")
        if overlap.positive_overlap_pair_count: errors.append("POSITIVE_OVERLAP")
        if overlap.degenerate_uv_triangle_count: errors.append("DEGENERATE_UV")
        if overlap.out_of_bounds_triangle_count: errors.append("UV_OUT_OF_BOUNDS")
        if utilization < MIN_ATLAS_UTILIZATION: errors.append("ATLAS_UTILIZATION")
        if charts > MAX_CHART_COUNT: errors.append("CHART_BUDGET")
        if not np.isfinite(stretch_p95) or stretch_p95 > MAX_STRETCH_P95: errors.append("STRETCH")
        if tiny > MAX_TINY_CHART_SURFACE_PERCENT: errors.append("TINY_CHARTS")
        metrics = UvCandidateMetrics(
            preset=preset.name, chart_count=charts, atlas_utilization=utilization,
            atlas_count=int(atlas.atlas_count), atlas_width=int(atlas.width), atlas_height=int(atlas.height),
            overlap_pair_count=overlap.positive_overlap_pair_count,
            overlap_texel_area=overlap.positive_overlap_total_texels_equivalent,
            degenerate_triangle_count=overlap.degenerate_uv_triangle_count,
            out_of_bounds_triangle_count=overlap.out_of_bounds_triangle_count,
            stretch_p95=stretch_p95, tiny_chart_surface_percent=tiny,
            runtime_seconds=time.monotonic() - started, valid=not errors, errors=errors,
            max_cost=preset.max_cost,
        )
        report.update({
            "status": "passed" if metrics.valid else "invalid",
            "metrics": metrics.as_dict(), "exact_overlap": overlap.as_dict(),
            "last_operation": "write_promotable_candidate" if metrics.valid else "validation_complete",
        })
        if metrics.valid:
            write_glb(Path(args.output), out_positions, out_normals, out_uv, out_indices)
            report["output"] = str(Path(args.output))
        write_report()
        print(json.dumps({"status": report["status"], "preset": preset.name}), flush=True)
        return 0 if metrics.valid else 2
    except Exception as exc:  # durable child report; parent classifies it as a failed preset
        report.update({"status": "failed", "error": repr(exc), "last_operation": report.get("last_operation")})
        write_report()
        print(f"XATLAS_CANDIDATE_FAILED {exc!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
