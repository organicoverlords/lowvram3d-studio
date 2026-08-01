"""Profile-aware rig-readiness analysis for Pipeline V2.

This gate proves geometric clearance; it never pretends fused limbs are riggable. Humanoids are
checked for arm valleys, independent lower-body lobes, a real gap between the feet, and sufficient
front-to-back depth. Complex-accessory humanoids receive one explicit garment-aware rule: a long
robe may occupy the centre of the full lower body, but the asset is leg-clear only when both feet
and the lower shin band still form two independent lobes. Standard humanoids retain the stricter
full lower-body valley requirement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb

RIGGED_PROFILES = {"humanoid", "humanoid_complex_accessories", "quadruped", "flying_creature"}


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = np.ones(radius * 2 + 1, np.float64)
    kernel /= kernel.sum()
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _histogram(points: np.ndarray, axis: int, lo: float, hi: float, bins: int = 64) -> np.ndarray:
    if len(points) == 0 or hi <= lo:
        return np.zeros(bins, np.float64)
    hist, _ = np.histogram(points[:, axis], bins=bins, range=(lo, hi))
    return _smooth(hist)


def _two_lobe_gap(histogram: np.ndarray) -> dict:
    left = float(histogram[10:30].max(initial=0.0))
    right = float(histogram[34:54].max(initial=0.0))
    centre = float(histogram[29:35].mean())
    peak = max(min(left, right), 1.0)
    ratio = centre / peak
    return {
        "left_peak": left,
        "right_peak": right,
        "centre_mean": centre,
        "gap_ratio": ratio,
        "clear": bool(left > 0 and right > 0 and ratio <= 0.62),
    }


def _foot_gap(
    positions: np.ndarray,
    width_axis: int,
    height_axis: int,
    lower_fraction: float = 0.08,
    centre_half_width_fraction: float = 0.06,
) -> dict:
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    extent = hi - lo
    height = max(float(extent[height_axis]), 1e-9)
    width = max(float(extent[width_axis]), 1e-9)
    centre = float((lo[width_axis] + hi[width_axis]) * 0.5)
    height_fraction = (positions[:, height_axis] - lo[height_axis]) / height
    lower = positions[height_fraction <= lower_fraction]
    if not len(lower):
        return {
            "lower_vertices": 0,
            "centre_vertices": 0,
            "centre_fraction": 1.0,
            "left_vertices": 0,
            "right_vertices": 0,
            "feet_clear": False,
        }
    relative = (lower[:, width_axis] - centre) / width
    centre_mask = np.abs(relative) <= centre_half_width_fraction
    left = int((relative < -centre_half_width_fraction).sum())
    right = int((relative > centre_half_width_fraction).sum())
    centre_count = int(centre_mask.sum())
    centre_fraction = centre_count / max(len(lower), 1)
    minimum_lobe = max(int(len(lower) * 0.12), 8)
    clear = centre_fraction <= 0.08 and left >= minimum_lobe and right >= minimum_lobe
    return {
        "lower_vertices": int(len(lower)),
        "centre_vertices": centre_count,
        "centre_fraction": float(centre_fraction),
        "left_vertices": left,
        "right_vertices": right,
        "minimum_lobe_vertices": minimum_lobe,
        "lower_band": lower_fraction,
        "centre_half_width_fraction": centre_half_width_fraction,
        "feet_clear": bool(clear),
    }


def _humanoid_clearance(
    positions: np.ndarray,
    axes: tuple[int, int, int],
    profile: str = "humanoid",
) -> dict:
    width_axis, depth_axis, height_axis = axes
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    extent = hi - lo
    height = max(float(extent[height_axis]), 1e-9)
    width = max(float(extent[width_axis]), 1e-9)
    depth = max(float(extent[depth_axis]), 1e-9)
    fraction = (positions[:, height_axis] - lo[height_axis]) / height

    upper = positions[(fraction >= 0.50) & (fraction <= 0.72)]
    lower = positions[(fraction >= 0.05) & (fraction <= 0.43)]
    shins = positions[(fraction >= 0.03) & (fraction <= 0.22)]
    upper_hist = _histogram(upper, width_axis, lo[width_axis], hi[width_axis])
    lower_hist = _histogram(lower, width_axis, lo[width_axis], hi[width_axis])
    shin_hist = _histogram(shins, width_axis, lo[width_axis], hi[width_axis])

    left_outer = float(upper_hist[2:18].max(initial=0.0))
    right_outer = float(upper_hist[-18:-2].max(initial=0.0))
    left_valley = float(upper_hist[18:29].min(initial=0.0))
    right_valley = float(upper_hist[35:46].min(initial=0.0))
    outer_floor = max(min(left_outer, right_outer), 1.0)
    arm_valley_ratio = max(left_valley, right_valley) / outer_floor
    arms_clear = left_outer > 0 and right_outer > 0 and arm_valley_ratio <= 0.58

    full_lower = _two_lobe_gap(lower_hist)
    shin_lower = _two_lobe_gap(shin_hist)
    feet = _foot_gap(positions, width_axis, height_axis)
    garment_aware = profile == "humanoid_complex_accessories"
    garment_clearance = bool(
        garment_aware and feet["feet_clear"] and shin_lower["clear"]
    )
    legs_clear = bool(full_lower["clear"] or garment_clearance)

    depth_ratio = depth / height
    return {
        "height_axis": int(height_axis),
        "width_axis": int(width_axis),
        "depth_axis": int(depth_axis),
        "extent": [float(v) for v in extent],
        "height": height,
        "width": width,
        "depth": depth,
        "depth_to_height": depth_ratio,
        "upper_band_points": int(len(upper)),
        "lower_band_points": int(len(lower)),
        "shin_band_points": int(len(shins)),
        "arm_valley_ratio": arm_valley_ratio,
        "arms_clear": bool(arms_clear),
        "leg_gap_ratio": float(full_lower["gap_ratio"]),
        "full_lower_body_clear": bool(full_lower["clear"]),
        "shin_gap_ratio": float(shin_lower["gap_ratio"]),
        "shins_clear": bool(shin_lower["clear"]),
        "garment_aware_profile": garment_aware,
        "garment_clearance_applied": garment_clearance,
        "legs_clear": legs_clear,
        "feet": feet,
        "depth_clear": bool(depth_ratio >= 0.14),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    positions, _, _, triangles = read_glb(Path(args.mesh))
    positions = positions.astype(np.float64)
    extent = positions.max(axis=0) - positions.min(axis=0)
    height_axis = int(np.argmax(extent))
    remaining = [axis for axis in range(3) if axis != height_axis]
    depth_axis = remaining[int(np.argmin(extent[remaining]))]
    width_axis = next(axis for axis in remaining if axis != depth_axis)

    report = {
        "mesh": args.mesh,
        "profile": args.profile,
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "failure_codes": [],
    }

    if args.profile not in RIGGED_PROFILES:
        report.update({"ready": True, "rig_required": False, "reason": "profile is static"})
    elif args.profile.startswith("humanoid"):
        measured = _humanoid_clearance(
            positions,
            (width_axis, depth_axis, height_axis),
            args.profile,
        )
        report["measured"] = measured
        if not measured["depth_clear"]:
            report["failure_codes"].append("BLOCKED_SHALLOW_DEPTH")
        if not measured["arms_clear"]:
            report["failure_codes"].append("BLOCKED_FUSED_ARMS")
        if not measured["legs_clear"]:
            report["failure_codes"].append("BLOCKED_FUSED_LEGS")
        if not measured["feet"]["feet_clear"]:
            report["failure_codes"].append("FEET_TOO_CLOSE_FOR_RIGGING")
        report["ready"] = not report["failure_codes"]
        report["rig_required"] = True
    else:
        report.update({
            "ready": False,
            "rig_required": True,
            "failure_codes": ["RIG_PROFILE_ANALYZER_REQUIRED"],
            "reason": "automatic readiness is currently proven only for humanoids and static assets",
            "extent": [float(v) for v in extent],
        })

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RIG_READINESS ready={report['ready']} codes={report['failure_codes']}", flush=True)
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
