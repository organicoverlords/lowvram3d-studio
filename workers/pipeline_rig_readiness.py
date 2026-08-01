"""Profile-aware rig-readiness analysis for Pipeline V2.

This is deliberately conservative.  It proves enough geometric clearance for an automatic rig; it
does not try to make fused limbs riggable.  Ambiguous silhouettes fail closed with measurements.
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


def _humanoid_clearance(positions: np.ndarray, axes: tuple[int, int, int]) -> dict:
    width_axis, depth_axis, height_axis = axes
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    extent = hi - lo
    height = max(float(extent[height_axis]), 1e-9)
    width = max(float(extent[width_axis]), 1e-9)
    depth = max(float(extent[depth_axis]), 1e-9)
    fraction = (positions[:, height_axis] - lo[height_axis]) / height

    upper = positions[(fraction >= 0.50) & (fraction <= 0.72)]
    lower = positions[(fraction >= 0.05) & (fraction <= 0.43)]
    upper_hist = _histogram(upper, width_axis, lo[width_axis], hi[width_axis])
    lower_hist = _histogram(lower, width_axis, lo[width_axis], hi[width_axis])

    # The torso dominates the middle.  Independent arms produce valleys between that middle mass
    # and the outer left/right lobes.  Requiring both valleys makes robes or one stray prop unable to
    # manufacture a pass.
    centre = len(upper_hist) // 2
    left_outer = float(upper_hist[2:18].max(initial=0.0))
    right_outer = float(upper_hist[-18:-2].max(initial=0.0))
    left_valley = float(upper_hist[18:29].min(initial=0.0))
    right_valley = float(upper_hist[35:46].min(initial=0.0))
    outer_floor = max(min(left_outer, right_outer), 1.0)
    arm_valley_ratio = max(left_valley, right_valley) / outer_floor
    arms_clear = left_outer > 0 and right_outer > 0 and arm_valley_ratio <= 0.58

    # Two legs should create left/right lower-body lobes with a lower-density centre.  Long robes
    # legitimately fail this automatic gate: a human must author the separation before skinning.
    left_leg = float(lower_hist[10:30].max(initial=0.0))
    right_leg = float(lower_hist[34:54].max(initial=0.0))
    centre_leg = float(lower_hist[29:35].mean())
    leg_peak = max(min(left_leg, right_leg), 1.0)
    leg_gap_ratio = centre_leg / leg_peak
    legs_clear = left_leg > 0 and right_leg > 0 and leg_gap_ratio <= 0.62

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
        "arm_valley_ratio": arm_valley_ratio,
        "arms_clear": bool(arms_clear),
        "leg_gap_ratio": leg_gap_ratio,
        "legs_clear": bool(legs_clear),
        "depth_clear": bool(depth_ratio >= 0.14),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    positions, _, _, triangles = read_glb(Path(args.mesh))
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
        measured = _humanoid_clearance(positions.astype(np.float64),
                                       (width_axis, depth_axis, height_axis))
        report["measured"] = measured
        if not measured["depth_clear"]:
            report["failure_codes"].append("BLOCKED_SHALLOW_DEPTH")
        if not measured["arms_clear"]:
            report["failure_codes"].append("BLOCKED_FUSED_ARMS")
        if not measured["legs_clear"]:
            report["failure_codes"].append("BLOCKED_FUSED_LEGS")
        report["ready"] = not report["failure_codes"]
        report["rig_required"] = True
    else:
        # Quadruped/flying rigs need profile-specific limb and wing landmarks.  Until those
        # detectors exist, claiming readiness would be less safe than stopping explicitly.
        report.update({
            "ready": False,
            "rig_required": True,
            "failure_codes": ["RIG_PROFILE_ANALYZER_REQUIRED"],
            "reason": "automatic readiness is currently proven only for humanoids and static assets",
            "extent": [float(v) for v in extent],
        })

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RIG_READINESS ready={report['ready']} codes={report['failure_codes']}", flush=True)
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
