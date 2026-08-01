"""Decide between generated geometry routes against the current baseline, and fail closed.

A route only wins by being clearly better than what we already have. "Different" is not "better",
and a route that improves one measure while breaking a blocking one does not win at all - the
blocking checks are gates, not terms in a weighted sum, because averaging is how a candidate with
fused legs and a smoother surface ends up looking like an improvement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DECISIONS = ("ROUTE_B_BETTER", "ROUTE_C_BETTER", "NEITHER_ROUTE_BETTER")

# Fraction of the baseline's value below which a route counts as clearly worse on that measure.
CLEARLY_WORSE = 0.80
# Fraction above the baseline required before a route counts as clearly better.
CLEARLY_BETTER = 1.10


def load(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def blocking_verdicts(quality: dict, rig: dict, baseline_quality: dict) -> list[str]:
    blockers = []
    feet = (rig.get("measured") or {}).get("feet") or {}
    if rig and not rig.get("ready", False):
        codes = ", ".join(rig.get("failure_codes", [])) or "not rig-ready"
        blockers.append(f"FUSED_OR_UNRIGGABLE_LOWER_BODY ({codes}; "
                        f"feet centre fraction {feet.get('centre_fraction')})")

    detail = quality.get("upper_band_detail")
    base_detail = baseline_quality.get("upper_band_detail")
    if detail is not None and base_detail:
        if detail < base_detail * CLEARLY_WORSE:
            blockers.append(f"HEAD_WORSE_THAN_BASELINE (upper-band detail {detail:.1f} "
                            f"vs baseline {base_detail:.1f})")

    triangles = quality.get("triangles")
    base_triangles = baseline_quality.get("triangles")
    if triangles and base_triangles and triangles < base_triangles * 0.25:
        blockers.append(f"MISSING_THIN_FEATURES (only {triangles:,} triangles against a baseline "
                        f"of {base_triangles:,}; cords, pendants and beak taper cannot survive)")

    debris = quality.get("debris_triangle_fraction")
    if debris is not None and debris > 0.02:
        blockers.append(f"FLOATING_DEBRIS ({debris:.3%} of triangles in tiny components)")

    asymmetry = quality.get("symmetry_median_distance_fraction")
    base_asymmetry = baseline_quality.get("symmetry_median_distance_fraction")
    if asymmetry is not None and base_asymmetry and asymmetry > base_asymmetry * 2.0:
        blockers.append(f"ASYMMETRY_WORSE (median mirror distance {asymmetry:.4f} "
                        f"vs baseline {base_asymmetry:.4f})")
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-report", required=True,
                        help="geometry_quality_metrics output covering baseline and routes")
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--reference-name", default="")
    parser.add_argument("--route", action="append", default=[],
                        help="ROUTE_X=mesh_basename[:rig_report_path]")
    parser.add_argument("--blocked-route", action="append", default=[],
                        help="ROUTE_X=reason for a route that could not run at all")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    results = {Path(r["mesh"]).name: r for r in load(args.quality_report).get("results", [])}
    baseline = results.get(args.baseline_name)
    if not baseline:
        raise SystemExit(f"baseline {args.baseline_name} not present in the quality report")
    reference = results.get(args.reference_name) if args.reference_name else None

    routes: dict[str, dict] = {}
    for spec in args.blocked_route:
        name, _, reason = spec.partition("=")
        routes[name] = {"runnable": False, "blockers": [reason], "verdict": "reject"}

    for spec in args.route:
        name, _, rest = spec.partition("=")
        mesh_name, _, rig_path = rest.partition(":")
        quality = results.get(mesh_name)
        if not quality:
            routes[name] = {"runnable": False, "blockers": [f"no metrics for {mesh_name}"],
                            "verdict": "reject"}
            continue
        rig = load(rig_path) if rig_path else {}
        blockers = blocking_verdicts(quality, rig, baseline)
        detail = quality.get("upper_band_detail") or 0.0
        base_detail = baseline.get("upper_band_detail") or 0.0
        clearly_better = bool(not blockers and base_detail and detail >= base_detail * CLEARLY_BETTER)
        routes[name] = {
            "runnable": True,
            "mesh": quality.get("mesh"),
            "triangles": quality.get("triangles"),
            "components": quality.get("components"),
            "upper_band_detail": quality.get("upper_band_detail"),
            "curvature_energy": quality.get("curvature_energy"),
            "symmetry_median_distance_fraction": quality.get("symmetry_median_distance_fraction"),
            "rig_ready": rig.get("ready"),
            "rig_failure_codes": rig.get("failure_codes", []),
            "blockers": blockers,
            "clearly_better_than_baseline": clearly_better,
            "verdict": "keep" if clearly_better else "reject",
        }

    winners = [n for n, r in routes.items() if r.get("clearly_better_than_baseline")]
    if len(winners) == 1 and winners[0] in ("ROUTE_B", "ROUTE_C"):
        decision = f"{winners[0]}_BETTER"
    else:
        decision = "NEITHER_ROUTE_BETTER"
    assert decision in DECISIONS

    report = {
        "decision": decision,
        "baseline": {
            "name": args.baseline_name,
            "mesh": baseline.get("mesh"),
            "triangles": baseline.get("triangles"),
            "components": baseline.get("components"),
            "upper_band_detail": baseline.get("upper_band_detail"),
            "curvature_energy": baseline.get("curvature_energy"),
            "symmetry_median_distance_fraction": baseline.get("symmetry_median_distance_fraction"),
        },
        "read_only_reference": {
            "name": args.reference_name,
            "triangles": reference.get("triangles") if reference else None,
            "upper_band_detail": reference.get("upper_band_detail") if reference else None,
            "curvature_energy": reference.get("curvature_energy") if reference else None,
            "note": "quality comparator only; never modified, never a source of geometry",
        } if reference else None,
        "routes": routes,
        "policy": {
            "clearly_worse_fraction": CLEARLY_WORSE,
            "clearly_better_fraction": CLEARLY_BETTER,
            "rule": "blocking checks are gates, not weighted terms; a route wins only by being "
                    "clearly better than the existing baseline on the head measure with no blockers",
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, route in routes.items():
        print(f"{name}: verdict={route['verdict']} blockers={route['blockers']}", flush=True)
    print(f"DECISION {decision}", flush=True)


if __name__ == "__main__":
    main()
