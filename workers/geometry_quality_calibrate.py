"""Position pipeline candidates inside the frozen reference distribution, metric by metric.

Deliberately does not collapse the metrics into one score. A single number hides which property
failed, and the whole reason the bad head survived was that every summary statistic looked healthy.

What this can and cannot establish is worth being blunt about. With a large set of good references
and only one confirmed bad model, a metric on which the rejected candidate is an extreme outlier is
a *candidate* discriminator - nothing more. Separation demonstrated against a single negative is not
demonstrated separation, because any metric with enough variance will make some model an outlier
somewhere. Promoting one of these to a hard gate needs more confirmed negatives; until then they are
ranking signals, and the report says so per metric rather than quietly rounding up to a threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Lower is better for these; every other metric is read as higher-is-better.
LOWER_IS_BETTER = {
    "symmetry_median_distance_fraction", "sliver_triangle_fraction", "debris_triangle_fraction",
    "non_manifold_edge_fraction", "boundary_edge_fraction", "tiny_components",
    "triangle_area_cv", "axis_ratio", "components",
}

SKIP = {"mesh", "label", "extent", "triangles", "vertices"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--outlier-percentile", type=float, default=5.0,
                        help="a candidate outside this tail of the reference distribution is flagged")
    args = parser.parse_args()

    references = json.loads(Path(args.references).read_text(encoding="utf-8"))["results"]
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))["results"]
    if not references:
        raise SystemExit("no reference measurements; cannot position anything")

    metrics = sorted({k for r in references for k, v in r.items()
                      if k not in SKIP and isinstance(v, (int, float))})

    findings = {}
    for metric in metrics:
        values = np.array([r[metric] for r in references
                           if isinstance(r.get(metric), (int, float))
                           and np.isfinite(r[metric])], float)
        if len(values) < 4:
            findings[metric] = {"usable": False, "reason": f"only {len(values)} reference values"}
            continue

        lower_better = metric in LOWER_IS_BETTER
        entry = {
            "usable": True,
            "direction": "lower_is_better" if lower_better else "higher_is_better",
            "reference_n": int(len(values)),
            "reference": {
                "min": round(float(values.min()), 6),
                "p05": round(float(np.percentile(values, 5)), 6),
                "median": round(float(np.median(values)), 6),
                "p95": round(float(np.percentile(values, 95)), 6),
                "max": round(float(values.max()), 6),
                "std": round(float(values.std()), 6),
            },
            "candidates": {},
        }
        for candidate in candidates:
            value = candidate.get(metric)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                continue
            percentile = float((values < value).mean() * 100.0)
            # Distance from the reference median in reference standard deviations, signed so that
            # negative always means "worse than the references".
            spread = max(float(values.std()), 1e-12)
            z = (value - float(np.median(values))) / spread
            if lower_better:
                z = -z
            outside = (percentile < args.outlier_percentile if not lower_better
                       else percentile > 100.0 - args.outlier_percentile)
            entry["candidates"][Path(candidate["mesh"]).name] = {
                "value": round(float(value), 6),
                "reference_percentile": round(percentile, 2),
                "z_vs_reference_median": round(float(z), 3),
                "outside_reference_range": bool(outside),
            }
        findings[metric] = entry

    flagged = {name: [m for m, e in findings.items()
                      if e.get("usable") and e["candidates"].get(name, {}).get("outside_reference_range")]
               for name in {Path(c["mesh"]).name for c in candidates}}

    report = {
        "reference_count": len(references),
        "candidate_count": len(candidates),
        "outlier_percentile": args.outlier_percentile,
        "interpretation": (
            "Metrics are reported independently and are ranking signals, not gates. A candidate "
            "outside the reference range on a metric is a flag to investigate. Promotion of any "
            "metric to a hard gate requires several confirmed bad models, not one."
        ),
        "metrics": findings,
        "candidates_flagged_on": flagged,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, metric_names in flagged.items():
        print(f"CANDIDATE {name}: outside reference range on {len(metric_names)} metrics "
              f"{metric_names}", flush=True)
    print(f"CALIBRATE references={len(references)} candidates={len(candidates)} "
          f"metrics={len(metrics)}", flush=True)


if __name__ == "__main__":
    main()
