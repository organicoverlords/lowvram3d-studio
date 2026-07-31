"""Create a non-destructive, debris-audited high-resolution geometry master."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup


_MICRO_RETRY_ASSET_TYPES = {"character", "creature"}


def config_for_asset_type(
    asset_type: str,
    *,
    render_size: int,
    samples: int,
    max_passes: int,
) -> AuditConfig:
    """Return scale-relative cleanup limits without hard-coding individual assets.

    Continuous organic reconstructions commonly contain sizeable floating shells. Multi-part
    hard-surface and architectural assets legitimately contain disconnected pieces, so automatic
    removal is restricted to much smaller relative surface areas for those families.
    """
    kind = str(asset_type).strip().lower()
    common = {
        "render_size": max(192, render_size),
        "total_samples": max(50_000, samples),
        "max_passes": max(1, min(max_passes, 6)),
    }
    if kind in {"building", "room"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.003,
            hover_max_area_fraction=0.002,
            internal_max_area_fraction=0.0,
            preserve_area_fraction=0.01,
            preserve_face_fraction=0.05,
        )
    if kind in {"scene", "level"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.0015,
            hover_max_area_fraction=0.001,
            internal_max_area_fraction=0.0,
            preserve_area_fraction=0.005,
            preserve_face_fraction=0.02,
        )
    if kind in {"vehicle", "prop"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.015,
            hover_max_area_fraction=0.008,
            internal_max_area_fraction=0.0005,
            preserve_area_fraction=0.03,
            preserve_face_fraction=0.10,
        )
    if kind in {"natural", "vegetation"}:
        return AuditConfig(
            **common,
            outboard_max_area_fraction=0.012,
            hover_max_area_fraction=0.006,
            internal_max_area_fraction=0.0005,
            preserve_area_fraction=0.02,
            preserve_face_fraction=0.08,
        )
    return AuditConfig(**common)


def _micro_retry_candidates(result: dict, asset_type: str) -> list[dict]:
    """Return tiny organic fragments eligible for a higher-confidence second audit.

    The baseline audit remains authoritative. A retry is allowed only when every unresolved
    component is tiny, unstructured, unsupported by the source alpha mask, materially detached,
    and separated from the main surface in many canonical projections. This does not apply to
    buildings, scenes, vehicles, props, vegetation, or natural multi-part assets.
    """
    if str(asset_type).strip().lower() not in _MICRO_RETRY_ASSET_TYPES:
        return []
    errors = [str(item) for item in result.get("errors", [])]
    allowed_errors = all(
        item == "cleanup did not converge within max_passes"
        or (item.endswith("visible components remain audit-required") and item.split()[0].isdigit())
        for item in errors
    )
    if not errors or not allowed_errors:
        return []
    unresolved = [
        item
        for item in (result.get("final_audit", {}).get("decisions", []) or [])
        if item.get("action") == "AUDIT_REQUIRED"
    ]
    if not unresolved:
        return []
    for item in unresolved:
        projection = item.get("projection", {}) or {}
        outboard_evidence = (
            int(projection.get("island_views", 0)) >= 6
            and int(projection.get("gap_views", 0)) >= 6
            and float(projection.get("aggregate_outside_percent", 0.0)) >= 70.0
        )
        hover_evidence = (
            int(projection.get("depth_separated_views", 0)) >= 4
            and int(projection.get("overlap_views", 0)) >= 4
            and float(projection.get("median_depth_gap_diag", 0.0)) >= 0.05
        )
        confirmed_micro_candidate = (
            float(item.get("area_fraction", 1.0)) <= 0.003
            and float(item.get("nearest_distance_diag", 0.0)) >= 0.01
            and float(item.get("elongation", 999.0)) < 6.0
            and float(projection.get("source_support_percent", 100.0)) < 1.0
            and (outboard_evidence or hover_evidence)
        )
        if not confirmed_micro_candidate:
            return []
    return unresolved


def _strict_micro_retry_config(config: AuditConfig) -> AuditConfig:
    """Increase sampling and require broad multi-view evidence for tiny outboard removal."""
    return replace(
        config,
        min_component_samples=max(config.min_component_samples, 1024),
        max_component_samples=max(config.max_component_samples, 16_384),
        max_passes=max(config.max_passes, 6),
        outboard_percent=70.0,
        outboard_views=6,
        gap_views=6,
        outboard_max_area_fraction=min(config.outboard_max_area_fraction, 0.003),
    )


def run_cleanup(
    input_path: str,
    output_path: str,
    *,
    asset_type: str,
    source_image: str | None,
    config: AuditConfig,
    seed: int,
) -> dict:
    """Run the baseline audit, then one bounded evidence-driven retry when justified."""
    baseline = audit_and_cleanup(
        input_path,
        output_path,
        asset_type=asset_type,
        source_image=source_image,
        config=config,
        seed=seed,
    )
    candidates = _micro_retry_candidates(baseline, asset_type)
    if baseline.get("success") or not candidates:
        baseline["audit_policy"] = {
            "selected": "baseline",
            "micro_retry_attempted": False,
        }
        return baseline

    strict_config = _strict_micro_retry_config(config)
    retried = audit_and_cleanup(
        input_path,
        output_path,
        asset_type=asset_type,
        source_image=source_image,
        config=strict_config,
        seed=seed,
    )
    retried["audit_policy"] = {
        "selected": "strict_micro_debris_retry",
        "micro_retry_attempted": True,
        "baseline_errors": baseline.get("errors", []),
        "eligible_signatures": [str(item.get("signature", "")) for item in candidates],
        "eligibility_rule": {
            "asset_types": sorted(_MICRO_RETRY_ASSET_TYPES),
            "max_area_fraction": 0.003,
            "min_nearest_distance_diag": 0.01,
            "max_elongation": 6.0,
            "max_source_support_percent": 1.0,
            "min_outboard_views": 6,
            "min_gap_views": 6,
            "min_outside_percent": 70.0,
        },
        "baseline": baseline,
    }
    return retried


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--asset-type", default="prop")
    parser.add_argument("--source-image", default="")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--samples", type=int, default=220_000)
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = config_for_asset_type(
        args.asset_type,
        render_size=args.render_size,
        samples=args.samples,
        max_passes=args.max_passes,
    )
    result = run_cleanup(
        args.input,
        args.output,
        asset_type=args.asset_type,
        source_image=args.source_image or None,
        config=config,
        seed=args.seed,
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    policy = result.get("audit_policy", {}).get("selected", "baseline")
    print(
        "COMPONENT_AUDIT_CLEANUP "
        f"success={result['success']} "
        f"faces={result['topology_before']['faces']}->{result['topology_after']['faces']} "
        f"removed={result['faces_removed_percent']:.4f}% "
        f"boundary={result['topology_before']['boundary_edges']}->{result['topology_after']['boundary_edges']} "
        f"passes={len(result['passes'])} policy={policy}",
        flush=True,
    )
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
