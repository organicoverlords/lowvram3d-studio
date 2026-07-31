"""Create a non-destructive, debris-audited high-resolution geometry master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup


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
    result = audit_and_cleanup(
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
    print(
        "COMPONENT_AUDIT_CLEANUP "
        f"success={result['success']} "
        f"faces={result['topology_before']['faces']}->{result['topology_after']['faces']} "
        f"removed={result['faces_removed_percent']:.4f}% "
        f"boundary={result['topology_before']['boundary_edges']}->{result['topology_after']['boundary_edges']} "
        f"passes={len(result['passes'])}",
        flush=True,
    )
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
