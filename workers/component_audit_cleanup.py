"""Create a non-destructive, debris-audited high-resolution geometry master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lowvram3d.component_audit import AuditConfig, audit_and_cleanup


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

    config = AuditConfig(
        render_size=max(192, args.render_size),
        total_samples=max(50_000, args.samples),
        max_passes=max(1, min(args.max_passes, 6)),
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
