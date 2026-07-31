"""Compare one reduced geometry candidate against a clean high-resolution master."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lowvram3d.geometry_compare import compare_meshes
from lowvram3d.quality_ladder import AssetFamily, family_for_asset_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--asset-type", default="auto")
    parser.add_argument("--asset-family", choices=[family.value for family in AssetFamily], default="")
    parser.add_argument("--quality", choices=("background", "gameplay", "hero"), default="hero")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--silhouette-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="candidate")
    args = parser.parse_args()

    family = AssetFamily(args.asset_family) if args.asset_family else family_for_asset_type(args.asset_type)
    report = compare_meshes(
        args.master,
        args.candidate,
        asset_family=family,
        quality=args.quality,
        sample_count=max(20_000, args.samples),
        silhouette_size=max(128, args.silhouette_size),
        seed=args.seed,
        candidate_name=args.name,
    )
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluation = report["evaluation"]
    print(
        "GEOMETRY_COMPARE "
        f"name={evaluation['name']} faces={evaluation['face_count']} "
        f"iou={evaluation['silhouette_iou_min']:.6f} "
        f"p95={evaluation['surface_distance_p95_diag']:.6f} "
        f"thin={evaluation['thin_feature_recall']:.6f} "
        f"components={evaluation['meaningful_component_recall']:.6f} "
        f"valid={evaluation['valid']}",
        flush=True,
    )
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
