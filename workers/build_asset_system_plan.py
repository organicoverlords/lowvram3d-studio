"""Build one deterministic plan for PARTS, POSE_PREP and MATERIALS.

This worker performs no inference and no Blender mutations.  It converts optional
semantic proposals into production decisions that later Blender workers can
apply.  Missing proposals produce safe passthrough plans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lowvram3d.material_feature_policy import build_material_plan  # noqa: E402
from lowvram3d.part_semantics import build_parts_plan  # noqa: E402
from lowvram3d.pose_preparation_policy import evaluate_a_pose_eligibility  # noqa: E402

SCHEMA_VERSION = 1


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(
    *,
    profile: str,
    parts_manifest=None,
    material_manifest=None,
    geometry_metrics: dict[str, Any] | None = None,
    separate_props: bool = False,
    hero_soft_surfaces: bool = False,
) -> dict[str, Any]:
    parts = build_parts_plan(parts_manifest, separate_props=separate_props)
    pose = evaluate_a_pose_eligibility(
        profile,
        parts_manifest,
        geometry_metrics=geometry_metrics or {},
    ).to_dict()
    materials = build_material_plan(
        material_manifest,
        profile_name=profile,
        hero_soft_surfaces=hero_soft_surfaces,
    )

    blocking: list[str] = []
    advisory: list[str] = []
    if not materials["material_slot_budget_ok"]:
        blocking.extend(materials["failure_codes"])
    if parts["status"] == "safe_no_semantic_parts":
        advisory.append("PARTS_SEMANTIC_MANIFEST_UNAVAILABLE")
    if not pose["eligible"]:
        advisory.extend(pose["reason_codes"])
    if materials["status"] == "ordinary_pbr_fallback":
        advisory.append("MATERIALS_FEATURE_MANIFEST_UNAVAILABLE")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blocking else "ready",
        "profile": profile,
        "parts": parts,
        "pose_prep": pose,
        "materials": materials,
        "blocking_codes": sorted(set(blocking)),
        "advisory_codes": sorted(set(advisory)),
        "safe_fallbacks": {
            "parts": "preserve_fused_mesh_and_unknown_regions",
            "pose_prep": "preserve_source_pose",
            "materials": "ordinary_pbr",
            "soft_surfaces": "disabled",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--parts-manifest", default="")
    parser.add_argument("--material-manifest", default="")
    parser.add_argument("--geometry-metrics", default="")
    parser.add_argument("--separate-props", action="store_true")
    parser.add_argument("--hero-soft-surfaces", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    geometry_metrics = _load_json(args.geometry_metrics)
    plan = build_plan(
        profile=args.profile,
        parts_manifest=args.parts_manifest or None,
        material_manifest=args.material_manifest or None,
        geometry_metrics=geometry_metrics,
        separate_props=args.separate_props,
        hero_soft_surfaces=args.hero_soft_surfaces,
    )
    plan["inputs"] = {
        "parts_manifest": args.parts_manifest or None,
        "parts_manifest_sha256": _sha256(args.parts_manifest),
        "material_manifest": args.material_manifest or None,
        "material_manifest_sha256": _sha256(args.material_manifest),
        "geometry_metrics": args.geometry_metrics or None,
        "geometry_metrics_sha256": _sha256(args.geometry_metrics),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(
        "ASSET_SYSTEM_PLAN "
        f"status={plan['status']} "
        f"parts={plan['parts']['part_count']} "
        f"pose={plan['pose_prep']['action']} "
        f"materials={plan['materials']['enabled_feature_count']}",
        flush=True,
    )
    if plan["blocking_codes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
