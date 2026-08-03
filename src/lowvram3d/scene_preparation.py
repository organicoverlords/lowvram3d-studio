"""Create deterministic Blender preparation plans from SceneSpec v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .scene_spec import validate_scene_spec

COLLECTION_BY_REPRESENTATION = {
    "visual_shell": "SCENE_VISUAL_SHELL",
    "editable_mesh": "SCENE_EDITABLE",
    "gameplay_proxy": "SCENE_GAMEPLAY_PROXY",
    "procedural_module": "SCENE_PROCEDURAL_MODULES",
}
REQUIRED_COLLECTIONS = [
    "SCENE_VISUAL_SHELL",
    "SCENE_EDITABLE",
    "SCENE_GAMEPLAY_PROXY",
    "SCENE_PROCEDURAL_MODULES",
    "SCENE_REFERENCE_ONLY",
]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_scene_preparation_plan(spec: dict[str, Any]) -> dict[str, Any]:
    validation = validate_scene_spec(spec)
    if not validation["scene_spec_valid"]:
        return {
            "schema_version": "scene_preparation_plan_v1",
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "validation_errors": validation["errors"],
            "gpu_work_required": False,
        }

    errors: list[dict[str, str]] = []
    tasks: list[dict[str, Any]] = []
    asset_by_id = {asset["id"]: asset for asset in spec.get("assets", [])}

    for index, asset in enumerate(spec.get("assets", [])):
        representation = asset["representation"]
        collection = COLLECTION_BY_REPRESENTATION[representation]
        source_uri = asset.get("source_uri")
        action = "placeholder"
        if source_uri:
            suffix = Path(source_uri).suffix.lower()
            if suffix not in {".glb", ".gltf"}:
                errors.append(
                    {
                        "code": "UNSUPPORTED_SCENE_ASSET_FORMAT",
                        "path": f"$.assets[{index}].source_uri",
                        "message": suffix or "missing extension",
                    }
                )
            else:
                action = "import_gltf"

        if representation == "visual_shell" and (
            asset.get("collision") != "none" or asset.get("navigation") != "ignored"
        ):
            errors.append(
                {
                    "code": "UNSAFE_VISUAL_SHELL_GAMEPLAY_POLICY",
                    "path": f"$.assets[{index}]",
                    "message": "visual shell must have collision=none and navigation=ignored",
                }
            )

        tasks.append(
            {
                "task_id": f"asset:{asset['id']}",
                "asset_id": asset["id"],
                "action": action,
                "collection": collection,
                "source_uri": source_uri,
                "transform": asset["transform"],
                "representation": representation,
                "semantic_class": asset["semantic_class"],
                "collision": asset["collision"],
                "navigation": asset["navigation"],
                "lod_policy": asset["lod_policy"],
                "material_policy": asset.get("material_policy"),
                "triangle_budget": asset.get("triangle_budget", 0),
                "tags": asset.get("tags", []),
            }
        )

    for index, population in enumerate(spec.get("populations", [])):
        if population.get("execution") != "gpu_pcg":
            continue
        for ref in population.get("asset_refs", []):
            asset = asset_by_id[ref]
            if asset.get("collision") != "none" or asset.get("navigation") != "ignored":
                errors.append(
                    {
                        "code": "GPU_POPULATION_REFERENCES_GAMEPLAY_ASSET",
                        "path": f"$.populations[{index}].asset_refs",
                        "message": ref,
                    }
                )

    spline_tasks: list[dict[str, Any]] = []
    for spline in spec.get("splines", []):
        spline_tasks.append(
            {
                "task_id": f"spline:{spline['id']}",
                "spline_id": spline["id"],
                "action": "create_reference_curve",
                "collection": "SCENE_REFERENCE_ONLY",
                "kind": spline["kind"],
                "points_m": spline["points_m"],
                "width_m": spline["width_m"],
                "exclusion_radius_m": spline.get("exclusion_radius_m", 0.0),
                "grammar_id": spline.get("grammar_id"),
                "tags": spline.get("tags", []),
            }
        )

    camera = spec["camera"]["source_camera"]
    plan = {
        "schema_version": "scene_preparation_plan_v1",
        "classification": "PROVEN" if not errors else "REJECTED",
        "scene_spec_valid": True,
        "scene_id": spec["scene_id"],
        "scene_spec_sha256": _canonical_hash(spec),
        "collections": REQUIRED_COLLECTIONS,
        "camera": camera,
        "asset_tasks": tasks,
        "spline_tasks": spline_tasks,
        "counts": {
            "asset_tasks": len(tasks),
            "import_tasks": sum(task["action"] == "import_gltf" for task in tasks),
            "placeholder_tasks": sum(task["action"] == "placeholder" for task in tasks),
            "spline_tasks": len(spline_tasks),
        },
        "errors": errors,
        "gpu_work_required": False,
        "neural_work_required": False,
        "geometry_generation_required": False,
    }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Blender preparation plan from SceneSpec v1")
    parser.add_argument("--scene-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec_path = Path(args.scene_spec)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("SceneSpec root must be an object")
        plan = build_scene_preparation_plan(spec)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        plan = {
            "schema_version": "scene_preparation_plan_v1",
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "errors": [{"code": "PLAN_INPUT_FAILED", "path": str(spec_path), "message": str(exc)}],
            "gpu_work_required": False,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SCENE_PREPARATION_PLAN={plan['classification']}")
    return 0 if plan["classification"] == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
