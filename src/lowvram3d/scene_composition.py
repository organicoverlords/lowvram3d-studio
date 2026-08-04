"""Reusable SceneSpec-to-actor composition stage.

This stage is intentionally pure Python.  It consumes analysis products and
explicit scene data, then emits the complete deterministic actor manifest that
the Unreal adapter is allowed to execute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .builders import architecture, crossing, environment, terrain, vegetation, water
from .builders.instructions import stable_hash


SCHEMA_VERSION = "scene_content_manifest_v1"
_BUILDERS = {
    "terrain": terrain.build_instructions,
    "architecture": architecture.build_instructions,
    "water": water.build_instructions,
    "crossing": crossing.build_instructions,
    "vegetation": vegetation.build_instructions,
    "environment": environment.build_instructions,
}


def load_scene_overrides(scene_id: str, root: str | Path = "scenes") -> dict[str, Any]:
    """Load only an explicit, schema-checked override data file."""

    path = Path(root) / str(scene_id) / "overrides.json"
    if not path.is_file():
        return {"schema_version": "scene_overrides_v1", "scene_id": str(scene_id), "overrides": []}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_scene_overrides(value, scene_id=str(scene_id))
    return value


def validate_scene_overrides(value: Mapping[str, Any], *, scene_id: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if value.get("schema_version") != "scene_overrides_v1":
        errors.append("schema_version must be scene_overrides_v1")
    if scene_id is not None and str(value.get("scene_id")) != str(scene_id):
        errors.append("scene_id does not match the requested SceneSpec")
    overrides = value.get("overrides")
    if not isinstance(overrides, list):
        errors.append("overrides must be an array")
        overrides = []
    seen: set[str] = set()
    required = {"override_id", "target_region_id", "original_inference", "corrected_value", "reason", "evidence", "confidence", "pipeline_stage"}
    for index, item in enumerate(overrides):
        if not isinstance(item, Mapping):
            errors.append(f"overrides[{index}] must be an object")
            continue
        missing = sorted(required - set(item))
        if missing:
            errors.append(f"overrides[{index}] missing {','.join(missing)}")
        key = str(item.get("override_id", ""))
        if not key or key in seen:
            errors.append(f"overrides[{index}] has duplicate or empty override_id")
        seen.add(key)
        try:
            confidence = float(item.get("confidence"))
            if not 0.0 <= confidence <= 1.0:
                errors.append(f"overrides[{index}].confidence must be in [0,1]")
        except (TypeError, ValueError):
            errors.append(f"overrides[{index}].confidence must be numeric")
    if errors:
        raise ValueError("invalid scene overrides: " + "; ".join(errors))
    return {"schema_version": "scene_overrides_v1", "classification": "PROVEN", "override_count": len(overrides), "override_ids": sorted(seen)}


def _override_maps(value: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    entries = [dict(item) for item in value.get("overrides", [])]
    region_map = {}
    for item in entries:
        corrected = item.get("corrected_value", {})
        if isinstance(corrected, Mapping):
            region_map[str(item["target_region_id"])] = dict(corrected)
    return region_map, entries


def build_scene_content_manifest(*, spec: Mapping[str, Any], analysis_bundle: Mapping[str, Any], camera_contract: Mapping[str, Any], semantic_masks: Mapping[str, Any] | None = None, depth_bands: Mapping[str, Any] | None = None, contours: Mapping[str, Any] | None = None, world_anchors: Mapping[str, Any] | None = None, support_relationships: Mapping[str, Any] | None = None, visibility: Mapping[str, Any] | None = None, representation_manifest: Mapping[str, Any] | None = None, material_plan: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compose all generic builders into one manifest owned by the pipeline."""

    if not isinstance(spec, Mapping) or not isinstance(analysis_bundle, Mapping) or not isinstance(camera_contract, Mapping):
        raise ValueError("spec, analysis_bundle, and camera_contract are required mappings")
    override_data = overrides or {"schema_version": "scene_overrides_v1", "scene_id": str(spec.get("scene_id")), "overrides": []}
    validate_scene_overrides(override_data, scene_id=str(spec.get("scene_id")))
    known_ids = {str(region.get("id")) for region in spec.get("regions", [])} | {str(spline.get("id")) for spline in spec.get("splines", [])}
    unknown_targets = sorted({str(item.get("target_region_id")) for item in override_data.get("overrides", [])} - known_ids)
    if unknown_targets:
        raise ValueError(f"scene override targets are absent from SceneSpec: {unknown_targets}")
    region_overrides, override_records = _override_maps(override_data)
    layers: dict[str, dict[str, Any]] = {}
    all_actors: list[dict[str, Any]] = []
    for builder_id, builder in _BUILDERS.items():
        result = builder(spec, representation_manifest=representation_manifest, overrides=region_overrides)
        layers[builder_id] = result
        all_actors.extend(result.get("actors", []))
    actor_ids = [str(actor.get("actor_id")) for actor in all_actors]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("builder output contains duplicate actor_id values")
    required = {"actor_id", "semantic_region_id", "builder_id", "builder_version", "source_evidence", "transform_derivation", "world_transform", "geometry_parameters", "material_class", "collision_policy", "navigation_policy", "deterministic_seed", "asset_path"}
    missing = {key for actor in all_actors for key in required - set(actor)}
    if missing:
        raise ValueError(f"actor manifest missing required fields: {sorted(missing)}")
    material_records = [{"actor_id": actor["actor_id"], "semantic_region_id": actor["semantic_region_id"], "material_class": actor["material_class"], "assignment": "pipeline_manifest", "default_grey_allowed": False} for actor in all_actors]
    inputs = {
        "analysis_bundle_schema": analysis_bundle.get("schema_version"),
        "camera_contract": camera_contract,
        "semantic_masks": semantic_masks or {},
        "depth_bands": depth_bands or spec.get("depth_bands", []),
        "contours": contours or {},
        "world_anchors": world_anchors or spec.get("landmarks", []),
        "support_relationships": support_relationships or spec.get("relationships", []),
        "visibility": visibility or {},
        "representation_manifest": representation_manifest or {},
        "material_plan": material_plan or {},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "PROVEN",
        "scene_id": str(spec.get("scene_id")),
        "source_image_sha256": spec.get("source", {}).get("sha256"),
        "deterministic_seed": int(spec.get("intent", {}).get("deterministic_seed", 0)),
        "inputs": inputs,
        "builders": {key: {"builder_id": value.get("builder_id"), "builder_version": value.get("builder_version"), "region_ids": value.get("region_ids", []), "actor_count": len(value.get("actors", [])), "classification": value.get("classification")} for key, value in layers.items()},
        "layers": layers,
        "actors": all_actors,
        "materials": {"schema_version": "scene_material_build_manifest_v1", "classification": "PROVEN", "records": material_records, "default_grey_allowed": False},
        "overrides": override_records,
        "override_count": len(override_records),
        "manual_only_actor_count": sum(1 for actor in all_actors if actor.get("manual_only")),
        "manifest_hash": stable_hash({"actors": all_actors, "materials": material_records, "overrides": override_records}),
        "rebuild_policy": "delete_generated_map_and_scene_local_assets_then_reapply_manifest",
    }


def write_scene_content_manifest(manifest: Mapping[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def material_build_receipt(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = list(manifest.get("materials", {}).get("records", []))
    return {"schema_version": "material_build_receipt_v1", "classification": "PROVEN" if records and all(item.get("assignment") == "pipeline_manifest" and item.get("default_grey_allowed") is False for item in records) else "REJECTED", "record_count": len(records), "records": records, "source": "scene_content_manifest", "manual_assignments": 0, "default_grey_allowed": False}
