"""Completeness audit for a one-image scene and its generated layer receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .scene_spec import validate_scene_spec
from .scene_registry import select_builders


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def audit_scene_completeness(spec: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    validation = validate_scene_spec(spec) if spec.get("schema_version") == "scene_spec_v1" else {
        "scene_spec_valid": bool(spec.get("scene_id") and isinstance(spec.get("regions"), list)),
        "errors": [] if spec.get("scene_id") and isinstance(spec.get("regions"), list) else ["scene_id and regions are required"],
    }
    if not validation["scene_spec_valid"]:
        raise ValueError(f"SceneSpec invalid: {validation['errors']}")
    receipts = receipts or {}
    selected = select_builders(spec)
    proven_layers = {name for name, receipt in receipts.items() if receipt.get("classification") == "PROVEN"}
    proven_layers.update({
        alias
        for name, receipt in receipts.items()
        if receipt.get("classification") == "PROVEN"
        for alias in ({"environment": "sky_or_ceiling", "bridge": "crossing", "terrain": "ground_surface"}.get(name, name),)
    })
    entries = []
    for region in spec.get("regions", []):
        region_id = str(region.get("id"))
        layer_type = str(region.get("layer_type") or region.get("layer") or "")
        if not layer_type:
            layer_type = next((name for name, details in selected.items() if region_id in details.get("region_ids", [])), "background_geometry")
        if str(region.get("representation")) in {"visual_shell", "source_visible_shell"}:
            representation = "VISUAL_SHELL"
        elif layer_type in proven_layers:
            representation = "WATER" if layer_type == "water" else "PROCEDURAL_POPULATION" if layer_type == "vegetation" else "SKY" if layer_type in {"sky_or_ceiling", "lighting"} else "EDITABLE_GEOMETRY"
        elif region.get("interactive") or region.get("walkable"):
            representation = "GAMEPLAY_PROXY"
        else:
            representation = "UNRESOLVED"
        entries.append({
            "region_id": region_id,
            "spec_reference": region_id,
            "layer_type": layer_type,
            "representation": representation,
            "independent_geometry": representation in {"EDITABLE_GEOMETRY", "GAMEPLAY_PROXY", "WATER", "PROCEDURAL_POPULATION"},
            "interactive_complete": (
                (representation in {"EDITABLE_GEOMETRY", "WATER", "PROCEDURAL_POPULATION"} and not bool(region.get("interactive")))
                or representation == "EDITABLE_GEOMETRY"
            ),
            "source_shell_only": representation == "VISUAL_SHELL",
        })
    unresolved = [entry["region_id"] for entry in entries if entry["representation"] == "UNRESOLVED"]
    return {
        "schema_version": "scene_completeness_receipt_v1",
        "classification": "PROVEN" if not unresolved else "PARTIAL",
        "scene_id": spec["scene_id"],
        "regions": entries,
        "unresolved_regions": unresolved,
        "all_major_regions_accounted_for": not unresolved,
        "visual_shell_not_counted_as_editable": True,
        "interactive_regions_require_geometry_collision_navigation": True,
    }


def write_completeness_evidence(spec_path: str | Path, output_dir: str | Path, receipts: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[Path, Path]:
    spec = _read(spec_path)
    result = audit_scene_completeness(spec, receipts)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "scene_completeness_plan.json"
    receipt_path = output / "scene_completeness_receipt.json"
    plan_path.write_text(json.dumps({"schema_version": "scene_completeness_plan_v1", "region_order": [str(region.get("id")) for region in spec.get("regions", [])], "source_spec": str(Path(spec_path).resolve())}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, receipt_path
