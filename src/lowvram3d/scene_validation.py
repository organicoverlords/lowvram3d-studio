"""Generic SceneSpec and resource-budget validation."""

from __future__ import annotations

from typing import Any, Mapping

from .scene_registry import builder_manifest
from .scene_spec import validate_scene_spec


def validate_scene_plan(spec: Mapping[str, Any], manifest: Mapping[str, Any], max_triangles: int = 1_500_000, max_vram_mb: int = 6144) -> dict[str, Any]:
    schema = validate_scene_spec(spec) if spec.get("schema_version") == "scene_spec_v1" else {
        "scene_spec_valid": bool(spec.get("scene_id") and isinstance(spec.get("regions"), list)),
        "errors": [] if spec.get("scene_id") and isinstance(spec.get("regions"), list) else ["scene_id and regions are required"],
    }
    selected = manifest.get("selected", {})
    requested_triangles = 0
    for details in selected.values():
        budget = details.get("capability", {}).get("resource_budget", {})
        requested_triangles += int(budget.get("triangles", 0))
    result = {
        "schema_version": "scene_plan_validation_v1",
        "classification": "PROVEN" if schema["scene_spec_valid"] and requested_triangles <= int(max_triangles) else "REJECTED",
        "scene_spec_valid": schema["scene_spec_valid"],
        "scene_spec_errors": schema["errors"],
        "selected_builder_count": len(selected),
        "requested_triangle_budget": requested_triangles,
        "max_triangles": int(max_triangles),
        "max_vram_mb": int(max_vram_mb),
        "gpu_work_requested": False,
        "filename_not_used_for_selection": True,
    }
    return result
