"""Generic environment actor instruction builder."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BuilderContract
from .instructions import actor_record, region_parts, semantic_class


def contract() -> BuilderContract:
    return BuilderContract("environment", ("sky_or_ceiling", "lighting_source"), ("sky_environment", "visual_shell"), ("regions", "camera"), ("SkyAtmosphere", "DirectionalLight", "ExponentialHeightFog"), "none", "ignored", "environment", ("environment", "lighting"), "flat_environment", {"draw_calls": 10})


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for region in spec.get("regions", []):
        if semantic_class(region) not in {"sky_or_ceiling", "lighting_source"}:
            continue
        region_id = str(region.get("id"))
        region_ids.append(region_id)
        parts = region_parts(spec, region, overrides)
        part = parts[0] if parts else {"center_m": [0.0, 0.0, 0.0], "size_m": [1.0, 1.0, 1.0]}
        actors.append(actor_record(actor_id=f"environment_{region_id}", region_id=region_id, builder_id="environment", spec=spec, part=part, material_class="environment", collision_policy="none", navigation_policy="ignored", source_evidence=region.get("evidence", ["SceneSpec.camera", "SceneSpec.region"]), transform_derivation=["SceneSpec environment semantics", "camera contract"], geometry_parameters={"environment_kind": str(region.get("environment_kind", "atmosphere"))}, asset_path=str(region.get("asset_path", "/Engine/Transient/Environment")), semantic_class=semantic_class(region)))
    return {"schema_version": "scene_environment_build_instructions_v1", "classification": "PROVEN", "builder_id": "environment", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"draw_calls": 10}}
