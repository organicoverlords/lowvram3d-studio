"""Generic modular architecture instruction builder."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BuilderContract
from .instructions import actor_record, region_parts, semantic_class


def contract() -> BuilderContract:
    return BuilderContract("architecture", ("architecture", "interior_structure", "wall", "roof", "tower", "opening"), ("modular_architecture", "editable_mesh"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking_by_region", "walkable_openings", "local_architecture", ("geometry", "collision"), "conservative_blocker", {"triangles": 200000})


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for region in spec.get("regions", []):
        semantic = semantic_class(region)
        if semantic not in {"architecture", "interior_structure", "wall", "roof", "tower", "opening"}:
            continue
        region_id = str(region.get("id"))
        region_ids.append(region_id)
        for index, part in enumerate(region_parts(spec, region, overrides, part_key="architecture_parts")):
            part_id = str(part.get("id", f"part_{index + 1:03d}"))
            actor_id = f"architecture_{region_id}_{part_id}"
            actors.append(actor_record(actor_id=actor_id, region_id=region_id, builder_id="architecture", spec=spec, part=part, material_class=str(part.get("material_class", "local_architecture")), collision_policy=str(part.get("collision_policy", "blocking_by_region")), navigation_policy=str(part.get("navigation_policy", "walkable_openings")), source_evidence=region.get("evidence", ["analysis_bundle.regions", "SceneSpec"]), transform_derivation=["SceneSpec.region", "region geometry data", "deterministic modular architecture primitive"], geometry_parameters={"architectural_role": part.get("architectural_role", "module"), "opening": bool(part.get("opening", False))}, semantic_class=semantic))
    return {"schema_version": "scene_architecture_build_instructions_v1", "classification": "PROVEN", "builder_id": "architecture", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"triangles": 200000 * max(1, len(region_ids))}}
