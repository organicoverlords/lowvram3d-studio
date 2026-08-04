"""Generic terrain instruction builder."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BuilderContract
from .instructions import actor_record, region_parts, semantic_class


def contract() -> BuilderContract:
    return BuilderContract("terrain", ("ground", "terrain", "cliff"), ("terrain", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "blocking", "walkable", "rock_or_ground", ("geometry", "collision"), "visual_shell", {"triangles": 100000})


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for region in spec.get("regions", []):
        semantic = semantic_class(region)
        override = (overrides or {}).get(str(region.get("id")), {})
        if semantic != "terrain" and not (isinstance(override, Mapping) and override.get("terrain_parts")):
            continue
        region_id = str(region.get("id"))
        region_ids.append(region_id)
        for index, part in enumerate(region_parts(spec, region, overrides, part_key="terrain_parts")):
            part_id = str(part.get("id", f"part_{index + 1:03d}"))
            actors.append(actor_record(actor_id=f"terrain_{region_id}_{part_id}", region_id=region_id, builder_id="terrain", spec=spec, part=part, material_class=str(part.get("material_class", "rock_or_ground")), collision_policy=str(part.get("collision_policy", "blocking")), navigation_policy=str(part.get("navigation_policy", "walkable")), source_evidence=region.get("evidence", ["analysis_bundle.regions", "SceneSpec"]), transform_derivation=["SceneSpec.region", "region geometry data", "deterministic terrain primitive"], geometry_parameters={"landform_role": part.get("landform_role", "bounded_surface")}, semantic_class=semantic))
    return {"schema_version": "scene_terrain_build_instructions_v1", "classification": "PROVEN", "builder_id": "terrain", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"triangles": 100000 * max(1, len(region_ids))}}
