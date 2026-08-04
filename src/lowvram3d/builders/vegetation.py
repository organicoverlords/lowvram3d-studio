"""Generic deterministic vegetation population builder."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BuilderContract
from .instructions import actor_record, region_parts, semantic_class


def contract() -> BuilderContract:
    return BuilderContract("vegetation", ("vegetation",), ("procedural_population", "procedural_mesh"), ("regions", "visibility"), ("StaticMeshActor",), "none", "ignored", "biome_local", ("population", "exclusion"), "visual_shell", {"instances": 1000})


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for region in spec.get("regions", []):
        if semantic_class(region) != "vegetation":
            continue
        region_id = str(region.get("id"))
        region_ids.append(region_id)
        override = (overrides or {}).get(region_id, {})
        population = override.get("population") if isinstance(override, Mapping) else None
        population = population if population is not None else region.get("population")
        if population is None:
            population = [{"id": "instance_000", "center_m": part.get("center_m", [0.0, 0.0, 0.0]), "size_m": part.get("size_m", [1.0, 1.0, 2.0]), "primitive": "foliage_cross"} for part in region_parts(spec, region, overrides)]
        for index, part in enumerate(population):
            part = dict(part)
            species = str(part.get("species", "vegetation"))
            actor_id = f"vegetation_{region_id}_{species}_{part.get('id', f'instance_{index:03d}') }"
            actors.append(actor_record(actor_id=actor_id, region_id=region_id, builder_id="vegetation", spec=spec, part=part, material_class=str(part.get("material_class", "biome_local")), collision_policy="none", navigation_policy="ignored", source_evidence=region.get("evidence", ["analysis_bundle.regions", "SceneSpec"]), transform_derivation=["SceneSpec region", "deterministic population data", "vegetation exclusion policy"], geometry_parameters={"species": species, "exclusion_regions": list((overrides or {}).get(region_id, {}).get("exclusions", [])) if isinstance((overrides or {}).get(region_id, {}), Mapping) else []}, asset_path=str(part.get("asset_path", "/Engine/BasicShapes/Cube.Cube")), semantic_class="vegetation"))
    return {"schema_version": "scene_vegetation_build_instructions_v1", "classification": "PROVEN", "builder_id": "vegetation", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"instances": 1000 * max(1, len(region_ids))}}
