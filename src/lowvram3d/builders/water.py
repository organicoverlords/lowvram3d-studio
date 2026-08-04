"""Generic spline-derived water instruction builder."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .base import BuilderContract
from .instructions import actor_record, region_parts, semantic_class


def contract() -> BuilderContract:
    return BuilderContract("water", ("water",), ("water_surface", "spline_structure"), ("regions", "visibility"), ("StaticMeshActor",), "none", "excluded", "water", ("visibility", "exclusion"), "visual_shell", {"triangles": 20000})


def _segment_part(start: list[float], end: list[float], width: float) -> dict[str, Any]:
    dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 0.0:
        raise ValueError("water spline contains a zero-length segment")
    return {"primitive": "box", "center_m": [(start[i] + end[i]) / 2.0 for i in range(3)], "size_m": [length, max(0.1, float(width)), 0.1], "rotation_deg": [0.0, math.degrees(math.atan2(dy, dx)), 0.0]}


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for spline in spec.get("splines", []):
        kind = str(spline.get("kind", "")).lower()
        if not ("water" in kind or "river" in kind or "stream" in kind or "water" in [str(tag).lower() for tag in spline.get("tags", [])]):
            continue
        spline_id = str(spline.get("id", "water_spline"))
        points = spline.get("points_m", [])
        width = float(spline.get("width_m", 1.0))
        for index, (start, end) in enumerate(zip(points, points[1:])):
            part = _segment_part(list(start), list(end), width)
            actors.append(actor_record(actor_id=f"water_{spline_id}_segment_{index:03d}", region_id=spline_id, builder_id="water", spec=spec, part=part, material_class="water", collision_policy="none", navigation_policy="excluded", source_evidence=["SceneSpec.splines"], transform_derivation=["SceneSpec spline points", "segment midpoint and tangent", "water width"], geometry_parameters={"spline_id": spline_id, "segment_index": index, "exclusion_radius_m": float(spline.get("exclusion_radius_m", 0.0))}, semantic_class="water"))
        region_ids.append(spline_id)
    for region in spec.get("regions", []):
        if semantic_class(region) != "water" or region.get("id") in region_ids:
            continue
        region_ids.append(str(region.get("id")))
        for index, part in enumerate(region_parts(spec, region, overrides)):
            actors.append(actor_record(actor_id=f"water_{region.get('id')}_{index:03d}", region_id=str(region.get("id")), builder_id="water", spec=spec, part=part, material_class="water", collision_policy="none", navigation_policy="excluded", source_evidence=region.get("evidence", ["SceneSpec.region"]), transform_derivation=["SceneSpec.region geometry", "water material policy"], semantic_class="water"))
    return {"schema_version": "scene_water_build_instructions_v1", "classification": "PROVEN", "builder_id": "water", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"triangles": 20000 * max(1, len(region_ids))}}
