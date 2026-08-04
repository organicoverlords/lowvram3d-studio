"""Generic crossing instruction builder derived from crossing splines."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .base import BuilderContract
from .builders_helpers import spline_segments
from .instructions import actor_record


def contract() -> BuilderContract:
    return BuilderContract("crossing", ("crossing", "road_or_path"), ("spline_structure", "procedural_mesh", "gameplay_proxy"), ("regions", "scene_graph"), ("StaticMeshActor",), "blocking", "walkable", "wood_or_local", ("geometry", "collision", "navigation"), "unresolved", {"triangles": 50000})


def build_instructions(spec: Mapping[str, Any], *, representation_manifest: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actors = []
    region_ids = []
    for spline in spec.get("splines", []):
        kind = str(spline.get("kind", "")).lower()
        tags = {str(tag).lower() for tag in spline.get("tags", [])}
        if not ("bridge" in kind or "crossing" in kind or "road" in kind or "crossing" in tags):
            continue
        spline_id = str(spline.get("id", "crossing_spline"))
        width = max(0.5, float(spline.get("width_m", 2.0)))
        region_ids.append(spline_id)
        for index, (start, end) in enumerate(spline_segments(spline.get("points_m", []))):
            dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length <= 0.0:
                raise ValueError("crossing spline contains a zero-length segment")
            part = {"primitive": "box", "center_m": [(start[i] + end[i]) / 2.0 for i in range(3)], "size_m": [length, width, 0.3], "rotation_deg": [0.0, math.degrees(math.atan2(dy, dx)), 0.0]}
            actors.append(actor_record(actor_id=f"crossing_{spline_id}_deck_{index:03d}", region_id=spline_id, builder_id="crossing", spec=spec, part=part, material_class="wood_or_local", collision_policy="blocking", navigation_policy="walkable", source_evidence=["SceneSpec.splines", "scene graph crossing relation"], transform_derivation=["crossing spline endpoints", "segment tangent", "crossing width"], geometry_parameters={"spline_id": spline_id, "segment_index": index, "support_policy": "generated_at_segment_joints"}, semantic_class="crossing"))
    return {"schema_version": "scene_crossing_build_instructions_v1", "classification": "PROVEN" if actors else "NOT_PROVEN", "builder_id": "crossing", "builder_version": "1.0.0", "region_ids": region_ids, "actors": actors, "resource_budget": {"triangles": 50000 * max(1, len(region_ids))}}
