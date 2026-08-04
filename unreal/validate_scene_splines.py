"""Validate applied semantic scene spline actors without promoting them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
OUT = REPO_ROOT / "evidence" / "latest-scene-splines" / "scene_spline_validation_receipt.json"


def _read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must be an object")
    return value


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "evidence/latest-scene-splines/scene_spline_spec.json"
spec = _read(spec_path)
world = unreal.EditorLevelLibrary.get_editor_world()
if world is None or not str(world.get_path_name()).endswith("L_Castlegrounds_Hybrid_V1"):
    raise RuntimeError("the output map must already be loaded")
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
records = []
for spline in spec["splines"]:
    actor_label = "SP_Spline_" + "_".join(part.title() for part in spline["id"].split("_"))
    matches = [actor for actor in actors if str(actor.get_actor_label()) == actor_label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {actor_label}, found {len(matches)}")
    actor = matches[0]
    component = actor.get_component_by_class(unreal.SplineComponent)
    if component is None:
        raise RuntimeError(f"{actor_label} has no SplineComponent")
    expected_cm = [[float(value) * 100.0 for value in point] for point in spline["points_m"]]
    actual_cm = [_vec(component.get_location_at_spline_point(index, unreal.SplineCoordinateSpace.WORLD)) for index in range(component.get_number_of_spline_points())]
    if actual_cm != expected_cm:
        raise RuntimeError(f"{actor_label} point mismatch: {actual_cm} != {expected_cm}")
    tags = {str(tag) for tag in list(actor.get_editor_property("tags") or [])}
    if "scene_spec_generated" not in tags or "unpromoted" not in tags:
        raise RuntimeError(f"{actor_label} is missing unpromoted scene tags")
    if spline["id"] == "river_main" and not {"water", "no_build"}.issubset(tags):
        raise RuntimeError("river_main semantic tags are incomplete")
    if spline["id"] == "bridge_axis_main" and "crossing" not in tags:
        raise RuntimeError("bridge_axis_main semantic tags are incomplete")
    records.append({
        "id": spline["id"],
        "actor_label": actor_label,
        "component_name": str(component.get_name()),
        "point_count": int(component.get_number_of_spline_points()),
        "points_cm": actual_cm,
        "closed_loop": bool(component.is_closed_loop()),
        "tags": sorted(tags),
        "promoted": False,
    })
result = {
    "schema_version": "scene_spline_validation_receipt_v1",
    "classification": "PROVEN",
    "map": MAP_PATH,
    "spec_path": str(spec_path.resolve()),
    "records": records,
    "all_points_match_spec": True,
    "all_unpromoted": True,
    "geometry_generation": False,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
unreal.log("SCENE_SPLINES_VALIDATION=PROVEN")
