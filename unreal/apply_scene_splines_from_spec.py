"""Apply semantic scene splines to the already-built hybrid map.

Only unpromoted SplineComponents are created. Width and river exclusion are
kept as spec metadata; this stage does not create water, bridge geometry, PCG,
or collision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
OUT = REPO_ROOT / "evidence" / "latest-scene-splines" / "scene_spline_apply_receipt.json"


def _read_spec(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "scene_spline_spec_v1":
        raise RuntimeError("invalid scene spline spec")
    return value


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "evidence/latest-scene-splines/scene_spline_spec.json"
spec = _read_spec(spec_path)
level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if not bool(level.load_level(MAP_PATH)):
    raise RuntimeError(f"could not load {MAP_PATH}")
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
records = []
for spline in spec["splines"]:
    actor_label = "SP_Spline_" + "_".join(part.title() for part in spline["id"].split("_"))
    matches = [actor for actor in actors.get_all_level_actors() if str(actor.get_actor_label()) == actor_label]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate spline actor {actor_label}")
    if matches:
        actor = matches[0]
        existed = True
    else:
        actor = actors.spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
        if actor is None:
            raise RuntimeError(f"could not spawn {actor_label}")
        actor.set_actor_label(actor_label)
        existed = False
    components = actor.get_components_by_class(unreal.SplineComponent) if hasattr(actor, "get_components_by_class") else []
    component = components[0] if components else actor.get_component_by_class(unreal.SplineComponent)
    if component is None:
        raise RuntimeError(
            f"Unreal Python cannot provision or reflect a dynamic SplineComponent for {actor_label}; "
            "provision it through the existing typed add_component_to_actor tool, then apply points"
        )
    tags = list(spline["tags"])
    tags.extend([f"spline_id={spline['id']}", f"width_m={float(spline['width_m']):g}"])
    if "exclusion_radius_m" in spline:
        tags.append(f"exclusion_radius_m={float(spline['exclusion_radius_m']):g}")
    actor.set_editor_property("tags", sorted(set(tags)))
    component.clear_spline_points(False)
    points_cm = []
    for point_m in spline["points_m"]:
        point_cm = [float(value) * 100.0 for value in point_m]
        points_cm.append(point_cm)
        component.add_spline_point(unreal.Vector(*point_cm), unreal.SplineCoordinateSpace.WORLD, False)
    component.set_closed_loop(False)
    component.update_spline()
    records.append({
        "id": spline["id"],
        "kind": spline["kind"],
        "actor_label": actor_label,
        "actor_name": str(actor.get_name()),
        "component_name": str(component.get_name()),
        "existed": existed,
        "point_count": len(points_cm),
        "points_m": spline["points_m"],
        "points_cm": points_cm,
        "width_m": float(spline["width_m"]),
        "exclusion_radius_m": spline.get("exclusion_radius_m"),
        "tags": sorted(set(tags)),
        "promoted": False,
    })
if not bool(level.save_current_level()):
    raise RuntimeError("scene spline map save failed")
result = {
    "schema_version": "scene_spline_apply_receipt_v1",
    "classification": "PROVEN",
    "map": MAP_PATH,
    "spec_path": str(spec_path.resolve()),
    "spec_schema_version": spec["schema_version"],
    "promotion": False,
    "geometry_generation": False,
    "records": records,
    "save_result": True,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
unreal.log("SCENE_SPLINES_APPLY=PROVEN")
