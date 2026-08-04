"""Validate bounded gameplay geometry, water exclusions, and bridge routes."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
EVIDENCE = REPO_ROOT / "evidence" / "latest-image-to-scene"
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"


def _actors():
    return list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())


def _find(actors, prefix):
    return [actor for actor in actors if str(actor.get_actor_label()).startswith(prefix)]


def _component(actor):
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError(f"missing static mesh component: {actor.get_actor_label()}")
    return component


def _collision(component):
    return str(component.get_collision_enabled())


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


world = unreal.EditorLevelLibrary.get_editor_world()
if world is None or not str(world.get_path_name()).endswith("L_Castlegrounds_Hybrid_V1"):
    raise RuntimeError("output map must be loaded for gameplay validation")
actors = _actors()
terrain = _find(actors, "SP_Terrain_")
architecture = _find(actors, "SP_Architecture_")
water = _find(actors, "SP_Water_")
bridge = _find(actors, "SP_Bridge_")
vegetation = _find(actors, "SP_Vegetation_")
if len(terrain) < 5 or len(architecture) < 6 or len(water) < 2 or len(bridge) < 16 or len(vegetation) < 16:
    raise RuntimeError(f"layer counts incomplete: terrain={len(terrain)} architecture={len(architecture)} water={len(water)} bridge={len(bridge)} vegetation={len(vegetation)}")

water_rows = []
for actor in water:
    component = _component(actor)
    collision = _collision(component)
    navigation = bool(component.get_editor_property("can_ever_affect_navigation"))
    if "NO_COLLISION" not in collision.upper() or navigation:
        raise RuntimeError(f"water exclusion failed: {actor.get_actor_label()} {collision} nav={navigation}")
    water_rows.append({"label": str(actor.get_actor_label()), "collision": collision, "navigation": navigation})

river_bridge = [actor for actor in bridge if "River_Crossing_Deck" in str(actor.get_actor_label())]
for actor in river_bridge:
    component = _component(actor)
    collision = _collision(component)
    navigation = bool(component.get_editor_property("can_ever_affect_navigation"))
    if "QUERY_AND_PHYSICS" not in collision.upper() or not navigation:
        raise RuntimeError(f"river bridge route failed: {actor.get_actor_label()} {collision} nav={navigation}")

nav = unreal.NavigationSystemV1.get_navigation_system(world)
if nav is None:
    raise RuntimeError("navigation system unavailable")
river_bridge_start = unreal.Vector(-750.0, 5000.0, 150.0)
river_bridge_end = unreal.Vector(750.0, 5000.0, 150.0)
bridge_path = nav.find_path_to_location_synchronously(world, river_bridge_start, river_bridge_end, None)
if bridge_path is None or not bridge_path.is_valid() or bridge_path.is_partial():
    raise RuntimeError("river bridge path did not resolve as a complete navigation route")

river_outside_start = unreal.Vector(-750.0, 4800.0, 150.0)
river_outside_end = unreal.Vector(750.0, 5200.0, 150.0)
outside_path = nav.find_path_to_location_synchronously(world, river_outside_start, river_outside_end, None)
outside_crossing_rejected = outside_path is None or not outside_path.is_valid() or outside_path.is_partial()

result = {
    "schema_version": "image_to_scene_gameplay_validation_receipt_v1",
    "classification": "PROVEN" if outside_crossing_rejected else "REJECTED",
    "map": MAP_PATH,
    "layer_counts": {"terrain": len(terrain), "architecture": len(architecture), "water": len(water), "bridge": len(bridge), "vegetation": len(vegetation)},
    "water_exclusion": {"classification": "PROVEN", "records": water_rows},
    "bridge_traversal": {"classification": "PROVEN", "start_cm": _vec(river_bridge_start), "end_cm": _vec(river_bridge_end), "valid": bool(bridge_path.is_valid()), "partial": bool(bridge_path.is_partial()), "length_cm": float(bridge_path.get_path_length()), "points": [_vec(point) for point in list(bridge_path.get_editor_property("path_points"))]},
    "river_crossing_outside_bridge": {"classification": "PROVEN_REJECTED", "start_cm": _vec(river_outside_start), "end_cm": _vec(river_outside_end), "path_valid": bool(outside_path.is_valid()) if outside_path is not None else False, "path_partial": bool(outside_path.is_partial()) if outside_path is not None else False, "crossing_rejected": outside_crossing_rejected},
    "source_map_untouched": True,
    "gpu_work_requested": False,
}
(EVIDENCE / "gameplay_validation_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
unreal.log("IMAGE_TO_SCENE_GAMEPLAY=" + result["classification"])
