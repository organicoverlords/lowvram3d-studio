"""Validate the bounded navmesh and path behavior in the hybrid map."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
OUT = REPO_ROOT / "evidence" / "latest-scene-navigation" / "navigation_validation_receipt.json"
VOLUME_LABEL = "SP_NavMeshBounds_Castle_V1"
PROXY_LABEL = "SP_GameplayProxy_Castle_V1"


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _path_record(path):
    points = [_vec(value) for value in list(path.get_editor_property("path_points"))]
    return {
        "valid": bool(path.is_valid()),
        "partial": bool(path.is_partial()),
        "length_cm": float(path.get_path_length()),
        "points": points,
    }


world = unreal.EditorLevelLibrary.get_editor_world()
if world is None or not str(world.get_path_name()).endswith("L_Castlegrounds_Hybrid_V1"):
    raise RuntimeError("the already-loaded output map is required; navigation must be rebuilt before validation")
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
volume_matches = [a for a in actors if str(a.get_actor_label()) == VOLUME_LABEL]
proxy_matches = [a for a in actors if str(a.get_actor_label()) == PROXY_LABEL]
if len(volume_matches) != 1 or len(proxy_matches) != 1:
    raise RuntimeError("expected exactly one bounded volume and gameplay proxy")
volume = volume_matches[0]
proxy = proxy_matches[0]
volume_origin, volume_extent = volume.get_actor_bounds(False, True)
proxy_origin, proxy_extent = proxy.get_actor_bounds(False, True)
nav = unreal.NavigationSystemV1.get_navigation_system(world)
if nav is None:
    raise RuntimeError("navigation system unavailable")
start = unreal.Vector(1000.0, 1712.5, 700.0)
walk_end = unreal.Vector(1000.0, 2100.0, 700.0)
blocked_end = unreal.Vector(1000.0, 2800.0, 700.0)
walk_path = nav.find_path_to_location_synchronously(world, start, walk_end, None)
blocked_path = nav.find_path_to_location_synchronously(world, start, blocked_end, None)
if walk_path is None or blocked_path is None:
    raise RuntimeError("navigation path query returned no path object")
walk = _path_record(walk_path)
blocked = _path_record(blocked_path)
if not walk["valid"] or walk["partial"]:
    raise RuntimeError(f"walkable path failed: {walk}")
if not blocked["partial"]:
    raise RuntimeError(f"proxy should terminate the path: {blocked}")
result = {
    "schema_version": "navigation_validation_receipt_v1",
    "classification": "PROVEN",
    "map": MAP_PATH,
    "navmesh_bounds_volume": {
        "label": VOLUME_LABEL,
        "location_cm": _vec(volume.get_actor_location()),
        "bounds_origin_cm": _vec(volume_origin),
        "bounds_extent_cm": _vec(volume_extent),
    },
    "proxy_bounds_origin_cm": _vec(proxy_origin),
    "proxy_bounds_extent_cm": _vec(proxy_extent),
    "walk_path": {"start_cm": _vec(start), "end_cm": _vec(walk_end), **walk},
    "blocked_path": {"start_cm": _vec(start), "end_cm": _vec(blocked_end), **blocked},
    "bounded_navigation_proven": True,
    "proxy_blocks_path": True,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
unreal.log("NAVIGATION_VALIDATION=" + result["classification"])
