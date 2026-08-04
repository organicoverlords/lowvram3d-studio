"""Record bounded PIE traversal and collision evidence for the hybrid map."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
OUT = REPO_ROOT / "evidence" / "latest-scene-live-review" / "traversal_receipt.json"
PLAYER_START_LABEL = "SP_PlayerStart_Castle_V1"
PROXY_LABEL = "SP_GameplayProxy_Castle_V1"


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _actor(world, label):
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    matches = [actor for actor in actors if str(actor.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label}, found {len(matches)}")
    return matches[0]


worlds = list(unreal.EditorLevelLibrary.get_pie_worlds(False))
if not worlds:
    raise RuntimeError("PIE world is not running")
world = worlds[0]
pawn = unreal.GameplayStatics.get_player_pawn(world, 0)
if pawn is None:
    raise RuntimeError("PIE pawn is not available")
start = _actor(world, PLAYER_START_LABEL)
proxy = _actor(world, PROXY_LABEL)
pawn_location = pawn.get_actor_location()
start_location = start.get_actor_location()
proxy_origin, proxy_extent = proxy.get_actor_bounds(False, True)
front_boundary_y = float(proxy_origin.y - proxy_extent.y)
travel_cm = abs(float(pawn_location.y - start_location.y))
clearance_cm = front_boundary_y - float(pawn_location.y)
result = {
    "schema_version": "pie_traversal_receipt_v1",
    "classification": "PROVEN" if travel_cm > 100.0 and clearance_cm >= 0.0 else "REJECTED",
    "map": str(world.get_path_name()),
    "pawn_class": str(pawn.get_class().get_name()),
    "start_location_cm": _vec(start_location),
    "after_forward_input_location_cm": _vec(pawn_location),
    "travel_distance_cm": travel_cm,
    "ground_height_cm": float(start_location.z),
    "pawn_height_after_input_cm": float(pawn_location.z),
    "proxy_bounds_origin_cm": _vec(proxy_origin),
    "proxy_bounds_extent_cm": _vec(proxy_extent),
    "proxy_front_boundary_y_cm": front_boundary_y,
    "pawn_clearance_before_proxy_cm": clearance_cm,
    "proxy_blocked_forward_traversal": clearance_cm >= 0.0,
    "ground_preserved": abs(float(pawn_location.z - start_location.z)) <= 1.0,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
unreal.log("PIE_TRAVERSAL=" + result["classification"])
