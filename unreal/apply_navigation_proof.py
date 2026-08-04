"""Add one bounded NavMesh volume for the gameplay-proxy smoke map."""

from __future__ import annotations

import json
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
LABEL = "SP_NavMeshBounds_Castle_V1"
OUT = REPO_ROOT / "evidence" / "latest-scene-navigation" / "navigation_apply_receipt.json"


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if not bool(level.load_level(MAP_PATH)):
    raise RuntimeError(f"could not load {MAP_PATH}")
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
matches = [a for a in actors.get_all_level_actors() if str(a.get_actor_label()) == LABEL]
if len(matches) > 1:
    raise RuntimeError(f"duplicate navigation volumes labeled {LABEL}")
if matches:
    volume = matches[0]
    existed = True
else:
    volume = actors.spawn_actor_from_class(unreal.NavMeshBoundsVolume, unreal.Vector(1000.0, 2800.0, 1000.0), unreal.Rotator(0.0, 0.0, 0.0))
    if volume is None:
        raise RuntimeError("NavMeshBoundsVolume spawn failed")
    volume.set_actor_label(LABEL)
    volume.set_editor_property("tags", ["gameplay_proxy", "replaceable", "scene_spec_generated", "unpromoted", "navigation_proof"])
    existed = False

volume.set_actor_scale3d(unreal.Vector(15.0, 15.0, 12.0))
origin, extent = volume.get_actor_bounds(False, True)
if not all(float(value) > 0.0 for value in (extent.x, extent.y, extent.z)):
    raise RuntimeError(f"navigation bounds invalid: {_vec(extent)}")
if extent.x < 1499.0 or extent.y < 1499.0 or extent.z < 1199.0:
    raise RuntimeError(f"navigation volume does not cover the bounded gameplay area: {_vec(extent)}")
if not bool(unreal.EditorLevelLibrary.save_current_level()):
    raise RuntimeError("navigation proof map save failed")
result = {
    "schema_version": "navigation_apply_receipt_v1",
    "classification": "PROVEN",
    "map": MAP_PATH,
    "label": LABEL,
    "existed": existed,
    "actor_class": str(volume.get_class().get_name()),
    "location_cm": _vec(volume.get_actor_location()),
    "scale": _vec(volume.get_actor_scale3d()),
    "bounds_origin_cm": _vec(origin),
    "bounds_extent_cm": _vec(extent),
    "bounded_to_gameplay_area": True,
    "save_result": True,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
unreal.log("NAVIGATION_VOLUME_APPLY=PROVEN")
