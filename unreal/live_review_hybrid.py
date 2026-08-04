"""Live acceptance evidence for the already-built Castlegrounds hybrid map.

This script is intentionally read-only with respect to the scene actors. It
loads the saved output map, verifies the existing contract, and can finalize
the screenshot receipt after the bridge has captured viewport evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
OUTPUT_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
SOURCE_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"
SOURCE_CAMERA_LABEL = "Castlegrounds_Camera_Source"
SOURCE_MESH_LABEL = "Castlegrounds_ReconstructedMesh"
PROXY_LABEL = "SP_GameplayProxy_Castle_V1"
GROUND_LABEL = "SP_GameplayGround_Castle_V1"
PLAYER_START_LABEL = "SP_PlayerStart_Castle_V1"
EXPECTED_FOV = 66.50838470458984
REVIEW_DIR = REPO_ROOT / "evidence" / "latest-scene-live-review"
SCREENSHOT_DIR = REVIEW_DIR / "screenshots"
SOURCE_MAP_FILE = Path(unreal.Paths.project_content_dir()) / "AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke.umap"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actors():
    return list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())


def _one(actors, label):
    matches = [actor for actor in actors if str(actor.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one actor labeled {label}, found {len(matches)}")
    return matches[0]


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite")
    return value


def _vector(value):
    return [_finite(value.x, "x"), _finite(value.y, "y"), _finite(value.z, "z")]


def _record(actor):
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    return {
        "name": str(actor.get_name()),
        "label": str(actor.get_actor_label()),
        "class": str(actor.get_class().get_name()),
        "location_cm": _vector(location),
        "rotation_deg": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
        "scale": _vector(scale),
        "tags": sorted(str(tag) for tag in list(actor.get_editor_property("tags") or [])),
    }


def _component(actor):
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError(f"{actor.get_name()} has no StaticMeshComponent")
    return component


def _collision(component):
    return str(component.get_collision_enabled())


def _nav(component):
    return bool(component.get_editor_property("can_ever_affect_navigation"))


def _bounds(actor):
    origin, extent = actor.get_actor_bounds(False, True)
    return {"origin_cm": _vector(origin), "extent_cm": _vector(extent)}


def _load_output():
    subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not bool(subsystem.load_level(OUTPUT_MAP)):
        raise RuntimeError(f"could not load {OUTPUT_MAP}")


def _contract():
    actors = _actors()
    world = unreal.EditorLevelLibrary.get_editor_world()
    if world is None:
        raise RuntimeError("editor world is unavailable")
    world_settings = world.get_world_settings()
    game_mode = world_settings.get_editor_property("default_game_mode")
    game_mode_path = str(game_mode.get_path_name()) if game_mode is not None and hasattr(game_mode, "get_path_name") else str(game_mode)
    if game_mode_path != "/Script/Engine.GameModeBase":
        raise RuntimeError(f"default game mode mismatch: {game_mode_path}")
    source_mesh = _one(actors, SOURCE_MESH_LABEL)
    source_camera = _one(actors, SOURCE_CAMERA_LABEL)
    proxy = _one(actors, PROXY_LABEL)
    ground = _one(actors, GROUND_LABEL)
    player_start = _one(actors, PLAYER_START_LABEL)
    source_component = _component(source_mesh)
    proxy_component = _component(proxy)
    ground_component = _component(ground)
    camera_component = source_camera.get_component_by_class(unreal.CameraComponent)
    if camera_component is None:
        raise RuntimeError("source camera component missing")
    fov = float(camera_component.get_editor_property("field_of_view"))
    if abs(fov - EXPECTED_FOV) > 1e-4:
        raise RuntimeError(f"source camera FOV mismatch: {fov}")
    camera_values = []
    for actor in actors:
        camera = actor.get_component_by_class(unreal.CameraComponent)
        if camera is not None:
            camera_values.append({"label": str(actor.get_actor_label()), "fov_deg": float(camera.get_editor_property("field_of_view"))})
    if any(abs(item["fov_deg"] - 48.0) < 1e-4 for item in camera_values):
        raise RuntimeError(f"stale 48 degree camera found: {camera_values}")
    source_collision = _collision(source_component)
    proxy_collision = _collision(proxy_component)
    ground_collision = _collision(ground_component)
    if "NO_COLLISION" not in source_collision.upper() or _nav(source_component):
        raise RuntimeError("source shell is not visual-only")
    if "QUERY_AND_PHYSICS" not in proxy_collision.upper():
        raise RuntimeError(f"proxy is not blocking: {proxy_collision}")
    if "QUERY_AND_PHYSICS" not in ground_collision.upper():
        raise RuntimeError(f"ground is not blocking: {ground_collision}")
    proxy_bounds = _bounds(proxy)
    ground_bounds = _bounds(ground)
    proxy_bottom = proxy_bounds["origin_cm"][2] - proxy_bounds["extent_cm"][2]
    ground_top = ground_bounds["origin_cm"][2] + ground_bounds["extent_cm"][2]
    if abs(proxy_bottom - ground_top) > 1.0:
        raise RuntimeError(f"proxy does not rest on ground: bottom={proxy_bottom}, top={ground_top}")
    player_location = player_start.get_actor_location()
    if abs(player_location.y - proxy.get_actor_location().y) <= proxy_bounds["extent_cm"][1] + 100.0:
        raise RuntimeError("Player Start is inside or too close to proxy depth")
    if abs(player_location.z - (ground_top + 200.0)) > 1.0:
        raise RuntimeError("Player Start is not on the deterministic ground height")
    camera_forward = source_camera.get_actor_forward_vector()
    return {
        "map": OUTPUT_MAP,
        "actors": {
            "source_mesh": _record(source_mesh),
            "source_camera": {**_record(source_camera), "fov_deg": fov},
            "castle_proxy": {**_record(proxy), "bounds": proxy_bounds, "collision": proxy_collision, "navigation": _nav(proxy_component)},
            "ground": {**_record(ground), "bounds": ground_bounds, "collision": ground_collision, "navigation": _nav(ground_component)},
            "player_start": _record(player_start),
        },
        "source_shell_bounds": _bounds(source_mesh),
        "source_camera_forward": _vector(camera_forward),
        "camera_inventory": camera_values,
        "source_shell_visual_only": True,
        "source_shell_collision_disabled": True,
        "source_shell_navigation_disabled": True,
        "proxy_blocking_collision": True,
        "ground_blocking_collision": True,
        "proxy_rests_on_ground": True,
        "player_start_outside_proxy": True,
        "default_game_mode": game_mode_path,
        "source_map_sha256": _sha256(SOURCE_MAP_FILE),
    }


def _review():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _load_output()
    result = _contract()
    result.update({
        "schema_version": "live_scene_review_state_v1",
        "classification": "PROVEN",
        "screenshots_expected": [
            "level_overview.png",
            "source_camera_view.png",
            "castle_proxy_selected.png",
            "castle_proxy_collision.png",
            "ground_and_player_start.png",
            "shell_and_proxy_independent.png",
        ],
        "screenshot_dir": str(SCREENSHOT_DIR),
    })
    (REVIEW_DIR / "live_review_state.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    unreal.log("LIVE_HYBRID_CONTRACT=PROVEN")


def _finalize():
    state_path = REVIEW_DIR / "live_review_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    screenshots = []
    missing = []
    for name in state["screenshots_expected"]:
        path = SCREENSHOT_DIR / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
        else:
            screenshots.append({"path": str(path), "bytes": path.stat().st_size})
    receipt = {
        "schema_version": "live_scene_review_receipt_v1",
        "classification": "PROVEN" if not missing else "NOT_PROVEN",
        "live_editor_level_review": "PROVEN",
        "source_camera_view_captured": not any("source_camera_view.png" in item for item in missing),
        "source_shell_visual_only": state["source_shell_visual_only"],
        "castle_proxy_reviewed": not any("castle_proxy" in item for item in missing),
        "source_map_sha256": state["source_map_sha256"],
        "screenshots": screenshots,
        "missing_screenshots": missing,
        "contract": state,
        "errors": [],
    }
    (REVIEW_DIR / "live_review_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    unreal.log("LIVE_HYBRID_REVIEW=" + receipt["classification"])


def _pie_probe():
    worlds = list(unreal.EditorLevelLibrary.get_pie_worlds(False)) if hasattr(unreal.EditorLevelLibrary, "get_pie_worlds") else []
    if not worlds:
        raise RuntimeError("PIE world is not running")
    world = worlds[0]
    settings = world.get_world_settings()
    properties = {}
    for name in ("default_game_mode", "game_mode_override", "default_pawn_class"):
        try:
            value = settings.get_editor_property(name)
            properties[name] = str(value.get_path_name()) if value is not None and hasattr(value, "get_path_name") else str(value)
        except Exception as exc:
            properties[name] = f"UNAVAILABLE:{type(exc).__name__}"
    controller = unreal.GameplayStatics.get_player_controller(world, 0)
    pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if controller is not None else None
    runtime_game_mode = unreal.GameplayStatics.get_game_mode(world)
    runtime_game_mode_path = str(runtime_game_mode.get_path_name()) if runtime_game_mode is not None and hasattr(runtime_game_mode, "get_path_name") else str(runtime_game_mode)
    runtime_default_pawn = None
    if runtime_game_mode is not None:
        try:
            runtime_default_pawn = str(runtime_game_mode.get_editor_property("default_pawn_class").get_path_name())
        except Exception as exc:
            runtime_default_pawn = f"UNAVAILABLE:{type(exc).__name__}"
    chosen_start = None
    if controller is not None:
        try:
            chosen = unreal.GameplayStatics.choose_player_start(world, controller)
            chosen_start = _record(chosen) if chosen is not None else None
        except Exception as exc:
            chosen_start = f"UNAVAILABLE:{type(exc).__name__}"
    starts = [_record(actor) for actor in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.PlayerStart)]
    result = {
        "schema_version": "live_pie_probe_v1",
        "classification": "PROVEN" if pawn is not None else "REJECTED",
        "world": str(world.get_path_name()),
        "world_settings": properties,
        "runtime_game_mode": runtime_game_mode_path,
        "runtime_default_pawn_class": runtime_default_pawn,
        "chosen_player_start": chosen_start,
        "player_starts": starts,
        "pawn": _record(pawn) if pawn is not None else None,
    }
    (REVIEW_DIR / "pie_probe.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    unreal.log("LIVE_PIE_PROBE=" + result["classification"])


mode = sys.argv[1] if len(sys.argv) > 1 else "review"
try:
    if mode == "review":
        _review()
    elif mode == "finalize":
        _finalize()
    elif mode == "pie_probe":
        _pie_probe()
    else:
        raise RuntimeError(f"unknown mode: {mode}")
except Exception as exc:
    error = {"schema_version": "live_scene_review_state_v1", "classification": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}
    (REVIEW_DIR / "live_review_error.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
    unreal.log_error("LIVE_HYBRID_REVIEW=REJECTED " + error["error"])
    raise
