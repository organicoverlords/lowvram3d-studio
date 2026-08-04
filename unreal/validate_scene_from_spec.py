"""Fresh-process validation for the bounded SceneSpec hybrid level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import unreal


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from lowvram3d.scene_spec import validate_scene_spec  # noqa: E402


PROTECTED_SOURCE_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"
PROXY_LABEL = "SP_GameplayProxy_Castle_V1"
GROUND_LABEL = "SP_GameplayGround_Castle_V1"
PLAYER_START_LABEL = "SP_PlayerStart_Castle_V1"
SOURCE_CAMERA_LABEL = "Castlegrounds_Camera_Source"
DEFAULT_GAME_MODE_PATH = "/Script/Engine.GameModeBase"


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_file(map_path: str) -> Path:
    if not map_path.startswith("/Game/"):
        raise RuntimeError(f"invalid map path: {map_path}")
    relative = map_path[len("/Game/") :].replace("/", os.sep)
    return Path(unreal.Paths.project_content_dir()) / f"{relative}.umap"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-spec", default=os.environ.get("SCENE_SPEC"))
    parser.add_argument("--proxy-plan", default=os.environ.get("SCENE_PROXY_PLAN"))
    parser.add_argument("--source-map", default=os.environ.get("SCENE_SOURCE_MAP", PROTECTED_SOURCE_MAP))
    parser.add_argument("--output-map", default=os.environ.get("SCENE_OUTPUT_MAP"))
    parser.add_argument("--build-receipt", default=os.environ.get("SCENE_BUILD_RECEIPT"))
    parser.add_argument("--receipt", default=os.environ.get("SCENE_VALIDATION_RECEIPT"))
    return parser.parse_args()


def _actors() -> list[Any]:
    return list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())


def _static(actor: Any) -> Any:
    return actor.get_component_by_class(unreal.StaticMeshComponent)


def _collision_text(component: Any) -> str:
    if not hasattr(component, "get_collision_enabled"):
        raise RuntimeError("StaticMeshComponent collision getter unavailable")
    return str(component.get_collision_enabled())


def _record(actor: Any) -> dict[str, Any]:
    loc = actor.get_actor_location()
    rot = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    return {"name": str(actor.get_name()), "label": str(actor.get_actor_label()), "class": str(actor.get_class().get_name()), "location_cm": [float(loc.x), float(loc.y), float(loc.z)], "rotation_deg": [float(rot.roll), float(rot.pitch), float(rot.yaw)], "scale": [float(scale.x), float(scale.y), float(scale.z)], "tags": sorted(str(tag) for tag in list(actor.get_editor_property("tags") or []))}


def _close(a: float, b: float, tolerance: float = 1e-3) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if not all((args.scene_spec, args.proxy_plan, args.output_map, args.build_receipt, args.receipt)):
        raise RuntimeError("scene-spec, proxy-plan, output-map, build-receipt, and receipt are required")
    spec = _read(args.scene_spec)
    plan = _read(args.proxy_plan)
    build = _read(args.build_receipt)
    spec_validation = validate_scene_spec(spec)
    if not spec_validation["scene_spec_valid"]:
        raise RuntimeError(f"SceneSpec validation failed: {spec_validation['errors']}")
    if build.get("classification") != "PROVEN":
        raise RuntimeError("build receipt is not PROVEN")
    if args.source_map != PROTECTED_SOURCE_MAP or args.output_map == args.source_map:
        raise RuntimeError("protected source map enforcement failed")
    source_file = _map_file(args.source_map)
    source_hash = _sha256(source_file)
    if source_hash != build.get("source_map_sha256_before") or source_hash != build.get("source_map_sha256_after"):
        raise RuntimeError("protected source map hash changed")
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not bool(level_subsystem.load_level(args.output_map)):
        raise RuntimeError(f"could not reload output map: {args.output_map}")
    world = unreal.EditorLevelLibrary.get_editor_world()
    if world is None:
        raise RuntimeError("editor world unavailable")
    world_settings = world.get_world_settings()
    game_mode = world_settings.get_editor_property("default_game_mode")
    game_mode_path = str(game_mode.get_path_name()) if game_mode is not None and hasattr(game_mode, "get_path_name") else str(game_mode)
    if game_mode_path != DEFAULT_GAME_MODE_PATH:
        raise RuntimeError(f"default game mode mismatch: {game_mode_path}")
    actors = _actors()
    source_asset_path = spec["outputs"]["unreal_source_mesh_asset"]
    source_mesh = []
    source_camera = []
    proxy = []
    ground = []
    starts = []
    for actor in actors:
        label = str(actor.get_actor_label())
        comp = _static(actor)
        if comp is not None:
            mesh = comp.get_editor_property("static_mesh")
            if mesh is not None and str(mesh.get_path_name()) == source_asset_path:
                source_mesh.append(actor)
        if label == SOURCE_CAMERA_LABEL:
            source_camera.append(actor)
        if label == PROXY_LABEL:
            proxy.append(actor)
        if label == GROUND_LABEL:
            ground.append(actor)
        if label == PLAYER_START_LABEL:
            starts.append(actor)
    if len(source_mesh) != 1 or len(source_camera) != 1 or len(proxy) != 1 or len(ground) != 1 or len(starts) != 1:
        raise RuntimeError(f"actor persistence counts invalid: mesh={len(source_mesh)} camera={len(source_camera)} proxy={len(proxy)} ground={len(ground)} player_start={len(starts)}")
    source_component = _static(source_mesh[0])
    collision = _collision_text(source_component)
    if "NO_COLLISION" not in collision.upper():
        raise RuntimeError(f"source visual shell collision is not disabled: {collision}")
    camera_component = source_camera[0].get_component_by_class(unreal.CameraComponent)
    if camera_component is None:
        raise RuntimeError("source camera component missing")
    fov = float(camera_component.get_editor_property("field_of_view"))
    expected_fov = float(spec["camera"]["source_camera"]["field_of_view_deg"])
    if not _close(fov, expected_fov, 1e-4):
        raise RuntimeError(f"source camera FOV mismatch: {fov} vs {expected_fov}")
    current_camera_record = _record(source_camera[0])
    expected_camera_record = build.get("source_camera") or {}
    for field in ("location_cm", "rotation_deg", "scale"):
        actual_values = current_camera_record.get(field)
        expected_values = expected_camera_record.get(field)
        if not isinstance(actual_values, list) or not isinstance(expected_values, list) or len(actual_values) != len(expected_values):
            raise RuntimeError(f"source camera transform record missing: {field}")
        if any(not _close(float(actual), float(expected), 1e-3) for actual, expected in zip(actual_values, expected_values)):
            raise RuntimeError(f"source camera transform changed: {field}")
    proxy_location = proxy[0].get_actor_location()
    proxy_scale = proxy[0].get_actor_scale3d()
    dims = plan["dimensions_cm"]
    expected_location = plan["actor_center_cm"]
    expected_scale = [dims["width_cm"] / 100.0, dims["depth_cm"] / 100.0, dims["height_cm"] / 100.0]
    if any(not _close(actual, expected, 1e-2) for actual, expected in zip([proxy_location.x, proxy_location.y, proxy_location.z], expected_location)):
        raise RuntimeError("proxy location does not match plan")
    if any(not _close(actual, expected, 1e-4) for actual, expected in zip([proxy_scale.x, proxy_scale.y, proxy_scale.z], expected_scale)):
        raise RuntimeError("proxy dimensions do not match plan")
    expected_start = (build.get("player_start") or {}).get("location_cm")
    if not isinstance(expected_start, list) or len(expected_start) != 3:
        raise RuntimeError("build receipt is missing deterministic Player Start location")
    start_location = starts[0].get_actor_location()
    if any(not _close(actual, expected, 1e-2) for actual, expected in zip([start_location.x, start_location.y, start_location.z], expected_start)):
        raise RuntimeError("Player Start location drifted from build receipt")
    if abs(start_location.y - proxy_location.y) <= dims["depth_cm"] / 2.0 + 100.0:
        raise RuntimeError("Player Start remains inside or too close to proxy depth")
    proxy_component = _static(proxy[0])
    proxy_collision = _collision_text(proxy_component)
    if "QUERY_AND_PHYSICS" not in proxy_collision.upper():
        raise RuntimeError(f"proxy collision is not blocking: {proxy_collision}")
    required_tags = {"gameplay_proxy", "replaceable", "unpromoted", "scene_spec_generated", "castle_proxy"}
    proxy_tags = {str(tag) for tag in list(proxy[0].get_editor_property("tags") or [])}
    if not required_tags.issubset(proxy_tags):
        raise RuntimeError(f"proxy tags missing: {sorted(required_tags - proxy_tags)}")
    ground_component = _static(ground[0])
    ground_collision = _collision_text(ground_component)
    if "QUERY_AND_PHYSICS" not in ground_collision.upper():
        raise RuntimeError("walkable ground collision missing")
    if not any("DirectionalLight" in str(actor.get_class().get_name()) for actor in actors):
        raise RuntimeError("directional light missing")
    if not any("SkyLight" in str(actor.get_class().get_name()) for actor in actors):
        raise RuntimeError("skylight missing")
    return {
        "schema_version": "hybrid_level_validation_receipt_v1",
        "classification": "PROVEN",
        "scene_spec_id": spec["scene_id"],
        "output_map": args.output_map,
        "source_map": args.source_map,
        "map_exists": bool(unreal.EditorAssetLibrary.does_asset_exist(args.output_map)),
        "map_reloaded": True,
        "source_map_sha256": source_hash,
        "source_map_unmodified": source_hash == build.get("source_map_sha256_before") == build.get("source_map_sha256_after"),
        "source_visual_shell": _record(source_mesh[0]),
        "source_visual_shell_collision": collision,
        "source_camera": {**current_camera_record, "field_of_view_deg": fov},
        "source_camera_contract_preserved": True,
        "castle_proxy": {**_record(proxy[0]), "collision": proxy_collision, "dimensions_cm": dims, "navigation_intent": "walkable"},
        "walkable_ground": {**_record(ground[0]), "collision": ground_collision},
        "player_start": _record(starts[0]),
        "player_start_policy": build.get("player_start_policy", "unrecorded"),
        "default_game_mode": game_mode_path,
        "lighting_present": {"directional_light": True, "sky_light": True},
        "fresh_process_validation": True,
        "navigation_proof": "intent_only_not_full_navigation_proof",
        "errors": [],
    }


def main() -> int:
    args = _args()
    try:
        receipt = validate(args)
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        unreal.log("FRESH_PROCESS_VALIDATION=PROVEN")
        return 0
    except Exception as exc:
        failure = {"schema_version": "hybrid_level_validation_receipt_v1", "classification": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        if args.receipt:
            Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
            Path(args.receipt).write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        unreal.log_error(f"FRESH_PROCESS_VALIDATION=REJECTED {failure['error']}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
