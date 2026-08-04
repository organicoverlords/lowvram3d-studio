"""Build the first Castlegrounds hybrid level from SceneSpec in Unreal 5.8.

This is a reusable commandlet adapter.  It copies a proven source map, keeps
the imported visual shell and source camera, and adds only a bounded gameplay
box, walkable ground, and minimal level support.  It never imports or edits a
source mesh and does not use MCP.
"""

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
EXPECTED_OUTPUT_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
PROXY_LABEL = "SP_GameplayProxy_Castle_V1"
GROUND_LABEL = "SP_GameplayGround_Castle_V1"
PLAYER_START_LABEL = "SP_PlayerStart_Castle_V1"
SOURCE_CAMERA_LABEL = "Castlegrounds_Camera_Source"
SOURCE_MESH_LABEL = "Castlegrounds_ReconstructedMesh"
MAX_PROXY_DIM_CM = 5000.0


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _map_file(long_package_name: str) -> Path:
    if not long_package_name.startswith("/Game/"):
        raise RuntimeError(f"map must be a /Game package: {long_package_name}")
    relative = long_package_name[len("/Game/") :].replace("/", os.sep)
    return Path(unreal.Paths.project_content_dir()) / f"{relative}.umap"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RuntimeError(f"{name} must be a 3-vector")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _find(items: list[dict[str, Any]], item_id: str, collection: str) -> dict[str, Any]:
    matches = [item for item in items if item.get("id") == item_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {collection} {item_id}, found {len(matches)}")
    return matches[0]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-spec", default=os.environ.get("SCENE_SPEC"))
    parser.add_argument("--proxy-plan", default=os.environ.get("SCENE_PROXY_PLAN"))
    parser.add_argument("--source-map", default=os.environ.get("SCENE_SOURCE_MAP", PROTECTED_SOURCE_MAP))
    parser.add_argument("--output-map", default=os.environ.get("SCENE_OUTPUT_MAP", EXPECTED_OUTPUT_MAP))
    parser.add_argument("--receipt", default=os.environ.get("SCENE_BUILD_RECEIPT"))
    parser.add_argument("--repair-existing-output", action="store_true", default=os.environ.get("SCENE_REPAIR_EXISTING_OUTPUT") == "1")
    return parser.parse_args()


def _actors() -> list[Any]:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return list(subsystem.get_all_level_actors())


def _spawn_asset(asset: Any, location: unreal.Vector) -> Any:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if hasattr(subsystem, "spawn_actor_from_object"):
        return subsystem.spawn_actor_from_object(asset, location, unreal.Rotator(0.0, 0.0, 0.0))
    return unreal.EditorLevelLibrary.spawn_actor_from_object(asset, location, unreal.Rotator(0.0, 0.0, 0.0))


def _spawn_class(actor_class: Any, location: unreal.Vector) -> Any:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if hasattr(subsystem, "spawn_actor_from_class"):
        return subsystem.spawn_actor_from_class(actor_class, location, unreal.Rotator(0.0, 0.0, 0.0))
    return unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, location, unreal.Rotator(0.0, 0.0, 0.0))


def _label(actor: Any, value: str) -> None:
    actor.set_actor_label(value)


def _tags(actor: Any, required: list[str]) -> None:
    existing = list(actor.get_editor_property("tags") or [])
    existing_text = {str(tag) for tag in existing}
    for tag in required:
        if tag not in existing_text:
            existing.append(unreal.Name(tag))
    actor.set_editor_property("tags", existing)


def _static_component(actor: Any) -> Any:
    return actor.get_component_by_class(unreal.StaticMeshComponent)


def _set_collision(actor: Any, enabled: bool, navigation: bool) -> dict[str, Any]:
    component = _static_component(actor)
    if component is None:
        raise RuntimeError(f"{actor.get_name()} has no static mesh component")
    if enabled:
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
        component.set_collision_profile_name("BlockAll")
    else:
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        component.set_collision_profile_name("NoCollision")
    component.set_editor_property("can_ever_affect_navigation", navigation)
    actual_navigation = bool(component.get_editor_property("can_ever_affect_navigation"))
    if actual_navigation != navigation:
        raise RuntimeError(f"navigation collision policy did not apply: expected {navigation}, got {actual_navigation}")
    if not hasattr(component, "get_collision_enabled"):
        raise RuntimeError("StaticMeshComponent collision getter unavailable")
    if not hasattr(component, "get_collision_profile_name"):
        raise RuntimeError("StaticMeshComponent collision profile getter unavailable")
    return {
        "collision_enabled": str(component.get_collision_enabled()),
        "collision_profile": str(component.get_collision_profile_name()),
        "can_ever_affect_navigation": bool(component.get_editor_property("can_ever_affect_navigation")),
    }


def _actor_record(actor: Any) -> dict[str, Any]:
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    return {
        "name": str(actor.get_name()),
        "label": str(actor.get_actor_label()),
        "class": str(actor.get_class().get_name()),
        "location_cm": [float(location.x), float(location.y), float(location.z)],
        "rotation_deg": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
        "scale": [float(scale.x), float(scale.y), float(scale.z)],
        "tags": sorted(str(tag) for tag in list(actor.get_editor_property("tags") or [])),
    }


def _save_as(level_subsystem: Any, output_map: str) -> bool:
    if hasattr(level_subsystem, "save_current_level_as"):
        return bool(level_subsystem.save_current_level_as(output_map))
    world = unreal.EditorLevelLibrary.get_editor_world()
    return bool(unreal.EditorLoadingAndSavingUtils.save_map(world, output_map))


def _load(level_subsystem: Any, map_path: str) -> bool:
    if hasattr(level_subsystem, "load_level"):
        return bool(level_subsystem.load_level(map_path))
    return bool(unreal.EditorLoadingAndSavingUtils.load_map(map_path))


def _validate_inputs(spec: dict[str, Any], plan: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation = validate_scene_spec(spec)
    if not validation["scene_spec_valid"]:
        raise RuntimeError(f"SceneSpec validation failed: {validation['errors']}")
    if plan.get("classification") != "PROVEN":
        raise RuntimeError("proxy plan is not PROVEN")
    if args.source_map != PROTECTED_SOURCE_MAP:
        raise RuntimeError("source map differs from protected authoritative source map")
    if args.output_map == args.source_map:
        raise RuntimeError("output map cannot equal protected source map")
    source_asset = _find(spec["assets"], "source_mesh_v2", "asset")
    proxy_asset = _find(spec["assets"], "castle_proxy", "asset")
    camera = spec["camera"]["source_camera"]
    if abs(_finite(camera["field_of_view_deg"], "camera FOV") - _finite(plan["authoritative_camera"]["horizontal_fov_deg"], "plan FOV")) > 1e-6:
        raise RuntimeError("camera FOV differs from authoritative plan")
    dimensions = plan["dimensions_cm"]
    for key in ("width_cm", "height_cm", "depth_cm"):
        value = _finite(dimensions[key], f"proxy {key}")
        if not 0.5 <= value <= MAX_PROXY_DIM_CM:
            raise RuntimeError(f"proxy {key} outside bounded limits: {value}")
    if plan.get("promotion") is True or "promoted" in plan.get("tags", []):
        raise RuntimeError("proxy plan is promoted")
    if source_asset.get("collision") != "none":
        raise RuntimeError("source mesh is not visual-only in SceneSpec")
    if proxy_asset.get("collision") == "none":
        raise RuntimeError("castle proxy has no collision policy")
    return source_asset, proxy_asset, camera


def _find_label(actors: list[Any], label: str) -> Any:
    matches = [actor for actor in actors if str(actor.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one actor labeled {label}, found {len(matches)}")
    return matches[0]


def _repair_existing_output(args: argparse.Namespace, spec: dict[str, Any], plan: dict[str, Any], camera_spec: dict[str, Any], source_file: Path, source_hash_before: str | None) -> dict[str, Any]:
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not _load(level_subsystem, args.output_map):
        raise RuntimeError(f"could not load existing generated output for bounded repair: {args.output_map}")
    actors = _actors()
    source_mesh = _find_label(actors, SOURCE_MESH_LABEL)
    source_camera = _find_label(actors, SOURCE_CAMERA_LABEL)
    proxy = _find_label(actors, PROXY_LABEL)
    ground = _find_label(actors, GROUND_LABEL)
    player_start = _find_label(actors, PLAYER_START_LABEL)
    source_mesh_collision = _set_collision(source_mesh, enabled=False, navigation=False)
    proxy_collision = _set_collision(proxy, enabled=True, navigation=True)
    ground_collision = _set_collision(ground, enabled=True, navigation=True)
    if not _save_as(level_subsystem, args.output_map):
        raise RuntimeError("could not save repaired generated output")
    if not _load(level_subsystem, args.output_map):
        raise RuntimeError("could not reload repaired generated output")
    actors = _actors()
    source_mesh = _find_label(actors, SOURCE_MESH_LABEL)
    source_camera = _find_label(actors, SOURCE_CAMERA_LABEL)
    proxy = _find_label(actors, PROXY_LABEL)
    ground = _find_label(actors, GROUND_LABEL)
    player_start = _find_label(actors, PLAYER_START_LABEL)
    camera_component = source_camera.get_component_by_class(unreal.CameraComponent)
    if camera_component is None:
        raise RuntimeError("repaired source camera has no CameraComponent")
    source_camera_record = _actor_record(source_camera)
    source_camera_record["field_of_view_deg"] = float(camera_component.get_editor_property("field_of_view"))
    if abs(source_camera_record["field_of_view_deg"] - float(camera_spec["field_of_view_deg"])) > 1e-4:
        raise RuntimeError("repaired source camera FOV is not authoritative")
    source_hash_after = _sha256(source_file)
    receipt = {
        "schema_version": "hybrid_level_build_receipt_v1",
        "classification": "PROVEN",
        "scene_spec_id": spec["scene_id"],
        "scene_spec_sha256": plan["source_scene_spec_sha256"],
        "proxy_plan_sha256": hashlib.sha256(Path(args.proxy_plan).read_bytes()).hexdigest(),
        "source_map": args.source_map,
        "output_map": args.output_map,
        "source_map_file": str(source_file),
        "source_map_sha256_before": source_hash_before,
        "source_map_sha256_after": source_hash_after,
        "source_map_unmodified": source_hash_before == source_hash_after,
        "source_mesh_asset": spec["outputs"]["unreal_source_mesh_asset"],
        "source_mesh_actor": _actor_record(source_mesh),
        "source_mesh_collision": _set_collision(source_mesh, enabled=False, navigation=False),
        "source_camera": source_camera_record,
        "source_camera_fov_deg": float(camera_spec["field_of_view_deg"]),
        "castle_proxy": {"label": PROXY_LABEL, "record": _actor_record(proxy), "dimensions_cm": plan["dimensions_cm"], "collision": proxy_collision, "navigation_intent": "walkable"},
        "walkable_ground": {"label": GROUND_LABEL, "record": _actor_record(ground), "collision": ground_collision},
        "player_start": _actor_record(player_start),
        "lighting_present": {"directional_light": any("DirectionalLight" in str(actor.get_class().get_name()) for actor in actors), "sky_light": any("SkyLight" in str(actor.get_class().get_name()) for actor in actors)},
        "source_visual_shell_preserved": True,
        "source_camera_contract_preserved": True,
        "source_mesh_remains_visual_only": True,
        "proxy_not_promoted": plan["promotion"] is False,
        "navigation_proof": "intent_only_not_full_navigation_proof",
        "save_result": True,
        "reload_result": True,
        "bounded_correction": "persisted_collision_profiles_on_existing_generated_output",
        "gpu_work_requested": False,
        "pcg_work_started": False,
        "errors": [],
    }
    return receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not all((args.scene_spec, args.proxy_plan, args.receipt)):
        raise RuntimeError("scene-spec, proxy-plan, and receipt are required")
    spec = _read_json(args.scene_spec)
    plan = _read_json(args.proxy_plan)
    source_asset, proxy_asset, camera_spec = _validate_inputs(spec, plan, args)
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    source_file = _map_file(args.source_map)
    source_hash_before = _sha256(source_file)
    if not unreal.EditorAssetLibrary.does_asset_exist(args.source_map):
        raise RuntimeError(f"source map does not exist: {args.source_map}")
    output_exists = unreal.EditorAssetLibrary.does_asset_exist(args.output_map)
    if output_exists and not args.repair_existing_output:
        raise RuntimeError(f"refusing to overwrite existing output map: {args.output_map}")
    if output_exists and args.repair_existing_output:
        receipt = _repair_existing_output(args, spec, plan, camera_spec, source_file, source_hash_before)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        unreal.log("PIPELINE_UNREAL_BUILD_ADAPTER=PROVEN_BOUNDED_REPAIR")
        return receipt

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not _load(level_subsystem, args.source_map):
        raise RuntimeError(f"could not load source map: {args.source_map}")
    source_actors = _actors()
    source_mesh_actors = []
    source_cameras = []
    for actor in source_actors:
        component = _static_component(actor)
        if component is not None:
            mesh = component.get_editor_property("static_mesh")
            if mesh is not None and str(mesh.get_path_name()) == spec["outputs"]["unreal_source_mesh_asset"]:
                source_mesh_actors.append(actor)
        if str(actor.get_actor_label()) == SOURCE_CAMERA_LABEL:
            source_cameras.append(actor)
    if len(source_mesh_actors) != 1:
        raise RuntimeError(f"expected one source mesh actor, found {len(source_mesh_actors)}")
    if len(source_cameras) != 1:
        raise RuntimeError(f"expected one source camera, found {len(source_cameras)}")
    source_mesh_record = _actor_record(source_mesh_actors[0])
    source_camera_record = _actor_record(source_cameras[0])
    camera_component = source_cameras[0].get_component_by_class(unreal.CameraComponent)
    if camera_component is None:
        raise RuntimeError("source camera has no CameraComponent")
    source_camera_record["field_of_view_deg"] = float(camera_component.get_editor_property("field_of_view"))
    if abs(source_camera_record["field_of_view_deg"] - float(camera_spec["field_of_view_deg"])) > 1e-4:
        raise RuntimeError("source camera FOV is not authoritative")
    source_mesh_collision = _set_collision(source_mesh_actors[0], enabled=False, navigation=False)
    _tags(source_mesh_actors[0], ["visual_shell", "visual_only", "source_locked", "source_mesh_v2"])
    if not _save_as(level_subsystem, args.output_map):
        raise RuntimeError(f"could not save output map: {args.output_map}")
    if not _load(level_subsystem, args.output_map):
        raise RuntimeError(f"could not reload output map after copy: {args.output_map}")

    cube_asset = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if cube_asset is None:
        raise RuntimeError("engine cube asset unavailable")
    centre = plan["actor_center_cm"]
    dimensions = plan["dimensions_cm"]
    proxy = _spawn_asset(cube_asset, unreal.Vector(*centre))
    if proxy is None:
        raise RuntimeError("could not spawn castle gameplay proxy")
    _label(proxy, PROXY_LABEL)
    proxy.set_actor_scale3d(unreal.Vector(dimensions["width_cm"] / 100.0, dimensions["depth_cm"] / 100.0, dimensions["height_cm"] / 100.0))
    _tags(proxy, plan["tags"])
    proxy_collision = _set_collision(proxy, enabled=True, navigation=True)

    if "base_anchor_cm" in plan:
        anchor = _vector(plan["base_anchor_cm"], "proxy base_anchor_cm")
    elif "base_anchor_m" in plan:
        anchor = [100.0 * value for value in _vector(plan["base_anchor_m"], "proxy base_anchor_m")]
    else:
        anchor = [centre[0], centre[1], centre[2] - dimensions["height_cm"] / 2.0]
    ground_width = max(3000.0, dimensions["width_cm"] * 1.5)
    ground_depth = max(3000.0, dimensions["depth_cm"] * 1.5)
    ground = _spawn_asset(cube_asset, unreal.Vector(anchor[0], anchor[1], anchor[2] - 50.0))
    if ground is None:
        raise RuntimeError("could not spawn walkable ground")
    _label(ground, GROUND_LABEL)
    ground.set_actor_scale3d(unreal.Vector(ground_width / 100.0, ground_depth / 100.0, 1.0))
    _tags(ground, ["gameplay_proxy", "replaceable", "unpromoted", "scene_spec_generated", "walkable_ground"])
    ground_collision = _set_collision(ground, enabled=True, navigation=True)

    player_start = _spawn_class(unreal.PlayerStart, unreal.Vector(anchor[0], anchor[1], anchor[2] + 200.0))
    if player_start is None:
        raise RuntimeError("could not spawn PlayerStart")
    _label(player_start, PLAYER_START_LABEL)
    _tags(player_start, ["gameplay_proxy", "replaceable", "unpromoted", "scene_spec_generated"])

    actors_after = _actors()
    has_directional = any("DirectionalLight" in str(actor.get_class().get_name()) for actor in actors_after)
    has_sky = any("SkyLight" in str(actor.get_class().get_name()) for actor in actors_after)
    if not has_directional:
        light = _spawn_class(unreal.DirectionalLight, unreal.Vector(anchor[0], anchor[1], anchor[2] + 1000.0))
        if light is None:
            raise RuntimeError("could not spawn directional light")
        _label(light, "SP_DirectionalLight_Castle_V1")
    if not has_sky:
        sky = _spawn_class(unreal.SkyLight, unreal.Vector(anchor[0], anchor[1], anchor[2] + 1000.0))
        if sky is None:
            raise RuntimeError("could not spawn skylight")
        _label(sky, "SP_SkyLight_Castle_V1")

    if not _save_as(level_subsystem, args.output_map):
        raise RuntimeError("could not save populated hybrid map")
    source_hash_after = _sha256(source_file)
    receipt = {
        "schema_version": "hybrid_level_build_receipt_v1",
        "classification": "PROVEN",
        "scene_spec_id": spec["scene_id"],
        "scene_spec_sha256": plan["source_scene_spec_sha256"],
        "proxy_plan_sha256": hashlib.sha256(Path(args.proxy_plan).read_bytes()).hexdigest(),
        "source_map": args.source_map,
        "output_map": args.output_map,
        "source_map_file": str(source_file),
        "source_map_sha256_before": source_hash_before,
        "source_map_sha256_after": source_hash_after,
        "source_map_unmodified": source_hash_before == source_hash_after,
        "source_mesh_asset": spec["outputs"]["unreal_source_mesh_asset"],
        "source_mesh_actor": source_mesh_record,
        "source_mesh_collision": source_mesh_collision,
        "source_camera": source_camera_record,
        "source_camera_fov_deg": float(camera_spec["field_of_view_deg"]),
        "castle_proxy": {"label": PROXY_LABEL, "record": _actor_record(proxy), "dimensions_cm": dimensions, "collision": proxy_collision, "navigation_intent": "walkable"},
        "walkable_ground": {"label": GROUND_LABEL, "record": _actor_record(ground), "collision": ground_collision},
        "player_start": _actor_record(player_start),
        "lighting_present": {"directional_light": True, "sky_light": True},
        "source_visual_shell_preserved": True,
        "source_camera_contract_preserved": True,
        "source_mesh_remains_visual_only": True,
        "proxy_not_promoted": plan["promotion"] is False,
        "navigation_proof": "intent_only_not_full_navigation_proof",
        "save_result": True,
        "reload_result": True,
        "gpu_work_requested": False,
        "pcg_work_started": False,
        "errors": [],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    unreal.log("PIPELINE_UNREAL_BUILD_ADAPTER=PROVEN")
    return receipt


def main() -> int:
    args = _args()
    receipt_path = Path(args.receipt) if args.receipt else None
    try:
        receipt = build(args)
        if receipt_path:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    except Exception as exc:
        failure = {"schema_version": "hybrid_level_build_receipt_v1", "classification": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "gpu_work_requested": False}
        if receipt_path:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        unreal.log_error(f"PIPELINE_UNREAL_BUILD_ADAPTER=REJECTED {failure['error']}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
