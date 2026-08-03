"""Read-only audit of the saved Castlegrounds Unreal scene.

Runs inside UnrealEditor-Cmd using the PythonScript commandlet. The script loads
an existing map, verifies the expected imported mesh actor and source camera,
and writes a receipt. It never saves, creates, deletes, imports, or modifies an
asset or actor.
"""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import unreal


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _safe_label(actor: unreal.Actor) -> str:
    try:
        return str(actor.get_actor_label())
    except Exception:
        return str(actor.get_name())


def _property(obj: Any, name: str) -> Any:
    try:
        return obj.get_editor_property(name)
    except Exception:
        return None


def _component(actor: unreal.Actor, component_class: type) -> Any:
    try:
        return actor.get_component_by_class(component_class)
    except Exception:
        return None


def main() -> int:
    output_path = Path(_env("SCENE_UNREAL_AUDIT_OUTPUT"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    receipt: dict[str, Any] = {
        "schema_version": "unreal_scene_contract_audit_v1",
        "classification": "REJECTED",
        "asset_write_attempted": False,
        "actor_write_attempted": False,
        "save_called": False,
        "import_called": False,
        "render_called": False,
        "gpu_work_requested": False,
    }

    try:
        map_path = _env("SCENE_UNREAL_MAP")
        mesh_asset_path = _env("SCENE_UNREAL_MESH_ASSET")
        scene_build = _read_json(_env("SCENE_BUILD_RECEIPT"))
        camera_contract = _read_json(_env("SCENE_CAMERA_CONTRACT"))

        if scene_build.get("classification") != "SCENE_BUILD_PROVEN":
            raise RuntimeError("existing scene build receipt is not proven")
        if camera_contract.get("classification") != "PROVEN":
            raise RuntimeError("authoritative camera contract is not proven")
        if scene_build.get("map") != map_path:
            raise RuntimeError("requested map differs from proven scene build receipt")
        if scene_build.get("mesh_asset") != mesh_asset_path:
            raise RuntimeError("requested mesh differs from proven scene build receipt")

        expected_fov = _float(camera_contract.get("fov_x_deg"), "camera fov_x_deg")
        receipt_fov = _float(scene_build.get("camera_fov_horizontal_deg"), "scene build camera fov")
        if abs(expected_fov - receipt_fov) > 1e-4:
            raise RuntimeError(
                f"authoritative and Unreal scene-build FOV disagree: {expected_fov} vs {receipt_fov}"
            )

        level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        loaded = bool(level_subsystem.load_level(map_path))
        if not loaded:
            raise RuntimeError(f"could not load level: {map_path}")

        asset_exists = bool(unreal.EditorAssetLibrary.does_asset_exist(mesh_asset_path))
        mesh_asset = unreal.load_asset(mesh_asset_path) if asset_exists else None
        if not asset_exists or mesh_asset is None:
            raise RuntimeError(f"expected static mesh asset is missing: {mesh_asset_path}")

        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
        if not actors:
            raise RuntimeError("loaded map contains no level actors")

        mesh_actor_records: list[dict[str, Any]] = []
        expected_mesh_actor_records: list[dict[str, Any]] = []
        camera_records: list[dict[str, Any]] = []
        expected_source_camera_name = str(scene_build["cameras"]["source"])
        source_camera_record: dict[str, Any] | None = None

        for actor in actors:
            name = str(actor.get_name())
            label = _safe_label(actor)
            class_name = str(actor.get_class().get_name())

            static_component = _component(actor, unreal.StaticMeshComponent)
            if static_component is not None:
                static_mesh = _property(static_component, "static_mesh")
                static_mesh_path = str(static_mesh.get_path_name()) if static_mesh is not None else None
                record = {
                    "name": name,
                    "label": label,
                    "class": class_name,
                    "mesh_asset": static_mesh_path,
                    "location_cm": _vector(actor.get_actor_location()),
                    "rotation_deg": [
                        float(actor.get_actor_rotation().roll),
                        float(actor.get_actor_rotation().pitch),
                        float(actor.get_actor_rotation().yaw),
                    ],
                    "scale": _vector(actor.get_actor_scale3d()),
                }
                mesh_actor_records.append(record)
                if static_mesh_path == mesh_asset_path:
                    expected_mesh_actor_records.append(record)

            camera_component = _component(actor, unreal.CameraComponent)
            if camera_component is not None:
                fov = _property(camera_component, "field_of_view")
                aspect_ratio = _property(camera_component, "aspect_ratio")
                projection_mode = _property(camera_component, "projection_mode")
                record = {
                    "name": name,
                    "label": label,
                    "class": class_name,
                    "field_of_view_deg": float(fov) if fov is not None else None,
                    "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else None,
                    "projection_mode": str(projection_mode) if projection_mode is not None else None,
                    "location_cm": _vector(actor.get_actor_location()),
                    "forward": _vector(actor.get_actor_forward_vector()),
                    "right": _vector(actor.get_actor_right_vector()),
                    "up": _vector(actor.get_actor_up_vector()),
                }
                camera_records.append(record)
                if name == expected_source_camera_name or label == expected_source_camera_name:
                    source_camera_record = record

        if len(expected_mesh_actor_records) != 1:
            raise RuntimeError(
                f"expected exactly one actor using source mesh, found {len(expected_mesh_actor_records)}"
            )
        if source_camera_record is None:
            raise RuntimeError(
                f"expected source camera actor was not found: {expected_source_camera_name}"
            )
        if source_camera_record["field_of_view_deg"] is None:
            raise RuntimeError("source camera exposes no field_of_view")

        fov_error = abs(source_camera_record["field_of_view_deg"] - expected_fov)
        expected_unreal_forward = [
            float(item) for item in scene_build.get("camera_target_direction", [])
        ]
        if len(expected_unreal_forward) != 3:
            raise RuntimeError("scene build receipt has no valid Unreal camera direction")
        forward_error = _distance(source_camera_record["forward"], expected_unreal_forward)

        if fov_error > 1e-4:
            raise RuntimeError(
                f"source camera FOV differs from authoritative contract by {fov_error} degrees"
            )
        if forward_error > 1e-4:
            raise RuntimeError(
                f"source camera forward differs from proven Unreal direction by {forward_error}"
            )

        receipt.update(
            {
                "classification": "PROVEN",
                "map": map_path,
                "map_loaded": loaded,
                "mesh_asset": mesh_asset_path,
                "mesh_asset_exists": asset_exists,
                "actor_count": len(actors),
                "mesh_actor_count": len(mesh_actor_records),
                "camera_actor_count": len(camera_records),
                "expected_mesh_actor_count": len(expected_mesh_actor_records),
                "expected_mesh_actor": expected_mesh_actor_records[0],
                "source_camera": source_camera_record,
                "authoritative_fov_x_deg": expected_fov,
                "unreal_scene_build_fov_x_deg": receipt_fov,
                "fov_error_deg": fov_error,
                "expected_unreal_forward": expected_unreal_forward,
                "forward_error": forward_error,
                "source_mesh_reference_preserved": True,
                "source_camera_contract_preserved": True,
                "scene_build_receipt": str(Path(_env("SCENE_BUILD_RECEIPT")).resolve()),
                "camera_contract": str(Path(_env("SCENE_CAMERA_CONTRACT")).resolve()),
            }
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.append(traceback.format_exc())

    receipt["errors"] = errors
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if receipt["classification"] == "PROVEN":
        unreal.log("UNREAL_SCENE_CONTRACT_AUDIT=PROVEN")
        return 0
    unreal.log_error("UNREAL_SCENE_CONTRACT_AUDIT=REJECTED")
    for error in errors:
        unreal.log_error(error)
    return 2


raise SystemExit(main())
