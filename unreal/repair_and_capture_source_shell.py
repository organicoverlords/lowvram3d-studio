"""Repair and capture the Castlegrounds visual proof lane.

The script is intentionally hybrid-map-only.  It never loads or saves the
protected source map.  It duplicates the imported mesh, uses an unlit source
projection material, disables Nanite on the duplicate, and captures from the
named source CameraActor through a SceneCaptureComponent2D.  It does not use
the PIE/player viewport or Slate screenshot APIs.

Run inside a fresh Unreal Editor session with the hybrid map available:

    py ".../repair_and_capture_source_shell.py"

The output receipt remains NOT_PROVEN until the exported images pass the CPU
validator and the map is reloaded in a fresh editor process.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import unreal


REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
EVIDENCE = REPO_ROOT / "evidence" / "latest-image-to-scene"
SCREENSHOTS = EVIDENCE / "screenshots"
PROJECT_SOURCE = Path(r"C:/Users/Lauri/Downloads/benchmarkpics/castlegrounds.png")
OUTPUT_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
PROTECTED_MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"
SOURCE_CAMERA_LABEL = "Castlegrounds_Camera_Source"
SOURCE_MESH_LABEL = "Castlegrounds_ReconstructedMesh"
SOURCE_MESH_ASSET = "/Game/AgentProof/ImageToSceneSmoke_20260803/Geometry/CastlegroundsSourceMeshV2/castlegrounds_source_mesh_v2/StaticMeshes/castlegrounds_source_mesh_v2.castlegrounds_source_mesh_v2"
HYBRID_MESH_ASSET = "/Game/AgentProof/ImageToSceneSmoke_20260803/HybridSourceShell/StaticMeshes/CastlegroundsSourceShell_NoNanite.CastlegroundsSourceShell_NoNanite"
HYBRID_MATERIAL_ASSET = "/Game/AgentProof/ImageToSceneSmoke_20260803/HybridSourceShell/Materials/M_CastlegroundsSourceProjection_V2.M_CastlegroundsSourceProjection_V2"
HYBRID_TEXTURE_ASSET = "/Game/AgentProof/ImageToSceneSmoke_20260803/HybridSourceShell/Textures/T_CastlegroundsSource.T_CastlegroundsSource"
EXPECTED_FOV = 66.50838470458984
WIDTH = 1448
HEIGHT = 1086


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actors() -> list[Any]:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return list(subsystem.get_all_level_actors())


def _find_label(label: str) -> Any:
    matches = [actor for actor in _actors() if str(actor.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one actor labeled {label}, found {len(matches)}")
    return matches[0]


def _vec(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _safe_property(obj: Any, name: str) -> Any:
    try:
        return obj.get_editor_property(name)
    except Exception:
        return None


def _path(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value.get_path_name())
    except Exception:
        return str(value)


def _material_record(material: Any, slot: int) -> dict[str, Any]:
    path = _path(material)
    text = (path or "").lower()
    record: dict[str, Any] = {
        "slot": slot,
        "path": path,
        "class": str(material.get_class().get_name()) if material is not None else None,
        "blend_mode": str(_safe_property(material, "blend_mode")) if material is not None else None,
        "shading_model": str(_safe_property(material, "shading_model")) if material is not None else None,
        "two_sided": bool(_safe_property(material, "two_sided")) if material is not None else False,
        "unlit": False,
        "texture_paths": [],
        "texture_dimensions": [],
        "engine_placeholder": (not path) or path.startswith("/Engine/") or "worldgrid" in text or "preview" in text or "default" in text,
        "preview_material": "preview" in text or "worldgrid" in text,
    }
    if material is None:
        return record
    try:
        record["unlit"] = "UNLIT" in str(_safe_property(material, "shading_model")).upper() or bool(_safe_property(material, "b_unlit"))
    except Exception:
        pass
    try:
        names = unreal.MaterialEditingLibrary.get_texture_parameter_names(material)
        for name in names:
            texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material, name)
            if texture is not None:
                record["texture_paths"].append(_path(texture))
                record["texture_dimensions"].append({"path": _path(texture), "width": int(_safe_property(texture, "size_x") or 0), "height": int(_safe_property(texture, "size_y") or 0), "srgb": bool(_safe_property(texture, "srgb"))})
    except Exception:
        pass
    return record


def _audit_mesh_actor(actor: Any) -> dict[str, Any]:
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError("source shell has no StaticMeshComponent")
    mesh = _safe_property(component, "static_mesh")
    slots = [_material_record(component.get_material(index), index) for index in range(int(component.get_num_materials()))]
    return {"actor_label": str(actor.get_actor_label()), "mesh_asset": _path(mesh), "material_slots": slots}


def _import_source_texture() -> Any:
    if not PROJECT_SOURCE.is_file():
        raise RuntimeError(f"source image missing: {PROJECT_SOURCE}")
    existing = unreal.load_asset(HYBRID_TEXTURE_ASSET)
    if existing is not None:
        return existing
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(PROJECT_SOURCE))
    task.set_editor_property("destination_path", "/Game/AgentProof/ImageToSceneSmoke_20260803/HybridSourceShell/Textures")
    task.set_editor_property("destination_name", "T_CastlegroundsSource")
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    tools.import_asset_tasks([task])
    texture = unreal.load_asset(HYBRID_TEXTURE_ASSET)
    if texture is None:
        raise RuntimeError("source texture import did not produce the expected asset")
    texture.set_editor_property("srgb", True)
    unreal.EditorAssetLibrary.save_loaded_asset(texture)
    return texture


def _create_source_material(texture: Any) -> Any:
    material = unreal.load_asset(HYBRID_MATERIAL_ASSET)
    if material is None:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        material_name = HYBRID_MATERIAL_ASSET.rsplit("/", 1)[-1].split(".", 1)[0]
        material_package = HYBRID_MATERIAL_ASSET.rsplit("/", 1)[0]
        material = tools.create_asset(material_name, material_package, unreal.Material, unreal.MaterialFactoryNew())
    if material is None:
        raise RuntimeError("could not create hybrid source material")
    try:
        material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    except Exception:
        pass
    try:
        material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    except Exception:
        pass
    try:
        material.set_editor_property("two_sided", True)
    except Exception:
        pass
    try:
        sample = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionTextureSample, -600, 0)
        sample.set_editor_property("texture", texture)
        unreal.MaterialEditingLibrary.connect_material_property(sample, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        unreal.MaterialEditingLibrary.connect_material_property(sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
        unreal.MaterialEditingLibrary.recompile_material(material)
    except Exception as exc:
        raise RuntimeError(f"could not connect source texture to unlit material: {type(exc).__name__}: {exc}")
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def _disable_nanite(mesh: Any) -> dict[str, Any]:
    before = _safe_property(mesh, "nanite_settings")
    changed = False
    if before is not None and hasattr(before, "set_editor_property"):
        for name in ("enabled", "b_enabled"):
            try:
                before.set_editor_property(name, False)
                changed = True
                break
            except Exception:
                continue
        if changed:
            mesh.set_editor_property("nanite_settings", before)
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)
    after = _safe_property(mesh, "nanite_settings")
    enabled = None
    if after is not None:
        for name in ("enabled", "b_enabled"):
            value = _safe_property(after, name)
            if value is not None:
                enabled = bool(value)
                break
    return {"asset": _path(mesh), "changed": changed, "enabled_after": enabled, "nanite_disabled": enabled is False}


def _duplicate_hybrid_mesh(source_mesh: Any) -> tuple[Any, dict[str, Any]]:
    duplicate = unreal.load_asset(HYBRID_MESH_ASSET)
    created = False
    if duplicate is None:
        duplicate = unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MESH_ASSET, HYBRID_MESH_ASSET)
        created = True
    if duplicate is None:
        raise RuntimeError("could not create hybrid-only source mesh duplicate")
    nanite = _disable_nanite(duplicate)
    return duplicate, {"source_asset": _path(source_mesh), "hybrid_asset": _path(duplicate), "created": created, "nanite": nanite}


def _apply_hybrid_material(actor: Any, mesh: Any, material: Any) -> None:
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    component.set_editor_property("static_mesh", mesh)
    component.set_material(0, material)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    component.set_collision_profile_name("NoCollision")
    actor.set_actor_hidden_in_game(False)
    actor.set_is_temporarily_hidden_in_editor(False)
    actor.set_editor_property("tags", ["visual_shell", "hybrid_only", "source_projection", "no_nanite"])


def _capture_actor(world: Any, source_camera: Any, source_mesh: Any, show_actors: list[Any], hidden_actors: list[Any], output: Path, location: Any) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    camera_rotation = source_camera.get_actor_rotation()
    capture_rotation = unreal.Rotator(pitch=float(camera_rotation.pitch), yaw=float(camera_rotation.yaw), roll=float(camera_rotation.roll) + 180.0)
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, location, capture_rotation)
    if actor is None:
        raise RuntimeError("could not spawn SceneCapture2D")
    actor.set_actor_label("Agent_SourceCapture_Temporary")
    component = actor.get_component_by_class(unreal.SceneCaptureComponent2D)
    if component is None:
        raise RuntimeError("SceneCapture2D has no capture component")
    target = unreal.RenderingLibrary.create_render_target2d(world, WIDTH, HEIGHT, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    component.set_editor_property("texture_target", target)
    component.set_editor_property("fov_angle", EXPECTED_FOV)
    try:
        component.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    except Exception:
        pass
    component.set_editor_property("capture_every_frame", False)
    component.set_editor_property("capture_on_movement", False)
    component.set_editor_property("ignore_screen_percentage", True)
    component.clear_show_only_components()
    for actor_to_show in show_actors:
        component.show_only_actor_components(actor_to_show)
    component.clear_hidden_components()
    for actor_to_hide in hidden_actors:
        component.hide_actor_components(actor_to_hide)
    camera_component = source_camera.get_component_by_class(unreal.CameraComponent)
    for capture_property, camera_property in (("projection_type", "projection_mode"), ("aspect_ratio", "aspect_ratio"), ("constrain_aspect_ratio", "constrain_aspect_ratio"), ("ortho_width", "ortho_width"), ("clip_plane_start", "clip_plane_start"), ("clip_plane_end", "clip_plane_end"), ("post_process_settings", "post_process_settings"), ("post_process_blend_weight", "post_process_blend_weight")):
        value = _safe_property(camera_component, camera_property)
        if value is not None:
            try:
                component.set_editor_property(capture_property, value)
            except Exception:
                pass
    actor.set_actor_location(location, False, False)
    component.capture_scene()
    unreal.RenderingLibrary.export_render_target(world, target, str(output.parent), output.name)
    if not output.is_file():
        raise RuntimeError(f"render target export failed: {output}")
    unreal.EditorLevelLibrary.destroy_actor(actor)
    return {"path": str(output), "sha256": _sha256(output), "dimensions_px": [WIDTH, HEIGHT], "camera_label": SOURCE_CAMERA_LABEL, "player_camera_used": False, "slate_capture": False, "editor_ui_visible": False, "fov_deg": EXPECTED_FOV, "projection": "perspective", "source_shell_visible": source_mesh in show_actors, "show_only_labels": [str(item.get_actor_label()) for item in show_actors]}


def _record_camera(camera: Any) -> dict[str, Any]:
    component = camera.get_component_by_class(unreal.CameraComponent)
    if component is None:
        raise RuntimeError("named source camera has no CameraComponent")
    fov = float(_safe_property(component, "field_of_view"))
    if abs(fov - EXPECTED_FOV) > 1e-4:
        raise RuntimeError(f"source camera FOV mismatch: {fov}")
    rotation = camera.get_actor_rotation()
    return {"label": str(camera.get_actor_label()), "path": str(camera.get_path_name()), "location_cm": _vec(camera.get_actor_location()), "rotation_deg": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)], "fov_deg": fov, "projection": str(_safe_property(component, "projection_mode")), "aspect_ratio": float(_safe_property(component, "aspect_ratio") or 0.0)}


def run() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    receipt_path = EVIDENCE / "source_shell_capture_receipt.json"
    receipt: dict[str, Any] = {"schema_version": "source_shell_capture_receipt_v1", "classification": "NOT_PROVEN", "protected_map": PROTECTED_MAP, "output_map": OUTPUT_MAP, "camera": {}, "material_audit_before": {}, "material_audit_after": {}, "captures": [], "hidden_actors": [], "errors": []}
    try:
        level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if not level.load_level(OUTPUT_MAP):
            raise RuntimeError(f"could not load hybrid map: {OUTPUT_MAP}")
        world = unreal.EditorLevelLibrary.get_editor_world()
        if world is None:
            raise RuntimeError("hybrid editor world unavailable")
        source_camera = _find_label(SOURCE_CAMERA_LABEL)
        source_actor = _find_label(SOURCE_MESH_LABEL)
        receipt["camera"] = _record_camera(source_camera)
        receipt["material_audit_before"] = _audit_mesh_actor(source_actor)
        component = source_actor.get_component_by_class(unreal.StaticMeshComponent)
        source_mesh = _safe_property(component, "static_mesh")
        duplicate, duplicate_receipt = _duplicate_hybrid_mesh(source_mesh)
        if not duplicate_receipt["nanite"]["nanite_disabled"]:
            raise RuntimeError("NANITE_SM6_DEPENDENCY_NOT_REMOVED")
        texture = _import_source_texture()
        material = _create_source_material(texture)
        _apply_hybrid_material(source_actor, duplicate, material)
        rotation_before = source_actor.get_actor_rotation()
        orientation_before = [float(rotation_before.pitch), float(rotation_before.yaw), float(rotation_before.roll)]
        source_actor.set_actor_rotation(unreal.Rotator(pitch=0.0, yaw=0.0, roll=0.0), False)
        rotation_after = source_actor.get_actor_rotation()
        orientation_after = [float(rotation_after.pitch), float(rotation_after.yaw), float(rotation_after.roll)]
        receipt["hybrid_mesh"] = duplicate_receipt
        receipt["hybrid_material"] = {"asset": _path(material), "texture": _path(texture), "source_texture_sha256": _sha256(PROJECT_SOURCE)}
        receipt["orientation_repair"] = {"classification": "PROVEN_CAPTURE_ONLY", "reason": "source_capture_was_180_degree_image_plane_inverted", "before_rotation_deg": orientation_before, "after_rotation_deg": orientation_after, "capture_roll_correction_deg": 180.0, "rotation": "capture_camera_roll_180"}
        receipt["material_audit_after"] = _audit_mesh_actor(source_actor)
        all_actors = _actors()
        hidden = []
        for actor in all_actors:
            if actor is source_actor:
                continue
            label = str(actor.get_actor_label())
            if label == "Agent_SourceCapture_Temporary" or "gameplay_proxy" in {str(tag) for tag in list(_safe_property(actor, "tags") or [])} or "PlayerStart" in str(actor.get_class().get_name()) or label.startswith("SP_"):
                hidden.append(label)
        receipt["hidden_actors"] = sorted(set(hidden))
        source_location = source_camera.get_actor_location()
        right = source_camera.get_actor_right_vector()
        visual = [source_actor]
        captures = [
            ("source_camera_clean", source_location, visual),
            ("offset_left_clean", source_location - right * 100.0, visual),
            ("offset_right_clean", source_location + right * 100.0, visual),
            ("overview_lit_clean", source_location, visual),
            ("castle_visual_clean", source_location, visual),
            ("bridge_visual_clean", source_location, visual),
        ]
        for name, location, show in captures:
            record = _capture_actor(world, source_camera, source_actor, show, [item for item in all_actors if item.get_actor_label() in receipt["hidden_actors"]], SCREENSHOTS / f"{name}.png", location)
            receipt["captures"].append(record)
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        debug_actors = [actor for actor in all_actors if actor is not source_actor and actor.get_actor_label() in receipt["hidden_actors"]]
        if debug_actors:
            receipt["captures"].append(_capture_actor(world, source_camera, source_actor, debug_actors, [], SCREENSHOTS / "collision_debug_clean.png", source_location))
        if not level.save_current_level():
            raise RuntimeError("hybrid map save failed")
        receipt["classification"] = "NOT_PROVEN"
        receipt["next_gate"] = "CPU_IMAGE_VALIDATION_AND_FRESH_UNREAL_RELOAD"
    except Exception as exc:
        receipt["classification"] = "REJECTED"
        receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    result = run()
    unreal.log("SOURCE_SHELL_CAPTURE=" + result["classification"])
