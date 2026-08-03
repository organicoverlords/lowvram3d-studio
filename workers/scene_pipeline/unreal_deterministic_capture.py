"""Attempt deterministic Unreal SceneCaptureComponent2D proof for the v2 map."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import unreal


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"
EXPECTED = "/Game/AgentProof/ImageToSceneSmoke_20260803/Geometry/CastlegroundsSourceMeshV2/"
WIDTH, HEIGHT = 1280, 960


def write_json(payload: dict) -> None:
    (ROOT / "unreal_deterministic_capture_receipt.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def file_stats(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "mtime": path.stat().st_mtime}


def main() -> None:
    if not unreal.EditorLevelLibrary.load_level(MAP):
        raise RuntimeError("UNREAL_CAPTURE_MAP_LOAD_FAILED")
    labels = {actor.get_actor_label(): actor for actor in unreal.EditorLevelLibrary.get_all_level_actors()}
    mesh_actor = labels.get("Castlegrounds_ReconstructedMesh")
    mesh_component = mesh_actor.get_component_by_class(unreal.StaticMeshComponent) if mesh_actor else None
    mesh = mesh_component.get_editor_property("static_mesh") if mesh_component else None
    mesh_ok = bool(mesh and mesh.get_path_name().startswith(EXPECTED))
    results: list[dict[str, object]] = []
    errors: list[str] = []
    cameras = {name: labels.get(label) for name, label in {"source": "Castlegrounds_Camera_Source", "left": "Castlegrounds_Camera_Left", "right": "Castlegrounds_Camera_Right", "forward": "Castlegrounds_Camera_Forward", "elevated": "Castlegrounds_Camera_Elevated"}.items()}
    try:
        capture_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        capture = capture_actor.get_component_by_class(unreal.SceneCaptureComponent2D)
        if capture is None:
            raise RuntimeError("UNREAL_SCENE_CAPTURE_COMPONENT_MISSING")
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        target = asset_tools.create_asset(
            "CastlegroundsDeterministicCaptureRT",
            "/Game/AgentProof/ImageToSceneSmoke_20260803/Capture",
            unreal.TextureRenderTarget2D,
            unreal.TextureRenderTarget2DFactoryNew(),
        )
        if target is None:
            raise RuntimeError("UNREAL_RENDER_TARGET_CREATE_FAILED")
        target.set_editor_property("render_target_format", unreal.TextureRenderTargetFormat.RTF_RGBA8)
        target.set_editor_property("size_x", WIDTH)
        target.set_editor_property("size_y", HEIGHT)
        capture.set_editor_property("texture_target", target)
        for name, camera in cameras.items():
            if camera is None:
                errors.append(f"MISSING_CAMERA:{name}"); continue
            capture_actor.set_actor_transform(camera.get_actor_transform(), False, False)
            capture.capture_scene()
            time.sleep(0.25)
            destination = ROOT / f"unreal_deterministic_{name}.png"
            try:
                unreal.RenderingLibrary.export_render_target(None, target, str(destination))
            except Exception as exc:
                errors.append(f"EXPORT:{name}:{type(exc).__name__}:{exc}")
            if destination.is_file():
                results.append({"name": name, "file": file_stats(destination)})
        unreal.EditorLevelLibrary.destroy_actor(capture_actor)
    except Exception as exc:
        errors.append(f"CAPTURE:{type(exc).__name__}:{exc}")
    receipt = {"schema": "unreal_deterministic_scene_capture_v1", "classification": "UNREAL_EXACT_SOURCE_CAMERA_RENDER_PROVEN" if mesh_ok and len(results) == len(cameras) and not errors else "UNREAL_VISUAL_RENDER_BLOCKED", "map": MAP, "mesh_v2_reference": mesh.get_path_name() if mesh else None, "mesh_v2_reference_valid": mesh_ok, "resolution": [WIDTH, HEIGHT], "route": "SceneCaptureComponent2D_TextureRenderTarget2D_RenderingLibrary.export_render_target", "renders": results, "errors": errors, "source_plane_present": "SceneSmoke_ImageSurface" in labels}
    write_json(receipt)
    unreal.SystemLibrary.quit_editor()


main()
