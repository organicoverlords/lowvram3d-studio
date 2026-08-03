"""Capture saved Castlegrounds cameras in a real-RHI offscreen editor run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import unreal


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"
EXPECTED = "/Game/AgentProof/ImageToSceneSmoke_20260803/Geometry/CastlegroundsSourceMeshV2/"
PROJECT_SCREENSHOTS = Path(unreal.SystemLibrary.get_project_directory()) / "Saved" / "Screenshots"
NAMES = {"source": "Castlegrounds_Camera_Source", "left": "Castlegrounds_Camera_Left", "right": "Castlegrounds_Camera_Right", "forward": "Castlegrounds_Camera_Forward", "elevated": "Castlegrounds_Camera_Elevated"}


def write_json(payload: dict) -> None:
    (ROOT / "unreal_offscreen_capture_receipt.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not unreal.EditorLevelLibrary.load_level(MAP):
        raise RuntimeError("UNREAL_OFFSCREEN_MAP_LOAD_FAILED")
    actors = {actor.get_actor_label(): actor for actor in unreal.EditorLevelLibrary.get_all_level_actors()}
    mesh_actor = actors.get("Castlegrounds_ReconstructedMesh")
    component = mesh_actor.get_component_by_class(unreal.StaticMeshComponent) if mesh_actor else None
    mesh = component.get_editor_property("static_mesh") if component else None
    mesh_path = mesh.get_path_name() if mesh else None
    mesh_ok = bool(mesh_path and mesh_path.startswith(EXPECTED))
    PROJECT_SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    errors: list[str] = []
    for name, label in NAMES.items():
        camera = actors.get(label)
        if camera is None:
            errors.append(f"MISSING_CAMERA:{label}")
            continue
        before = {path: path.stat().st_mtime_ns for path in PROJECT_SCREENSHOTS.rglob("*.png") if path.is_file()}
        started = time.time_ns()
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(camera.get_actor_location(), camera.get_actor_rotation())
        unreal.SystemLibrary.execute_console_command(None, "viewmode lit")
        unreal.SystemLibrary.execute_console_command(None, "HighResShot 1280x960")
        selected = None
        deadline = time.time() + 12.0
        while time.time() < deadline:
            candidates = [path for path in PROJECT_SCREENSHOTS.rglob("*.png") if path.is_file() and path.stat().st_mtime_ns >= started and path not in before]
            if candidates:
                selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
                break
            time.sleep(0.25)
        if selected is None:
            errors.append(f"NO_NEW_PNG:{name}")
            continue
        destination = ROOT / f"unreal_offscreen_{name}.png"
        destination.write_bytes(selected.read_bytes())
        results.append({"name": name, "camera_label": label, "path": str(destination), "bytes": destination.stat().st_size, "source_timestamp_ns": selected.stat().st_mtime_ns})
    receipt = {"schema": "unreal_real_rhi_offscreen_capture_v1", "classification": "UNREAL_EXACT_SOURCE_CAMERA_RENDER_PROVEN" if mesh_ok and len(results) == len(NAMES) and not errors else "UNREAL_VISUAL_RENDER_BLOCKED", "map": MAP, "mesh_reference": mesh_path, "mesh_v2_reference_valid": mesh_ok, "route": "UnrealEditor_RenderOffScreen_DX11_HighResShot", "requested_resolution": [1280, 960], "renders": results, "errors": errors, "source_plane_present": "SceneSmoke_ImageSurface" in actors}
    write_json(receipt)
    unreal.SystemLibrary.quit_editor()


main()
