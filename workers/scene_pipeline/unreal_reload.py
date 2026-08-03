"""Fresh-process reload proof for the MoGe image-to-scene map."""

from __future__ import annotations

import json
import time
from pathlib import Path

import unreal


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
MAP = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke"


def write_json(name: str, payload: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    loaded = unreal.EditorLevelLibrary.load_level(MAP)
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    labels = {actor.get_actor_label(): actor for actor in actors}
    required = [
        "Castlegrounds_ReconstructedMesh",
        "Castlegrounds_Camera_Source",
        "Castlegrounds_Camera_Left",
        "Castlegrounds_Camera_Right",
        "Castlegrounds_Camera_Forward",
        "Castlegrounds_Camera_Elevated",
        "Castlegrounds_Sun",
        "Castlegrounds_SkyLight",
    ]
    missing = [label for label in required if label not in labels]
    mesh_actor = labels.get("Castlegrounds_ReconstructedMesh")
    mesh_component = mesh_actor.get_component_by_class(unreal.StaticMeshComponent) if mesh_actor else None
    mesh = mesh_component.get_editor_property("static_mesh") if mesh_component else None
    source_camera = labels.get("Castlegrounds_Camera_Source")
    fov = source_camera.get_component_by_class(unreal.CameraComponent).get_editor_property("field_of_view") if source_camera else None
    receipt = {
        "schema": "unreal_image_to_scene_reload_v1",
        "classification": "SCENE_SAVE_RELOAD_PROVEN" if loaded and not missing and mesh else "SCENE_SAVE_RELOAD_REJECTED",
        "map": MAP,
        "map_loaded": bool(loaded),
        "missing_labels": missing,
        "mesh_asset_after_reload": mesh.get_path_name() if mesh else None,
        "mesh_present_after_reload": mesh is not None,
        "source_camera_fov": fov,
        "actor_count": len(actors),
        "source_plane_present": "SceneSmoke_ImageSurface" in labels,
    }
    write_json("map_reload_receipt.json", receipt)
    if not loaded or missing or mesh is None:
        raise RuntimeError("UNREAL_IMAGE_TO_SCENE_RELOAD_FAILED")
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(source_camera.get_actor_location(), source_camera.get_actor_rotation())
    unreal.SystemLibrary.execute_console_command(None, "HighResShot 1280x720")
    time.sleep(3.0)
    candidates = sorted(Path(unreal.SystemLibrary.get_project_directory()).glob("Saved/Screenshots/**/*.png"), key=lambda path: path.stat().st_mtime)
    if candidates:
        (ROOT / "reload_source.png").write_bytes(candidates[-1].read_bytes())
    unreal.SystemLibrary.quit_editor()


main()
