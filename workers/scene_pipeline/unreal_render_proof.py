"""Capture visual proof from the already-saved Unreal scene."""

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


def capture(name: str, actor: object) -> str:
    destination = ROOT / ("unreal_" + name + ".png")
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(actor.get_actor_location(), actor.get_actor_rotation())
    saved = False
    unreal.SystemLibrary.execute_console_command(None, "HighResShot 1280x720")
    time.sleep(4.0)
    candidates = sorted(
        Path(unreal.SystemLibrary.get_project_directory()).glob("Saved/Screenshots/**/*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if candidates:
        destination.write_bytes(candidates[-1].read_bytes())
        saved = True
    return str(destination) if saved and destination.is_file() else ""


def main() -> None:
    if not unreal.EditorLevelLibrary.load_level(MAP):
        raise RuntimeError("UNREAL_RENDER_MAP_LOAD_FAILED")
    labels = {actor.get_actor_label(): actor for actor in unreal.EditorLevelLibrary.get_all_level_actors()}
    names = {
        "source": "Castlegrounds_Camera_Source",
        "left": "Castlegrounds_Camera_Left",
        "right": "Castlegrounds_Camera_Right",
        "forward": "Castlegrounds_Camera_Forward",
        "elevated": "Castlegrounds_Camera_Elevated",
    }
    renders = []
    for name, label in names.items():
        actor = labels.get(label)
        if actor is None:
            raise RuntimeError("UNREAL_RENDER_CAMERA_MISSING:" + label)
        renders.append({"name": name, "path": capture(name, actor), "camera_label": label})
    wireframe = labels["Castlegrounds_Camera_Source"]
    unreal.SystemLibrary.execute_console_command(None, "viewmode wireframe")
    renders.append({"name": "wireframe", "path": capture("wireframe", wireframe), "camera_label": "Castlegrounds_Camera_Source"})
    unreal.SystemLibrary.execute_console_command(None, "viewmode lit")
    write_json("unreal_render_proof.json", {
        "schema": "unreal_image_to_scene_render_proof_v1",
        "classification": "UNREAL_VISUAL_RENDER_PROVEN" if all(item["path"] for item in renders) else "UNREAL_VISUAL_RENDER_REJECTED",
        "map": MAP,
        "resolution": [1280, 720],
        "renders": renders,
        "source_plane_present": False,
    })
    unreal.SystemLibrary.quit_editor()


main()
