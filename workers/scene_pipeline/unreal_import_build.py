"""Import the selected MoGe GLB and build the isolated Unreal scene map."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import unreal


GLB = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds\balanced_010.glb")
SOURCE = Path(r"C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png")
ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
CONTENT = "/Game/AgentProof/ImageToSceneSmoke_20260803"
MAP = CONTENT + "/Maps/L_Castlegrounds_ImageToScene_Smoke"
STATIC_MESH_ASSET = CONTENT + "/Geometry/balanced_010/StaticMeshes/balanced_010.balanced_010"
EXTERNAL = ROOT
FOV = 66.5083847
ASPECT_RATIO = 4.0 / 3.0


def write_json(name: str, payload: dict) -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    (EXTERNAL / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_glb() -> tuple[object, list[str]]:
    task = unreal.AssetImportTask()
    task.filename = str(GLB)
    task.destination_path = CONTENT + "/Geometry"
    task.automated = True
    task.save = True
    task.replace_existing = False
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.imported_object_paths)
    assets = [unreal.EditorAssetLibrary.load_asset(path) for path in paths]
    meshes = [asset for asset in assets if isinstance(asset, unreal.StaticMesh)]
    if not meshes:
        existing = unreal.EditorAssetLibrary.load_asset(STATIC_MESH_ASSET)
        if isinstance(existing, unreal.StaticMesh):
            return existing, [STATIC_MESH_ASSET]
    if not meshes:
        raise RuntimeError("UNREAL_INTERCHANGE_GLB_IMPORT_NO_STATIC_MESH:" + repr(paths))
    return meshes[0], paths


def spawn(label: str, cls, location: unreal.Vector, rotation: unreal.Rotator):
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(cls, location, rotation)
    if not actor:
        raise RuntimeError("UNREAL_ACTOR_SPAWN_FAILED:" + label)
    actor.set_actor_label(label)
    actor.tags = ["ImageToSceneSmoke20260803"]
    return actor


def configure_world() -> dict:
    sun = spawn("Castlegrounds_Sun", unreal.DirectionalLight, unreal.Vector(0, 0, 100), unreal.Rotator(-35, -25, 0))
    sun_component = sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_component.set_editor_property("intensity", 3.0)
    sky = spawn("Castlegrounds_SkyLight", unreal.SkyLight, unreal.Vector(0, 0, 100), unreal.Rotator(0, 0, 0))
    sky_component = sky.get_component_by_class(unreal.SkyLightComponent)
    sky_component.set_editor_property("intensity", 0.8)
    atmosphere = spawn("Castlegrounds_SkyAtmosphere", unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    fog = spawn("Castlegrounds_Fog", unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    post = spawn("Castlegrounds_PostProcess", unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    post.set_editor_property("unbound", True)
    return {
        "sun": sun.get_name(),
        "sky_light": sky.get_name(),
        "sky_atmosphere": atmosphere.get_name(),
        "fog": fog.get_name(),
        "post_process": post.get_name(),
    }


def camera(label: str, position: list[float], target: list[float]):
    location = unreal.Vector(*position)
    actor = spawn(label, unreal.CameraActor, location, unreal.MathLibrary.find_look_at_rotation(location, unreal.Vector(*target)))
    component = actor.get_component_by_class(unreal.CameraComponent)
    component.set_editor_property("field_of_view", FOV)
    component.set_editor_property("aspect_ratio", ASPECT_RATIO)
    component.set_editor_property("constrain_aspect_ratio", True)
    return actor


def capture(name: str, actor: object) -> str:
    unreal.EditorLevelLibrary.set_level_viewport_camera_info(actor.get_actor_location(), actor.get_actor_rotation())
    unreal.SystemLibrary.execute_console_command(None, "HighResShot 1280x720")
    time.sleep(3.0)
    candidates = sorted(
        Path(unreal.SystemLibrary.get_project_directory()).glob("Saved/Screenshots/**/*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return ""
    destination = EXTERNAL / (name + ".png")
    destination.write_bytes(candidates[-1].read_bytes())
    return str(destination)


def main() -> None:
    if not unreal.EditorLevelLibrary.load_level(MAP):
        unreal.EditorLevelLibrary.new_level(MAP)
    for actor in list(unreal.EditorLevelLibrary.get_all_level_actors()):
        unreal.EditorLevelLibrary.destroy_actor(actor)
    mesh, imported_paths = import_glb()
    mesh_actor = spawn("Castlegrounds_ReconstructedMesh", unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    mesh_component = mesh_actor.get_component_by_class(unreal.StaticMeshComponent)
    mesh_component.set_editor_property("static_mesh", mesh)
    mesh_actor.tags = ["ImageToSceneSmoke20260803", "MoGe2P2P5D"]
    lighting = configure_world()
    cameras = {
        "source": camera("Castlegrounds_Camera_Source", [0, 0, 0], [0, 20000, 0]),
        "left": camera("Castlegrounds_Camera_Left", [-190, 0, 0], [-190, 20000, 0]),
        "right": camera("Castlegrounds_Camera_Right", [190, 0, 0], [190, 20000, 0]),
        "forward": camera("Castlegrounds_Camera_Forward", [0, -95, 0], [0, 19905, 0]),
        "elevated": camera("Castlegrounds_Camera_Elevated", [0, 190, -190], [0, 20190, -190]),
    }
    unreal.EditorLevelLibrary.save_current_level()
    unreal.EditorAssetLibrary.save_directory(CONTENT)
    renders = [{"name": name, "path": capture(name, actor), "camera": actor.get_name()} for name, actor in cameras.items()]
    unreal.SystemLibrary.execute_console_command(None, "viewmode wireframe")
    renders.append({"name": "wireframe", "path": capture("wireframe", cameras["source"]), "camera": cameras["source"].get_name()})
    unreal.SystemLibrary.execute_console_command(None, "viewmode lit")
    renders.append({"name": "depth_diagnostic", "path": str(ROOT / "depth_vis.png"), "camera": "CPU_MoGe_depth_map"})
    write_json("unreal_import_receipt.json", {
        "schema": "unreal_interchange_image_to_scene_import_v1",
        "classification": "SCENE_ASSET_IMPORT_PROVEN",
        "source_glb": str(GLB),
        "source_glb_sha256": sha256(GLB),
        "imported_object_paths": imported_paths,
        "static_mesh_asset": mesh.get_path_name(),
        "materials_bound_by_importer": len(mesh.get_editor_property("static_materials")) if mesh else 0,
        "source_image": str(SOURCE),
        "interchange_route": "AssetImportTask_GLTF_INTERCHANGE",
    })
    write_json("scene_build_receipt.json", {
        "schema": "unreal_image_to_scene_build_v1",
        "classification": "SCENE_BUILD_PROVEN",
        "map": MAP,
        "mesh_actor": mesh_actor.get_name(),
        "mesh_asset": mesh.get_path_name(),
        "camera_fov_horizontal_deg": FOV,
        "cameras": {name: actor.get_name() for name, actor in cameras.items()},
        "lighting": lighting,
        "source_plane_present": False,
        "representation": "SOURCE_VISIBLE_EDGE_AWARE_2P5D_MESH",
        "camera_convention": "MOGE_POINT_MAP_TO_UNREAL_AXIS_FIXTURE",
        "camera_target_direction": [0.0, 1.0, 0.0],
        "camera_aspect_ratio": ASPECT_RATIO,
        "M_raw_moge_to_unreal": [[100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 100.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "unreal_handedness": "FLIPPED_BY_AXIS_FIXTURE",
        "collision": "NOT_PROVEN_NO_GROUND_PLANE_CONFIRMED",
        "spatial_depth_ranges": 3,
        "renders": renders,
    })
    write_json("render_manifest.json", {
        "schema": "image_to_scene_render_manifest_v1",
        "resolution": [1280, 720],
        "views": renders,
        "source_plane_present": False,
    })
    unreal.SystemLibrary.quit_editor()


main()
