"""Deterministic off-screen renders from exact named CameraActors.

Independent of PIE, the player pawn, the editor viewport and window focus: a
temporary SceneCapture2D copies each camera's projection, renders one frame
into a transient render target, and the target is exported to PNG. Temporary
actors and targets are always destroyed, including on failure.

Assign `CAPTURE_REQUESTS` (a list of dicts) before running to drive several
cameras in one pass; otherwise the authoritative source camera is captured.
Visibility profiles are applied to the capture component only, so the level's
own actor visibility is never mutated.

    python -m uemcp python @unreal/capture_named_camera.py --json
"""

import json
import os

import unreal

SHOTS_DIR = (r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803"
             r"\evidence\latest-image-to-scene\screenshots")

REQUESTS = globals().get("CAPTURE_REQUESTS") or [{
    "camera_label": "Castlegrounds_Camera_Source",
    "output": os.path.join(SHOTS_DIR, "source_camera_clean.png"),
    "width": 1448,
    "height": 1086,
    "visibility_profile": "source_visual_only",
}]

SOURCE_SHELL = "Castlegrounds_ReconstructedMesh"

# The generated scene stand-ins are cubes sitting between the camera and the
# reconstructed shell; they must not occlude a source-view proof.
GENERATED_PREFIXES = (
    "architecture_", "terrain_", "vegetation_", "crossing_", "environment_",
    "SP_", "gameplay_", "nav_", "debug_",
)
DEBUG_CLASSES = ("NavMeshBoundsVolume", "RecastNavMesh", "PlayerStart")


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def should_hide(actor, profile: str) -> bool:
    label = str(actor.get_actor_label())
    class_name = str(actor.get_class().get_name())
    if label == SOURCE_SHELL:
        return profile == "generated_visual_scene"
    if profile == "source_visual_only":
        return class_name in DEBUG_CLASSES or label.startswith(GENERATED_PREFIXES)
    if profile == "generated_visual_scene":
        return class_name in DEBUG_CLASSES
    if profile == "gameplay_debug":
        return False
    raise RuntimeError(f"unknown visibility profile: {profile}")


subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.EditorLevelLibrary.get_editor_world()
receipts = []

for request in REQUESTS:
    camera_label = request["camera_label"]
    output_path = request["output"]
    width = int(request.get("width", 1448))
    height = int(request.get("height", 1086))
    profile = request.get("visibility_profile", "source_visual_only")

    actors = list(subsystem.get_all_level_actors())
    cameras = [a for a in actors if str(a.get_actor_label()) == camera_label]
    if len(cameras) != 1:
        raise RuntimeError(
            f"expected exactly one actor labelled {camera_label}, found {len(cameras)}")
    camera = cameras[0]

    camera_component = camera.get_component_by_class(unreal.CameraComponent)
    if camera_component is None:
        raise RuntimeError(f"{camera_label} has no CameraComponent")

    rotation = camera.get_actor_rotation()
    contract = {
        "label": camera_label,
        "path": str(camera.get_path_name()),
        "location": vec(camera.get_actor_location()),
        "rotation_pyr": [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)],
        "fov": float(camera_component.get_editor_property("field_of_view")),
        "aspect_ratio": float(camera_component.get_editor_property("aspect_ratio")),
        "constrain_aspect_ratio": bool(
            camera_component.get_editor_property("constrain_aspect_ratio")),
        "projection_mode": str(camera_component.get_editor_property("projection_mode")),
    }

    hidden = [a for a in actors if should_hide(a, profile)]
    capture_actor = None
    try:
        capture_actor = subsystem.spawn_actor_from_class(
            unreal.SceneCapture2D, camera.get_actor_location(), camera.get_actor_rotation())
        if capture_actor is None:
            raise RuntimeError("could not spawn SceneCapture2D")
        capture_actor.set_actor_label("__NamedCameraCapture")
        capture_actor.set_actor_transform(camera.get_actor_transform(), False, False)

        component = capture_actor.get_component_by_class(unreal.SceneCaptureComponent2D)
        target = unreal.RenderingLibrary.create_render_target2d(
            world, width, height, unreal.TextureRenderTargetFormat.RTF_RGBA8)
        component.set_editor_property("texture_target", target)

        # Render on demand only; capture_scene() below is the single explicit frame.
        component.set_editor_property("capture_every_frame", False)
        component.set_editor_property("capture_on_movement", False)
        component.set_editor_property("fov_angle", contract["fov"])
        try:
            component.set_editor_property(
                "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        except Exception:
            pass

        for capture_prop, camera_prop in (
            ("projection_type", "projection_mode"),
            ("ortho_width", "ortho_width"),
            ("post_process_settings", "post_process_settings"),
            ("post_process_blend_weight", "post_process_blend_weight"),
        ):
            try:
                component.set_editor_property(
                    capture_prop, camera_component.get_editor_property(camera_prop))
            except Exception:
                pass

        component.clear_hidden_components()
        for actor in hidden:
            component.hide_actor_components(actor)

        component.capture_scene()

        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        unreal.RenderingLibrary.export_render_target(
            world, target, directory, os.path.basename(output_path))

        if not os.path.isfile(output_path):
            raise RuntimeError(f"render target export produced no file at {output_path}")

        receipts.append({
            "classification": "CAPTURED",
            "camera": contract,
            "output": output_path,
            "size_bytes": os.path.getsize(output_path),
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "visibility_profile": profile,
            "hidden_actor_count": len(hidden),
            "player_camera_used": False,
            "pie_used": False,
            "slate_capture": False,
        })
    finally:
        if capture_actor is not None:
            subsystem.destroy_actor(capture_actor)

result = json.dumps(receipts if len(receipts) != 1 else receipts[0])
