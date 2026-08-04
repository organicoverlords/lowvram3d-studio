"""Narrow Unreal MCP toolset for deterministic visual capture.

The toolset deliberately exposes no arbitrary Python, console-command, or
process-control entry point.  All writes are scoped to the named camera,
hybrid capture profiles, generated sequence assets, and project Saved output.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import unreal
import toolset_registry


CAMERA_LABEL = "Castlegrounds_Camera_Source"
SOURCE_MESH_LABEL = "Castlegrounds_ReconstructedMesh"
EXPECTED_FOV = 66.50838470458984
EXPECTED_WIDTH = 1448
EXPECTED_HEIGHT = 1086
EXPECTED_ASPECT = EXPECTED_WIDTH / EXPECTED_HEIGHT
ALLOWED_PROFILES = {"source_visual_only", "generated_visual_scene", "gameplay_debug", "collision_debug", "navigation_debug"}
_profile_receipts: dict[str, dict[str, Any]] = {}
_mrq_executor: Any = None


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _actors() -> list[Any]:
    return list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())


def _actor(label: str) -> Any:
    matches = [item for item in _actors() if str(item.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one actor labeled {label}, found {len(matches)}")
    return matches[0]


def _path(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(value.get_path_name())
    except Exception:
        return str(value)


def _safe(obj: Any, name: str) -> Any:
    try:
        return obj.get_editor_property(name)
    except Exception:
        return None


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _camera_record(camera: Any) -> dict[str, Any]:
    component = camera.get_component_by_class(unreal.CameraComponent)
    if component is None:
        raise RuntimeError("camera has no CameraComponent")
    fov = float(_safe(component, "field_of_view"))
    if abs(fov - EXPECTED_FOV) > 1e-4:
        raise RuntimeError(f"authoritative FOV mismatch: {fov}")
    rotation = camera.get_actor_rotation()
    return {
        "label": str(camera.get_actor_label()),
        "path": str(camera.get_path_name()),
        "location_cm": _vector(camera.get_actor_location()),
        "rotation_deg": [float(rotation.roll), float(rotation.pitch), float(rotation.yaw)],
        "fov_deg": fov,
        "projection_mode": str(_safe(component, "projection_mode")),
        "aspect_ratio": float(_safe(component, "aspect_ratio") or 0.0),
        "constrain_aspect_ratio": bool(_safe(component, "constrain_aspect_ratio")),
    }


def _safe_output(path: str) -> str:
    requested = Path(path)
    if not requested.is_absolute():
        requested = Path(unreal.Paths.project_saved_dir()) / requested
    requested = requested.resolve()
    saved = Path(unreal.Paths.project_saved_dir()).resolve()
    if saved not in requested.parents and requested != saved:
        raise RuntimeError("output_path must remain under the Unreal project Saved directory")
    requested.parent.mkdir(parents=True, exist_ok=True)
    return str(requested)


def _sha256(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tags(actor: Any) -> set[str]:
    return {str(item) for item in list(_safe(actor, "tags") or [])}


def _hidden_profile(profile_name: str, actor: Any) -> bool:
    tags = _tags(actor)
    label = str(actor.get_actor_label())
    if profile_name == "source_visual_only":
        return bool(tags & {"gameplay_proxy", "navigation_proof", "collision_only", "debug_geometry"}) or label.startswith("SP_") or "PlayerStart" in str(actor.get_class().get_name())
    if profile_name == "generated_visual_scene":
        return bool(tags & {"collision_only", "navigation_proof", "debug_geometry"})
    if profile_name in {"gameplay_debug", "collision_debug", "navigation_debug"}:
        return False
    raise RuntimeError(f"unknown visibility profile: {profile_name}")


def _write_receipt(name: str, value: dict[str, Any]) -> str:
    path = Path(unreal.Paths.project_saved_dir()) / "AgentProof" / "LowVRAM3DSceneTools" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value) + "\n", encoding="utf-8")
    return str(path)


@unreal.uclass()
class LowVRAM3DSceneTools(unreal.ToolsetDefinition):
    """Allowlisted named-camera, material-audit, and MRQ controls."""

    @toolset_registry.tool_call
    @staticmethod
    def get_camera_contract(camera_label: str) -> str:
        if camera_label != CAMERA_LABEL:
            return _json({"classification": "REJECTED", "error": "camera label is not allowlisted"})
        return _json({"classification": "PROVEN", "camera": _camera_record(_actor(camera_label)), "contract": {"width": EXPECTED_WIDTH, "height": EXPECTED_HEIGHT, "aspect_ratio": EXPECTED_ASPECT, "fov_deg": EXPECTED_FOV, "player_camera_dependency": False}})

    @toolset_registry.tool_call
    @staticmethod
    def capture_named_camera_fast(camera_label: str, width: int, height: int, output_path: str, visibility_profile: str = "source_visual_only") -> str:
        if camera_label != CAMERA_LABEL:
            return _json({"classification": "REJECTED", "error": "camera label is not allowlisted"})
        if (int(width), int(height)) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            return _json({"classification": "REJECTED", "error": "fast capture requires the authoritative 1448x1086 contract"})
        output = _safe_output(output_path)
        camera = _actor(camera_label)
        source = _actor(SOURCE_MESH_LABEL)
        world = unreal.EditorLevelLibrary.get_editor_world()
        if world is None:
            raise RuntimeError("editor world unavailable")
        hidden = [item for item in _actors() if _hidden_profile(visibility_profile, item) and item is not source]
        capture = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, camera.get_actor_location(), camera.get_actor_rotation())
        if capture is None:
            raise RuntimeError("could not create SceneCapture2D")
        try:
            component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
            target = unreal.RenderingLibrary.create_render_target2d(world, width, height, unreal.TextureRenderTargetFormat.RTF_RGBA8)
            capture.set_actor_transform(camera.get_actor_transform(), False, False)
            camera_rotation = camera.get_actor_rotation()
            capture.set_actor_rotation(unreal.Rotator(pitch=float(camera_rotation.pitch), yaw=float(camera_rotation.yaw), roll=float(camera_rotation.roll) + 180.0), False)
            component.set_editor_property("texture_target", target)
            component.clear_show_only_components()
            component.show_only_actor_components(source)
            component.set_editor_property("capture_every_frame", False)
            component.set_editor_property("capture_on_movement", False)
            component.set_editor_property("ignore_screen_percentage", True)
            component.set_editor_property("fov_angle", EXPECTED_FOV)
            component.clear_hidden_components()
            for hidden_actor in hidden:
                component.hide_actor_components(hidden_actor)
            camera_component = camera.get_component_by_class(unreal.CameraComponent)
            for capture_property, camera_property in (("projection_type", "projection_mode"), ("aspect_ratio", "aspect_ratio"), ("constrain_aspect_ratio", "constrain_aspect_ratio"), ("ortho_width", "ortho_width"), ("clip_plane_start", "clip_plane_start"), ("clip_plane_end", "clip_plane_end"), ("post_process_settings", "post_process_settings"), ("post_process_blend_weight", "post_process_blend_weight")):
                value = _safe(camera_component, camera_property)
                if value is not None:
                    try:
                        component.set_editor_property(capture_property, value)
                    except Exception:
                        pass
            component.capture_scene()
            unreal.RenderingLibrary.export_render_target(world, target, str(Path(output).parent), Path(output).name)
            if not os.path.isfile(output):
                raise RuntimeError("render-target export failed")
        finally:
            unreal.EditorLevelLibrary.destroy_actor(capture)
        receipt = {"schema_version": "lowvram3d_fast_capture_v1", "classification": "REQUESTED", "camera": _camera_record(camera), "output_path": output, "width": width, "height": height, "aspect_ratio": width / height, "visibility_profile": visibility_profile, "hidden_actor_labels": sorted(str(item.get_actor_label()) for item in hidden), "player_camera_used": False, "slate_capture": False, "image_sha256": _sha256(output)}
        receipt["receipt_path"] = _write_receipt("fast_capture_receipt.json", receipt)
        return _json(receipt)

    @toolset_registry.tool_call
    @staticmethod
    def create_camera_cut_sequence(camera_label: str, sequence_path: str) -> str:
        if camera_label != CAMERA_LABEL:
            return _json({"classification": "REJECTED", "error": "camera label is not allowlisted"})
        if not sequence_path.startswith("/Game/AgentProof/"):
            return _json({"classification": "REJECTED", "error": "sequence path is outside the proof namespace"})
        try:
            tools = unreal.AssetToolsHelpers.get_asset_tools()
            package = sequence_path.rsplit("/", 1)[0]
            name = sequence_path.rsplit("/", 1)[-1]
            sequence = unreal.load_asset(sequence_path) or tools.create_asset(name, package, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
            if sequence is None:
                raise RuntimeError("could not create LevelSequence")
            sequence.set_playback_end(1)
            movie_scene = sequence.get_movie_scene()
            binding = sequence.add_possessable(_actor(camera_label))
            track = movie_scene.add_track(unreal.MovieSceneCameraCutTrack)
            section = track.add_section()
            section.set_range(0, 1)
            section.set_editor_property("camera_binding_id", binding.get_id())
            unreal.EditorAssetLibrary.save_loaded_asset(sequence)
            result = {"classification": "PROVEN", "sequence_path": str(sequence.get_path_name()), "camera_label": camera_label, "frames": [0, 1]}
        except Exception as exc:
            result = {"classification": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}
        result["receipt_path"] = _write_receipt("camera_cut_sequence_receipt.json", result)
        return _json(result)

    @toolset_registry.tool_call
    @staticmethod
    def render_named_camera_mrq(camera_label: str, width: int, height: int, output_path: str) -> str:
        if camera_label != CAMERA_LABEL or (int(width), int(height)) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            return _json({"classification": "REJECTED", "error": "camera or resolution is outside the authoritative contract"})
        output = _safe_output(output_path)
        sequence_path = "/Game/AgentProof/ImageToSceneSmoke_20260803/Sequences/LS_Castlegrounds_Source_OneFrame"
        sequence_result = json.loads(LowVRAM3DSceneTools.create_camera_cut_sequence(camera_label, sequence_path))
        if sequence_result.get("classification") != "PROVEN":
            result = {"classification": "REJECTED", "camera_label": camera_label, "width": width, "height": height, "output_path": output, "sequence": sequence_result}
            result["receipt_path"] = _write_receipt("mrq_capture_receipt.json", result)
            return _json(result)
        try:
            global _mrq_executor
            subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
            queue = subsystem.get_queue()
            queue.delete_all_jobs()
            job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
            job.set_editor_property("sequence", unreal.load_asset(sequence_path))
            config = job.get_configuration()
            output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
            output_setting.set_editor_property("output_directory", unreal.DirectoryPath(str(Path(output).parent)))
            output_setting.set_editor_property("output_resolution", unreal.IntPoint(int(width), int(height)))
            output_setting.set_editor_property("file_name_format", Path(output).stem + ".{frame_number}")
            config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
            _mrq_executor = unreal.MoviePipelineInProcessExecutor()
            subsystem.render_queue_with_executor_instance(_mrq_executor)
            result = {"classification": "REQUESTED", "camera_label": camera_label, "width": width, "height": height, "output_path": output, "sequence_path": sequence_path, "player_camera_used": False, "async": True, "next_validation": "wait_for_mrq_output_then_validate_capture_image"}
        except Exception as exc:
            result = {"classification": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "sequence": sequence_result}
        result["receipt_path"] = _write_receipt("mrq_capture_receipt.json", result)
        return _json(result)

    @toolset_registry.tool_call
    @staticmethod
    def apply_visual_capture_profile(profile_name: str) -> str:
        before = {}
        after = {}
        for actor in _actors():
            label = str(actor.get_actor_label())
            before[label] = {"hidden_in_game": bool(actor.is_hidden()), "temporary_hidden": bool(actor.is_temporarily_hidden_in_editor())}
            hidden = _hidden_profile(profile_name, actor)
            actor.set_is_temporarily_hidden_in_editor(hidden)
            after[label] = {"hidden_in_game": bool(actor.is_hidden()), "temporary_hidden": hidden}
        receipt = {"classification": "PROVEN", "profile_name": profile_name, "before": before, "after": after}
        receipt["receipt_path"] = _write_receipt("visibility_profile_receipt.json", receipt)
        _profile_receipts[receipt["receipt_path"]] = receipt
        return _json(receipt)

    @toolset_registry.tool_call
    @staticmethod
    def restore_capture_visibility(receipt_path: str) -> str:
        receipt = _profile_receipts.get(receipt_path)
        if receipt is None:
            return _json({"classification": "REJECTED", "error": "visibility receipt is not owned by this session"})
        for actor in _actors():
            previous = receipt["before"].get(str(actor.get_actor_label()))
            if previous is not None:
                actor.set_is_temporarily_hidden_in_editor(bool(previous["temporary_hidden"]))
        return _json({"classification": "PROVEN", "restored_receipt": receipt_path})

    @toolset_registry.tool_call
    @staticmethod
    def audit_actor_materials(actor_label: str) -> str:
        actor = _actor(actor_label)
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component is None:
            return _json({"classification": "REJECTED", "error": "actor has no StaticMeshComponent"})
        slots = []
        for index in range(int(component.get_num_materials())):
            material = component.get_material(index)
            path = _path(material) or ""
            slot = {"slot": index, "path": path, "class": str(material.get_class().get_name()) if material else None, "engine_placeholder": path.startswith("/Engine/") or "worldgrid" in path.lower() or "preview" in path.lower(), "blend_mode": str(_safe(material, "blend_mode")) if material else None, "shading_model": str(_safe(material, "shading_model")) if material else None, "texture_paths": [], "texture_dimensions": []}
            if material is not None:
                try:
                    for parameter_name in unreal.MaterialEditingLibrary.get_texture_parameter_names(material):
                        texture = unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material, parameter_name)
                        if texture is not None:
                            texture_path = _path(texture)
                            slot["texture_paths"].append(texture_path)
                            slot["texture_dimensions"].append({"path": texture_path, "width": int(_safe(texture, "size_x") or 0), "height": int(_safe(texture, "size_y") or 0), "srgb": bool(_safe(texture, "srgb"))})
                except Exception:
                    pass
            slots.append(slot)
        result = {"classification": "PROVEN" if slots and not any(item["engine_placeholder"] for item in slots) else "REJECTED", "actor_label": actor_label, "mesh_asset": _path(_safe(component, "static_mesh")), "slots": slots}
        result["receipt_path"] = _write_receipt("material_audit_receipt.json", result)
        return _json(result)

    @toolset_registry.tool_call
    @staticmethod
    def audit_source_shell_projection(actor_label: str, camera_label: str) -> str:
        if camera_label != CAMERA_LABEL or actor_label != SOURCE_MESH_LABEL:
            return _json({"classification": "REJECTED", "error": "only the authoritative source shell/camera pair is allowlisted"})
        result = {"classification": "NOT_PROVEN", "actor_label": actor_label, "camera": _camera_record(_actor(camera_label)), "material_audit": json.loads(LowVRAM3DSceneTools.audit_actor_materials(actor_label)), "requirements": ["unlit source projection", "no preview/default material", "nanite disabled on hybrid duplicate"]}
        result["receipt_path"] = _write_receipt("source_shell_projection_audit_receipt.json", result)
        return _json(result)

    @toolset_registry.tool_call
    @staticmethod
    def validate_capture_image(image_path: str, expected_width: int, expected_height: int) -> str:
        path = _safe_output(image_path)
        result = {"classification": "NOT_PROVEN", "path": path, "exists": os.path.isfile(path), "bytes": os.path.getsize(path) if os.path.isfile(path) else 0, "sha256": _sha256(path), "expected_dimensions": [expected_width, expected_height], "dimensions": None, "defects": ["embedded Unreal Python image decoder not available; CPU validator required"]}
        try:
            from PIL import Image
            with Image.open(path) as image:
                result["dimensions"] = list(image.size)
                if tuple(image.size) != (int(expected_width), int(expected_height)):
                    result["defects"] = ["DIMENSIONS_MISMATCH"]
                else:
                    result["classification"] = "PROVEN"
        except Exception as exc:
            result["defects"].append(type(exc).__name__)
        result["receipt_path"] = _write_receipt("capture_validation_receipt.json", result)
        return _json(result)
