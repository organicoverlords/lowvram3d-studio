"""High-level viewport control, built on the editor bridge.

Every verb here is synchronous and returns data an agent can assert on, so a
capture can be validated without a human looking at the screen.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .bridge import Bridge, BridgeError

Vec3 = tuple[float, float, float]


def _vec(value: Vec3 | dict[str, float]) -> dict[str, float]:
    if isinstance(value, dict):
        return {"x": float(value["x"]), "y": float(value["y"]), "z": float(value["z"])}
    x, y, z = value
    return {"x": float(x), "y": float(y), "z": float(z)}


def _rot(pitch: float, yaw: float, roll: float = 0.0) -> dict[str, float]:
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def look_at_rotation(eye: Vec3, target: Vec3) -> dict[str, float]:
    """Rotator that points ``eye`` at ``target`` in Unreal's left-handed Z-up frame."""
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return _rot(pitch, yaw)


class Viewport:
    def __init__(self, bridge: Bridge | None = None) -> None:
        self.bridge = bridge or Bridge()

    # -- state -------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Current editor viewport location, rotation and FOV."""
        return self.bridge.call("get_viewport_info")

    # -- movement ----------------------------------------------------------
    def set_camera(self, location: Vec3 | None = None,
                   rotation: dict[str, float] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if location is not None:
            params["location"] = _vec(location)
        if rotation is not None:
            params["rotation"] = rotation
        if not params:
            raise BridgeError("set_camera needs a location, a rotation, or both")
        return self.bridge.call("set_viewport_camera", params)

    def look_at(self, eye: Vec3, target: Vec3) -> dict[str, Any]:
        """Place the camera at ``eye`` and aim it at ``target``."""
        return self.set_camera(eye, look_at_rotation(eye, target))

    def focus(self, actor_label: str) -> dict[str, Any]:
        """Frame a named actor, the same way pressing F in the editor does."""
        return self.bridge.call("focus_viewport_on_actor", {"actorLabel": actor_label})

    def orbit(self, target: Vec3, distance: float, yaw_deg: float,
              pitch_deg: float = -20.0) -> dict[str, Any]:
        """Place the camera on an orbit around ``target`` and aim inward."""
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        horizontal = distance * math.cos(pitch)
        eye = (
            target[0] - horizontal * math.cos(yaw),
            target[1] - horizontal * math.sin(yaw),
            target[2] - distance * math.sin(pitch),
        )
        return self.look_at(eye, target)

    # -- capture -----------------------------------------------------------
    def capture(self, output_path: Path | str, width: int = 1280, height: int = 720,
                fov: float = 90.0, location: Vec3 | None = None,
                rotation: dict[str, float] | None = None,
                focus_actor_label: str | None = None,
                focus_direction: Vec3 | None = None,
                focus_margin: float = 1.5,
                timeout: float = 300.0) -> dict[str, Any]:
        """Render an off-screen PNG from the editor world.

        This is a scene-capture render, not a screengrab: it carries no editor
        UI, does not need the window focused, and never touches PIE or the
        player pawn. Prefer it over any screenshot verb for visual proof.
        """
        params: dict[str, Any] = {
            "outputPath": str(Path(output_path)),
            "width": int(width),
            "height": int(height),
            "fov": float(fov),
            "world": "editor",
        }
        if location is not None:
            params["location"] = _vec(location)
        if rotation is not None:
            params["rotation"] = rotation
        if focus_actor_label:
            params["focusActorLabel"] = focus_actor_label
            params["focusMargin"] = float(focus_margin)
            if focus_direction is not None:
                params["focusDirection"] = _vec(focus_direction)
        return self.bridge.call("capture_scene_png", params, timeout=timeout)

    def capture_from_camera(self, camera_label: str, output_path: Path | str,
                            width: int, height: int,
                            timeout: float = 300.0) -> dict[str, Any]:
        """Render exactly what a named CameraActor sees.

        The camera's own transform and FOV are read back from the editor first,
        so the render reproduces its projection instead of approximating it.
        """
        contract = self.camera_contract(camera_label)
        location = tuple(contract["location"])
        rotation = _rot(*contract["rotation_pyr"])
        result = self.capture(output_path, width, height, contract["fov"],
                              location, rotation, timeout=timeout)
        result["camera"] = contract
        return result

    def camera_contract(self, camera_label: str) -> dict[str, Any]:
        """Read a CameraActor's authoritative projection settings.

        Fails when the label does not resolve to exactly one camera, so a
        capture can never silently render from the wrong actor.
        """
        code = f"""
import json, unreal

label = {camera_label!r}
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
matches = [a for a in subsystem.get_all_level_actors()
           if str(a.get_actor_label()) == label]
if len(matches) != 1:
    raise RuntimeError("expected exactly one actor labelled %s, found %d"
                       % (label, len(matches)))
actor = matches[0]
component = actor.get_component_by_class(unreal.CameraComponent)
if component is None:
    raise RuntimeError("actor %s has no CameraComponent" % label)
location = actor.get_actor_location()
rotation = actor.get_actor_rotation()
camera_contract = json.dumps({{
    "label": label,
    "path": str(actor.get_path_name()),
    "location": [location.x, location.y, location.z],
    "rotation_pyr": [rotation.pitch, rotation.yaw, rotation.roll],
    "fov": float(component.get_editor_property("field_of_view")),
    "aspect_ratio": float(component.get_editor_property("aspect_ratio")),
    "constrain_aspect_ratio": bool(component.get_editor_property("constrain_aspect_ratio")),
    "projection_mode": str(component.get_editor_property("projection_mode")),
}})
"""
        return self.bridge.python_json(code, "camera_contract")
