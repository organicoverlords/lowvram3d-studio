"""Correct the reconstructed shell's 180-degree roll about the camera view axis.

Scoring the render against the source under all four axis-flip candidates
(scripts/compare_source_render.py) put `rot180` far ahead -- correlation 0.50
versus negative correlation for identity, hflip and vflip. A mirror would have
favoured hflip or vflip, so this is a handedness-preserving rotation introduced
by the GLB-to-Unreal conversion, not an axis mirror.

The camera contract is authoritative and is not touched. The shell is rotated
180 degrees about the camera's forward axis instead. The camera looks down +Y
(yaw 90), and rotation about Y is Pitch in Unreal's rotator, so Pitch=180 is
the roll about the view direction.

    python -m uemcp python @unreal/fix_shell_roll.py --json
"""

import json

import unreal

SHELL_LABEL = "Castlegrounds_ReconstructedMesh"
CAMERA_LABEL = "Castlegrounds_Camera_Source"

subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = list(subsystem.get_all_level_actors())


def one(label: str):
    matches = [a for a in actors if str(a.get_actor_label()) == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


shell = one(SHELL_LABEL)
camera = one(CAMERA_LABEL)

before_rot = shell.get_actor_rotation()
before_origin, before_extent = shell.get_actor_bounds(False)

shell.set_actor_rotation(unreal.Rotator(roll=0.0, pitch=180.0, yaw=0.0), False)

after_rot = shell.get_actor_rotation()
after_origin, after_extent = shell.get_actor_bounds(False)

camera_rot = camera.get_actor_rotation()

result = json.dumps({
    "shell": str(shell.get_path_name()),
    "camera_rotation_pyr": [float(camera_rot.pitch), float(camera_rot.yaw),
                            float(camera_rot.roll)],
    "camera_untouched": True,
    "rotation_before": [float(before_rot.pitch), float(before_rot.yaw), float(before_rot.roll)],
    "rotation_after": [float(after_rot.pitch), float(after_rot.yaw), float(after_rot.roll)],
    "bounds_origin_before": [float(before_origin.x), float(before_origin.y), float(before_origin.z)],
    "bounds_origin_after": [float(after_origin.x), float(after_origin.y), float(after_origin.z)],
    "bounds_extent_after": [float(after_extent.x), float(after_extent.y), float(after_extent.z)],
})
