"""PATH B motion stage: breathing idle, staff-ready, hand test, cloth sway.

Clips are authored on the source-pose rig. Hidden leg bones are animated to
drive robe response and collision proxies; they carry no skin weight, so the
skirt reacts through the pelvis and cloth chains rather than splitting into
leg-shaped lobes.

Hand motion is Tier 0 (hand/wrist only) and says so: the hands are inside
sleeves and no finger separation is observable, so no finger chains exist to
animate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Euler, Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash  # noqa: E402

FPS = 24
LOOP_TOLERANCE = 1e-4


def radians(value: float) -> float:
    return value * math.pi / 180.0


def key(pose_bone, frame: int, *, rotation=None, location=None) -> None:
    if rotation is not None:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = Euler(rotation, "XYZ").to_quaternion()
        pose_bone.keyframe_insert("rotation_quaternion", frame=frame)
    if location is not None:
        pose_bone.location = location
        pose_bone.keyframe_insert("location", frame=frame)


def new_action(armature, name: str):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    armature.animation_data.action = action
    return action


def reset_pose(armature) -> None:
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
        bone.location = (0.0, 0.0, 0.0)
        bone.scale = (1.0, 1.0, 1.0)


def sine_track(pose_bone, frames: int, axis: int, amplitude_deg: float, cycles: float = 1.0,
               phase: float = 0.0) -> None:
    """Key a closed sine so frame 1 and frame `frames` land on the same value."""

    for step in range(0, frames + 1, 4):
        frame = 1 + step
        t = step / frames
        value = math.sin((t * cycles + phase) * 2.0 * math.pi) * radians(amplitude_deg)
        rotation = [0.0, 0.0, 0.0]
        rotation[axis] = value
        key(pose_bone, frame, rotation=rotation)


def clip_idle_breathe(armature, frames=96):
    new_action(armature, "idle_breathe")
    reset_pose(armature)
    pose = armature.pose.bones
    sine_track(pose["spine_01"], frames, 0, 1.1)
    sine_track(pose["spine_02"], frames, 0, 1.4)
    sine_track(pose["chest"], frames, 0, 1.8)
    sine_track(pose["neck"], frames, 0, -0.9)
    for side in ("l", "r"):
        sine_track(pose[f"clavicle_{side}"], frames, 2, 1.2 if side == "l" else -1.2)
    for tag in ("f", "b", "l", "r"):
        sine_track(pose[f"cloth_{tag}_01"], frames, 0, 0.5, phase=0.12)
        sine_track(pose[f"cloth_{tag}_02"], frames, 0, 0.8, phase=0.24)
    return frames


def clip_staff_ready(armature, frames=48):
    new_action(armature, "staff_ready")
    reset_pose(armature)
    pose = armature.pose.bones
    sine_track(pose["staff_deform"], frames, 0, 2.2)
    sine_track(pose["lowerarm_l"], frames, 0, 3.0)
    sine_track(pose["hand_l"], frames, 0, 2.0)
    sine_track(pose["chest"], frames, 2, 1.5)
    sine_track(pose["spine_02"], frames, 2, 1.0)
    return frames


def clip_hand_curl_test(armature, frames=48):
    new_action(armature, "hand_curl_test")
    reset_pose(armature)
    pose = armature.pose.bones
    # TIER 0: wrist articulation only. No finger chains exist to curl.
    for side in ("l", "r"):
        sine_track(pose[f"hand_{side}"], frames, 0, 12.0)
        sine_track(pose[f"lowerarm_{side}"], frames, 2, 4.0)
    return frames


def clip_secondary_sway(armature, frames=96):
    new_action(armature, "secondary_sway")
    reset_pose(armature)
    pose = armature.pose.bones
    for index, tag in enumerate(("f", "b", "l", "r")):
        phase = index * 0.18
        sine_track(pose[f"cloth_{tag}_01"], frames, 0, 2.0, phase=phase)
        sine_track(pose[f"cloth_{tag}_02"], frames, 0, 3.4, phase=phase + 0.10)
        sine_track(pose[f"cloth_{tag}_03"], frames, 0, 4.6, phase=phase + 0.20)
    sine_track(pose["head"], frames, 2, 1.5)
    return frames


def clip_torso_turn(armature, frames=48):
    new_action(armature, "torso_turn")
    reset_pose(armature)
    pose = armature.pose.bones
    sine_track(pose["spine_01"], frames, 2, 5.0)
    sine_track(pose["spine_02"], frames, 2, 7.0)
    sine_track(pose["chest"], frames, 2, 9.0)
    sine_track(pose["neck"], frames, 2, 4.0)
    return frames


def clip_leg_alternate(armature, frames=96):
    """Hidden leg bones move; the robe answers through pelvis and cloth chains."""

    new_action(armature, "leg_alternate_robe_response")
    reset_pose(armature)
    pose = armature.pose.bones
    sine_track(pose["thigh_l"], frames, 0, 14.0)
    sine_track(pose["thigh_r"], frames, 0, 14.0, phase=0.5)
    sine_track(pose["calf_l"], frames, 0, -10.0, phase=0.08)
    sine_track(pose["calf_r"], frames, 0, -10.0, phase=0.58)
    sine_track(pose["pelvis"], frames, 0, 1.6)
    for tag in ("f", "b", "l", "r"):
        sine_track(pose[f"cloth_{tag}_02"], frames, 0, 2.4, phase=0.15)
        sine_track(pose[f"cloth_{tag}_03"], frames, 0, 3.6, phase=0.30)
    return frames


def clip_wave(armature, frames=72):
    """Right-arm wave.

    Amplitude is deliberately moderate: the bind pose has the arms down against
    the cape, so a large raise would drag cape geometry that shares the sleeve
    influence volume. This is a sleeve-safe wave, not a full shoulder raise.
    """

    new_action(armature, "wave")
    reset_pose(armature)
    pose = armature.pose.bones
    for step in range(0, frames + 1, 3):
        frame = 1 + step
        t = step / frames
        # Raise over the first quarter, hold and oscillate, lower at the end.
        envelope = math.sin(min(t / 0.25, 1.0) * math.pi * 0.5) if t < 0.25 else (
            1.0 if t < 0.75 else math.cos((t - 0.75) / 0.25 * math.pi * 0.5)
        )
        wave = math.sin(t * 6.0 * math.pi) * envelope
        key(pose["clavicle_r"], frame, rotation=[0.0, 0.0, radians(-10.0 * envelope)])
        key(pose["upperarm_r"], frame, rotation=[radians(-6.0 * envelope), 0.0,
                                                 radians(-26.0 * envelope)])
        key(pose["lowerarm_r"], frame, rotation=[radians(-18.0 * envelope), 0.0,
                                                 radians(-20.0 * wave)])
        key(pose["hand_r"], frame, rotation=[radians(10.0 * wave), 0.0,
                                             radians(-34.0 * wave)])
        key(pose["chest"], frame, rotation=[0.0, 0.0, radians(3.0 * envelope)])
        key(pose["head"], frame, rotation=[0.0, 0.0, radians(4.0 * envelope)])
    return frames


def clip_walk_cycle(armature, frames=48):
    """In-place walk. Hidden legs drive the robe; the mesh answers via cloth."""

    new_action(armature, "walk_cycle")
    reset_pose(armature)
    pose = armature.pose.bones
    for step in range(0, frames + 1, 2):
        frame = 1 + step
        t = step / frames
        swing = math.sin(t * 2.0 * math.pi)
        opposite = math.sin(t * 2.0 * math.pi + math.pi)
        bob = abs(math.sin(t * 2.0 * math.pi)) * 0.018

        key(pose["thigh_l"], frame, rotation=[radians(24.0 * swing), 0.0, 0.0])
        key(pose["thigh_r"], frame, rotation=[radians(24.0 * opposite), 0.0, 0.0])
        key(pose["calf_l"], frame, rotation=[radians(-22.0 * max(-swing, 0.0) - 6.0), 0.0, 0.0])
        key(pose["calf_r"], frame, rotation=[radians(-22.0 * max(-opposite, 0.0) - 6.0), 0.0, 0.0])
        key(pose["foot_l"], frame, rotation=[radians(8.0 * swing), 0.0, 0.0])
        key(pose["foot_r"], frame, rotation=[radians(8.0 * opposite), 0.0, 0.0])

        # Pelvis translates only. Rotating it bends the entire character: the
        # pelvis parents the whole spine chain AND carries the largest share of
        # robe weight, so even 2-3 degrees leans the full silhouette.
        key(pose["pelvis"], frame, rotation=[0.0, 0.0, 0.0], location=(0.0, 0.0, bob))
        key(pose["spine_02"], frame, rotation=[0.0, 0.0, radians(-1.2 * swing)])
        key(pose["chest"], frame, rotation=[0.0, 0.0, radians(-2.0 * swing)])
        key(pose["clavicle_l"], frame, rotation=[radians(-9.0 * opposite), 0.0, 0.0])
        key(pose["clavicle_r"], frame, rotation=[radians(-9.0 * swing), 0.0, 0.0])
        key(pose["upperarm_l"], frame, rotation=[radians(-12.0 * opposite), 0.0, 0.0])
        key(pose["upperarm_r"], frame, rotation=[radians(-12.0 * swing), 0.0, 0.0])

        # Alternate the side chains against each other so the hem reads as a
        # stride rather than the whole skirt swinging as one rigid cone.
        for tag, phase in (("f", swing), ("b", swing), ("l", swing), ("r", opposite)):
            key(pose[f"cloth_{tag}_01"], frame, rotation=[radians(2.0 * phase), 0.0, 0.0])
            key(pose[f"cloth_{tag}_02"], frame, rotation=[radians(3.5 * phase), 0.0, 0.0])
            key(pose[f"cloth_{tag}_03"], frame, rotation=[radians(5.0 * phase), 0.0, 0.0])
    return frames


CLIPS = (
    ("idle_breathe", clip_idle_breathe, True),
    ("wave", clip_wave, False),
    ("walk_cycle", clip_walk_cycle, True),
    ("staff_ready", clip_staff_ready, True),
    ("hand_curl_test", clip_hand_curl_test, True),
    ("secondary_sway", clip_secondary_sway, True),
    ("torso_turn", clip_torso_turn, True),
    ("leg_alternate_robe_response", clip_leg_alternate, True),
)


def sample_matrices(armature, frame: int, bones: list[str]) -> dict:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    return {
        name: np.array(armature.pose.bones[name].matrix.copy()) for name in bones
    }


def validate(armature, action_name: str, frames: int, loop: bool, tracked: list[str]) -> dict:
    armature.animation_data.action = bpy.data.actions[action_name]
    first = sample_matrices(armature, 1, tracked)
    last = sample_matrices(armature, 1 + frames, tracked)
    middle = sample_matrices(armature, 1 + frames // 2, tracked)

    loop_delta = max(
        float(np.abs(first[name] - last[name]).max()) for name in tracked
    )
    finite = all(
        np.isfinite(matrix).all()
        for sample in (first, middle, last)
        for matrix in sample.values()
    )
    root_drift = max(
        float(np.linalg.norm(sample["root"][:3, 3] - first["root"][:3, 3]))
        for sample in (middle, last)
    )
    foot_drift = max(
        float(np.linalg.norm(sample[name][:3, 3] - first[name][:3, 3]))
        for sample in (middle, last)
        for name in ("foot_l", "foot_r")
    )
    scale_signs = []
    for sample in (first, middle, last):
        for matrix in sample.values():
            scale_signs.append(float(np.linalg.det(matrix[:3, :3])))

    failures = []
    if not finite:
        failures.append("NONFINITE_TRANSFORMS")
    if loop and loop_delta > LOOP_TOLERANCE:
        failures.append("LOOP_NOT_CLOSED")
    if min(scale_signs) <= 0.0:
        failures.append("SCALE_SIGN_FLIP")

    return {
        "action": action_name,
        "frames": frames,
        "fps": FPS,
        "duration_seconds": frames / FPS,
        "loop_required": loop,
        "loop_delta": loop_delta,
        "loop_tolerance": LOOP_TOLERANCE,
        "root_drift": root_drift,
        "foot_drift": foot_drift,
        "finite_transforms": finite,
        "min_scale_determinant": min(scale_signs),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qa-report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    bpy.ops.wm.open_mainfile(filepath=args.input)
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    bpy.context.view_layer.objects.active = armature
    if armature.animation_data is None:
        armature.animation_data_create()

    scene = bpy.context.scene
    scene.render.fps = FPS

    tracked = ["root", "pelvis", "chest", "head", "foot_l", "foot_r", "hand_l", "staff_deform"]
    clips = []
    for name, builder, loop in CLIPS:
        frames = builder(armature)
        clips.append({"name": name, "frames": frames, "loop": loop})

    reports = [validate(armature, item["name"], item["frames"], item["loop"], tracked) for item in clips]
    failures = sorted({code for report in reports for code in report["failures"]})

    height = 1.98
    for report in reports:
        if report["action"] == "idle_breathe":
            if report["root_drift"] > height * 0.001:
                failures.append("IDLE_ROOT_DRIFT")
            if report["foot_drift"] > height * 0.001:
                failures.append("IDLE_FOOT_DRIFT")
    failures = sorted(set(failures))

    manifest = {
        "stage": "MOTION",
        "fps": FPS,
        "bind_pose": "SOURCE_POSE",
        "finger_tier": 0,
        "finger_tier_note": (
            "hand_curl_test articulates wrist only. Hands are inside sleeves; "
            "no finger separation is observable, so no finger chains exist."
        ),
        "clip_count": len(clips),
        "clips": [
            {**clip, "duration_seconds": clip["frames"] / FPS} for clip in clips
        ],
        "hidden_leg_policy": (
            "Leg bones animate to drive robe response and collision proxies; "
            "they carry zero skin weight so the skirt cannot split into legs."
        ),
    }
    qa = {
        "stage": "MOTION_QA",
        "passed": not failures,
        "failures": failures,
        "clip_reports": reports,
    }

    for path, payload in ((args.manifest, manifest), (args.qa_report, qa)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures:
        print("MOTION_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)

    armature.animation_data.action = bpy.data.actions["idle_breathe"]
    scene.frame_start = 1
    scene.frame_end = 96
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)

    for report in reports:
        print(
            f"CLIP {report['action']:30s} frames={report['frames']:3d} "
            f"loop_delta={report['loop_delta']:.2e} root_drift={report['root_drift']:.2e} "
            f"foot_drift={report['foot_drift']:.2e}",
            flush=True,
        )
    print(f"MOTION_CLIPS={len(clips)}", flush=True)
    print("MOTION_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
