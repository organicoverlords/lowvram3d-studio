from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v4 as v4  # noqa: E402


base = v4.base


def create_model_space_face_mesh(
    vertices_sequence: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    boxes: np.ndarray,
    image_size: tuple[int, int],
    scene_frame_count: int,
) -> tuple[bpy.types.Object, bpy.types.Object, list[int]]:
    del image_size
    if vertices_sequence.ndim != 3 or vertices_sequence.shape[1] != 3:
        raise RuntimeError(
            f"Expected model-space vertices shaped (frames,3,vertices), got {vertices_sequence.shape}"
        )
    transformed = np.transpose(vertices_sequence, (0, 2, 1)).astype(np.float32)
    if not np.all(np.isfinite(transformed)):
        raise RuntimeError("Model-space FaceVerse sequence contains non-finite vertices")

    keyframe_index = min(1, len(transformed) - 1)
    key_box = boxes[keyframe_index]
    key_min = np.min(transformed[keyframe_index], axis=0)
    key_max = np.max(transformed[keyframe_index], axis=0)
    head_height = float(key_max[2] - key_min[2])
    head_width = float(key_max[0] - key_min[0])
    head_depth = float(key_max[1] - key_min[1])
    if not 2.15 <= head_height <= 2.45:
        raise RuntimeError(f"Model-space head height is implausible: {head_height:.4f}")
    if not 1.5 <= head_width <= 2.4:
        raise RuntimeError(f"Model-space head width is implausible: {head_width:.4f}")
    if not 1.2 <= head_depth <= 2.5:
        raise RuntimeError(f"Model-space head depth is implausible: {head_depth:.4f}")

    mesh = bpy.data.meshes.new("MESH_Antinous_TrueModelSpaceFace")
    mesh.from_pydata(transformed[0].tolist(), [], triangles.tolist())
    mesh.update()
    face = bpy.data.objects.new("CHAR_Antinous", mesh)
    bpy.context.collection.objects.link(face)
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    color_attribute = mesh.color_attributes.new(
        name="Col",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    rgba = np.ones((colors_rgb.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors_rgb, 0.0, 1.0)
    color_attribute.data.foreach_set("color", rgba.reshape(-1).tolist())
    mesh.materials.append(base.vertex_skin_material())

    face.shape_key_add(name="Basis")
    target_frames: list[int] = []
    key_blocks = []
    for sequence_index in range(len(transformed)):
        key = face.shape_key_add(name=f"Track_{sequence_index:04d}")
        key.data.foreach_set("co", transformed[sequence_index].reshape(-1).tolist())
        key.value = 0.0
        target = 1 + int(
            round(
                sequence_index
                / max(len(transformed) - 1, 1)
                * (scene_frame_count - 1)
            )
        )
        target_frames.append(target)
        key_blocks.append(key)

    for index, (key, target) in enumerate(zip(key_blocks, target_frames)):
        previous_target = target_frames[index - 1] if index > 0 else 1
        next_target = (
            target_frames[index + 1]
            if index + 1 < len(target_frames)
            else scene_frame_count
        )
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=previous_target)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=target)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=next_target)

    shape_keys = face.data.shape_keys
    v4.set_animation_interpolation(
        shape_keys.animation_data if shape_keys else None,
        "LINEAR",
        "MODEL_SPACE_FACE_SHAPE_KEYS",
    )

    follow = bpy.data.objects.new("RIG_HeadFollow", None)
    bpy.context.collection.objects.link(follow)
    follow.empty_display_type = "SPHERE"
    follow.empty_display_size = 0.18
    face.parent = follow

    center_x = float((key_box[0] + key_box[2]) * 0.5)
    center_y = float((key_box[1] + key_box[3]) * 0.5)
    base_box_height = max(float(key_box[3] - key_box[1]), 1.0)
    pixel_scale = 2.30 / base_box_height
    for sequence_index, target in enumerate(target_frames):
        box = boxes[sequence_index]
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        scale_ratio = max(
            0.90,
            min(1.10, float(box[3] - box[1]) / base_box_height),
        )
        follow.location = (
            (cx - center_x) * pixel_scale,
            0.0,
            (center_y - cy) * pixel_scale,
        )
        follow.scale = (scale_ratio, scale_ratio, scale_ratio)
        follow.keyframe_insert(data_path="location", frame=target)
        follow.keyframe_insert(data_path="scale", frame=target)

    v4.set_animation_interpolation(
        follow.animation_data,
        "BEZIER",
        "MODEL_SPACE_HEAD_FOLLOW",
    )
    print(
        "BLENDER_TRUE_MODEL_SPACE_FACE=PROVEN "
        f"VERTICES={len(transformed[0])} TRIANGLES={len(triangles)} "
        f"BOUNDS_MIN={key_min.tolist()} BOUNDS_MAX={key_max.tolist()} "
        f"TARGETS={target_frames} SOLIDIFY=ABSENT"
    )
    return face, follow, target_frames


def build_model_space_character(
    follow: bpy.types.Object,
    colors_rgb: np.ndarray,
) -> dict[str, object]:
    black_cloth = base.principled_material(
        "MAT_Black_AncientCloth_V5",
        (0.006, 0.004, 0.006, 1.0),
        0.76,
    )
    dark_hair = base.principled_material(
        "MAT_Antinous_DarkWavyHair_V5",
        (0.008, 0.003, 0.002, 1.0),
        0.31,
    )
    gold = base.principled_material(
        "MAT_Aged_Gold_V5",
        (0.32, 0.105, 0.020, 1.0),
        0.30,
        metallic=0.70,
    )
    skin_average = np.clip(np.median(colors_rgb, axis=0), 0.05, 0.95)
    neck_skin = base.principled_material(
        "MAT_Neck_Skin_V5",
        (
            float(skin_average[0]),
            float(skin_average[1]),
            float(skin_average[2]),
            1.0,
        ),
        0.54,
    )

    neck = base.create_uv_sphere(
        "CHAR_Antinous_Neck",
        (0.0, 0.34, -1.20),
        (0.48, 0.40, 0.66),
        neck_skin,
        segments=48,
        ring_count=24,
    )
    neck.parent = follow
    torso = base.create_uv_sphere(
        "CHAR_Antinous_Torso",
        (0.04, 0.52, -2.05),
        (1.76, 0.70, 1.08),
        black_cloth,
        segments=64,
        ring_count=32,
    )
    torso.parent = follow
    shoulder_left = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_L",
        (-1.50, 0.50, -1.66),
        (0.84, 0.58, 0.68),
        black_cloth,
    )
    shoulder_left.parent = follow
    shoulder_right = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_R",
        (1.50, 0.50, -1.66),
        (0.84, 0.58, 0.68),
        black_cloth,
    )
    shoulder_right.parent = follow
    trim = base.create_curve(
        "COSTUME_GoldNeckTrim",
        [
            (-0.72, -0.02, -1.43),
            (-0.38, -0.12, -1.61),
            (0.00, -0.15, -1.67),
            (0.38, -0.12, -1.61),
            (0.72, -0.02, -1.43),
        ],
        gold,
        0.027,
        parent=follow,
    )

    # Rear scalp stays behind the true facial surface.
    hair_cap = base.create_uv_sphere(
        "CHAR_Antinous_HairCap",
        (0.0, 0.36, 0.40),
        (0.90, 0.39, 0.98),
        dark_hair,
        segments=64,
        ring_count=32,
    )
    hair_cap.parent = follow

    wave_specs = [
        ((-0.72, -0.20, 0.70), (0.34, 0.20, 0.29), -0.14),
        ((-0.54, -0.18, 0.92), (0.36, 0.20, 0.30), 0.12),
        ((-0.31, -0.17, 1.08), (0.38, 0.20, 0.29), -0.13),
        ((-0.04, -0.16, 1.16), (0.40, 0.20, 0.30), 0.07),
        ((0.24, -0.17, 1.13), (0.38, 0.20, 0.29), -0.08),
        ((0.50, -0.18, 0.99), (0.36, 0.20, 0.30), 0.14),
        ((0.70, -0.20, 0.78), (0.34, 0.20, 0.30), -0.11),
    ]
    wave_masses: list[bpy.types.Object] = []
    for index, (location, scale, roll) in enumerate(wave_specs):
        mass = base.create_uv_sphere(
            f"HAIR_WaveMass_{index:02d}",
            location,
            scale,
            dark_hair,
            segments=32,
            ring_count=16,
        )
        mass.rotation_euler[1] = roll
        mass.parent = follow
        wave_masses.append(mass)

    strands: list[bpy.types.Object] = []
    for index, x in enumerate(np.linspace(-0.76, 0.74, 26)):
        phase = index * 0.69
        points = [
            (
                float(x),
                -0.56 + 0.018 * math.sin(phase),
                1.08 + 0.11 * math.cos(phase * 0.75),
            ),
            (
                float(x + 0.07 * math.sin(phase)),
                -0.34,
                0.96 + 0.08 * math.cos(phase),
            ),
            (
                float(x + 0.11 * math.sin(phase + 0.8)),
                -0.08,
                0.78 + 0.07 * math.cos(phase * 1.2),
            ),
        ]
        strands.append(
            base.create_curve(
                f"HAIR_Strand_{index:03d}",
                points,
                dark_hair,
                0.010 + 0.002 * ((index % 3) / 2.0),
                parent=follow,
            )
        )

    strand_index = len(strands)
    for side in (-1.0, 1.0):
        for row in range(10):
            z_top = 0.82 - row * 0.10
            x = side * (0.78 + 0.025 * math.sin(row * 0.8))
            points = [
                (x, -0.44, z_top),
                (x + side * 0.06, -0.30, z_top - 0.10),
                (x - side * 0.02, -0.12, z_top - 0.22),
            ]
            strands.append(
                base.create_curve(
                    f"HAIR_Strand_{strand_index:03d}",
                    points,
                    dark_hair,
                    0.011,
                    parent=follow,
                )
            )
            strand_index += 1

    eyebrow_left = base.create_curve(
        "FACIALHAIR_Eyebrow_L",
        [(-0.58, -0.84, 0.30), (-0.36, -0.88, 0.37), (-0.12, -0.87, 0.31)],
        dark_hair,
        0.025,
        parent=follow,
    )
    eyebrow_right = base.create_curve(
        "FACIALHAIR_Eyebrow_R",
        [(0.12, -0.87, 0.31), (0.36, -0.88, 0.37), (0.58, -0.84, 0.30)],
        dark_hair,
        0.025,
        parent=follow,
    )
    moustache_left = base.create_curve(
        "FACIALHAIR_Moustache_L",
        [(-0.50, -1.04, -0.28), (-0.30, -1.08, -0.24), (-0.03, -1.09, -0.27)],
        dark_hair,
        0.027,
        parent=follow,
    )
    moustache_right = base.create_curve(
        "FACIALHAIR_Moustache_R",
        [(0.03, -1.09, -0.27), (0.30, -1.08, -0.24), (0.50, -1.04, -0.28)],
        dark_hair,
        0.027,
        parent=follow,
    )
    sideburn_left = base.create_curve(
        "FACIALHAIR_Sideburn_L",
        [(-0.78, -0.52, 0.38), (-0.80, -0.53, 0.10), (-0.74, -0.55, -0.18)],
        dark_hair,
        0.031,
        parent=follow,
    )
    sideburn_right = base.create_curve(
        "FACIALHAIR_Sideburn_R",
        [(0.78, -0.52, 0.38), (0.80, -0.53, 0.10), (0.74, -0.55, -0.18)],
        dark_hair,
        0.031,
        parent=follow,
    )

    stubble_count = 0
    for index, x in enumerate(np.linspace(-0.42, 0.42, 11)):
        z = -0.64 - 0.14 * (1.0 - abs(float(x)) / 0.42)
        base.create_curve(
            f"FACIALHAIR_Chin_{index:02d}",
            [(float(x), -0.94, z + 0.05), (float(x) * 0.95, -0.96, z - 0.07)],
            dark_hair,
            0.009,
            parent=follow,
        )
        stubble_count += 1

    print(
        "BLENDER_MODEL_SPACE_GROOM=PROVEN "
        f"STRANDS={len(strands)} WAVES={len(wave_masses)} "
        f"FACIAL_HAIR={6 + stubble_count}"
    )
    return {
        "hair_cap": hair_cap,
        "neck": neck,
        "torso": torso,
        "trim": trim,
        "strand_count": len(strands),
        "wave_mass_count": len(wave_masses),
        "facial_hair_count": 6 + stubble_count,
        "eyebrow_left": eyebrow_left,
        "eyebrow_right": eyebrow_right,
        "moustache_left": moustache_left,
        "moustache_right": moustache_right,
        "sideburn_left": sideburn_left,
        "sideburn_right": sideburn_right,
        "variant": "FACEVERSE_SHARED_IDENTITY_V2_TRUE_MODEL_SPACE_GROOMED",
    }


def main() -> int:
    v4.create_face_mesh_compat = create_model_space_face_mesh
    v4.v3.v2.build_character_v2 = build_model_space_character
    return int(v4.main())


if __name__ == "__main__":
    raise SystemExit(main())
