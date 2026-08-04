from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v3 as v3  # noqa: E402


base = v3.v2.base


def iter_action_fcurves(action: bpy.types.Action | None) -> Iterator[bpy.types.FCurve]:
    if action is None:
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        for fcurve in legacy:
            yield fcurve
        return
    for layer in getattr(action, "layers", ()):  # Blender layered Actions.
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                for fcurve in getattr(channelbag, "fcurves", ()):
                    yield fcurve


def set_animation_interpolation(animation_data, interpolation: str, label: str) -> int:
    action = animation_data.action if animation_data else None
    count = 0
    for fcurve in iter_action_fcurves(action):
        for keyframe_point in fcurve.keyframe_points:
            keyframe_point.interpolation = interpolation
            count += 1
    print(
        f"BLENDER_ACTION_INTERPOLATION_COMPAT=PROVEN LABEL={label} "
        f"MODE={interpolation} KEYPOINTS={count}"
    )
    return count


def create_face_mesh_compat(
    vertices_sequence: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    boxes: np.ndarray,
    image_size: tuple[int, int],
    scene_frame_count: int,
) -> tuple[bpy.types.Object, bpy.types.Object, list[int]]:
    del image_size
    keyframe_index = int(len(vertices_sequence) * 0.62)
    keyframe_index = min(max(keyframe_index, 0), len(vertices_sequence) - 1)
    key_box = boxes[keyframe_index]
    center_x = float((key_box[0] + key_box[2]) * 0.5)
    center_y = float((key_box[1] + key_box[3]) * 0.5)
    face_height_pixels = max(float(key_box[3] - key_box[1]), 1.0)
    pixel_scale = 2.30 / face_height_pixels
    base_depth = float(np.median(vertices_sequence[keyframe_index, 2]))

    transformed = np.stack(
        [
            base.transform_vertices(
                frame_vertices,
                center_x,
                center_y,
                pixel_scale,
                base_depth,
            )
            for frame_vertices in vertices_sequence
        ],
        axis=0,
    )

    maximum_shape_keys = 42
    if len(transformed) <= maximum_shape_keys:
        selected_indices = list(range(len(transformed)))
    else:
        selected_indices = sorted(
            set(
                int(round(value))
                for value in np.linspace(0, len(transformed) - 1, maximum_shape_keys)
            )
        )

    base_vertices = transformed[selected_indices[0]]
    mesh = bpy.data.meshes.new("MESH_Antinous_ReconstructedFace")
    mesh.from_pydata(base_vertices.tolist(), [], triangles.tolist())
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

    solidify = face.modifiers.new("FaceShell", "SOLIDIFY")
    solidify.thickness = 0.012
    solidify.offset = -1.0

    face.shape_key_add(name="Basis")
    target_frames: list[int] = []
    key_blocks = []
    for sequence_index in selected_indices:
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
    set_animation_interpolation(
        shape_keys.animation_data if shape_keys else None,
        "LINEAR",
        "FACE_SHAPE_KEYS",
    )

    follow = bpy.data.objects.new("RIG_HeadFollow", None)
    bpy.context.collection.objects.link(follow)
    follow.empty_display_type = "SPHERE"
    follow.empty_display_size = 0.18

    base_box_height = max(float(key_box[3] - key_box[1]), 1.0)
    for sequence_index, target in zip(selected_indices, target_frames):
        box = boxes[sequence_index]
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        scale_ratio = max(
            0.85,
            min(1.15, float(box[3] - box[1]) / base_box_height),
        )
        follow.location = (
            (cx - center_x) * pixel_scale,
            0.0,
            (center_y - cy) * pixel_scale,
        )
        follow.scale = (scale_ratio, scale_ratio, scale_ratio)
        follow.keyframe_insert(data_path="location", frame=target)
        follow.keyframe_insert(data_path="scale", frame=target)

    set_animation_interpolation(
        follow.animation_data,
        "BEZIER",
        "HEAD_FOLLOW",
    )
    print(
        "BLENDER_FACE_MESH_COMPAT=PROVEN "
        f"VERTICES={len(base_vertices)} TRIANGLES={len(triangles)} "
        f"SHAPE_KEYS={len(key_blocks)} TARGETS={target_frames}"
    )
    return face, follow, target_frames


def main() -> int:
    base.create_face_mesh = create_face_mesh_compat
    return int(v3.main())


if __name__ == "__main__":
    raise SystemExit(main())
