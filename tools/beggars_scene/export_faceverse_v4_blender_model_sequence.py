from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


DEFAULT_ANCHORS = "21,31,35"
TARGET_HEAD_HEIGHT = 2.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export corrected FaceVerse coefficients as true model-space Blender geometry. "
            "Projected image coordinates are used only for source-color sampling."
        )
    )
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) < 2:
        raise ValueError("At least two Blender animation anchors are required")
    return indices


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bilinear_sample_rgb(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
    )
    sampled = np.zeros((len(points_xy), 3), dtype=np.float32)
    if not np.any(valid):
        return sampled, valid

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (xv - x0).astype(np.float32)[:, None]
    wy = (yv - y0).astype(np.float32)[:, None]
    top = (
        image_rgb[y0, x0].astype(np.float32) * (1.0 - wx)
        + image_rgb[y0, x1].astype(np.float32) * wx
    )
    bottom = (
        image_rgb[y1, x0].astype(np.float32) * (1.0 - wx)
        + image_rgb[y1, x1].astype(np.float32) * wx
    )
    sampled[valid] = (top * (1.0 - wy) + bottom * wy) / 255.0
    return sampled, valid


def decode_frames(
    clip_path: Path,
    frame_indices: list[int],
) -> tuple[dict[int, np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid clip metadata: fps={fps} width={width} height={height}"
        )
    frames: dict[int, np.ndarray] = {}
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode public frame {frame_index}")
        frames[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    capture.release()
    return frames, fps, (width, height)


def apply_faceverse_pose(
    model: Any,
    vertices_unposed: np.ndarray,
    coefficients: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    vertices_tensor = torch.from_numpy(vertices_unposed.astype(np.float32)).to(device)
    _, _, _, _, angles, translation, _ = model.split_coeffs(coefficients)
    rotation = model.compute_rotation_matrix(angles)
    posed = model.rigid_transform(vertices_tensor, rotation, translation)
    return posed.detach().cpu().numpy().astype(np.float32)


def map_faceverse_to_blender(vertices_faceverse: np.ndarray) -> np.ndarray:
    # FaceVerse: X horizontal, Y vertical/down, Z depth/front-negative.
    # Blender: X horizontal, Y depth/front-negative, Z vertical/up.
    mapped = np.empty_like(vertices_faceverse, dtype=np.float32)
    mapped[:, :, 0] = vertices_faceverse[:, :, 0]
    mapped[:, :, 1] = vertices_faceverse[:, :, 2]
    mapped[:, :, 2] = -vertices_faceverse[:, :, 1]
    return mapped


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    coefficient_path = Path(args.coefficients).resolve()
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    clip_path = Path(args.clip).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_report = Path(args.output_report).resolve()
    anchors = parse_indices(args.anchors)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    for required in (
        coefficient_path,
        faceverse_root,
        model_path,
        checkpoint_path,
        clip_path,
    ):
        if not required.exists():
            raise SystemExit(f"Model-space Blender input is missing: {required}")

    fit = np.load(coefficient_path)
    frame_indices = np.asarray(fit["frame_indices"], dtype=np.int32).tolist()
    optimized = np.asarray(fit["optimized_coefficients"], dtype=np.float32)
    bbox_matrix = np.asarray(fit["bbox_matrix"], dtype=np.float32)
    if optimized.shape[0] != len(frame_indices) or bbox_matrix.shape[0] != len(frame_indices):
        raise RuntimeError("Corrected fit arrays do not share the frame-index dimension")
    lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
    missing = [frame for frame in anchors if frame not in lookup]
    if missing:
        raise RuntimeError(f"Requested model-space anchors are absent: {missing}")
    rows = [lookup[frame] for frame in anchors]
    selected_coefficients = optimized[rows]
    selected_boxes = bbox_matrix[rows]

    frames_rgb, source_fps, image_size = decode_frames(clip_path, anchors)
    if str(faceverse_root) not in sys.path:
        sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    print(f"FACEVERSE_MODEL_SPACE_DEVICE={device}", flush=True)
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    coefficient_tensor = torch.from_numpy(selected_coefficients).to(device)

    vertices_unposed, projected, _normals, pca_colors = model.from_coeffs(
        coefficient_tensor,
        selected_boxes.astype(np.int32),
    )
    vertices_posed = apply_faceverse_pose(
        model,
        np.asarray(vertices_unposed, dtype=np.float32),
        coefficient_tensor,
        device,
    )
    mapped = map_faceverse_to_blender(vertices_posed)

    keyframe_index = anchors.index(31) if 31 in anchors else len(anchors) // 2
    key_vertices = mapped[keyframe_index]
    key_min = np.min(key_vertices, axis=0)
    key_max = np.max(key_vertices, axis=0)
    center = (key_min + key_max) * 0.5
    vertical_height = float(key_max[2] - key_min[2])
    if vertical_height <= 1e-6:
        raise RuntimeError(f"Model-space head has invalid vertical height: {vertical_height}")
    world_scale = TARGET_HEAD_HEIGHT / vertical_height
    world_vertices = (mapped - center[None, None, :]) * world_scale

    keyframe_rgb = frames_rgb[anchors[keyframe_index]]
    sampled_colors, valid = bilinear_sample_rgb(
        keyframe_rgb,
        np.asarray(projected[keyframe_index, :, :2], dtype=np.float32),
    )
    native_colors = np.asarray(pca_colors[keyframe_index], dtype=np.float32)
    colors_rgb = np.clip(native_colors, 0.0, 1.0)
    colors_rgb[valid] = np.clip(
        native_colors[valid] * 0.12 + sampled_colors[valid] * 0.88,
        0.0,
        1.0,
    )

    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)
    vertices_sequence = np.transpose(world_vertices, (0, 2, 1)).astype(np.float32)
    key_world = world_vertices[keyframe_index]
    key_world_min = np.min(key_world, axis=0)
    key_world_max = np.max(key_world, axis=0)

    np.savez_compressed(
        output_npz,
        vertices_raw=vertices_sequence,
        vertices_smoothed=vertices_sequence,
        triangles=triangles,
        colors_rgb=colors_rgb.astype(np.float32),
        parameters=selected_coefficients.astype(np.float32),
        boxes=selected_boxes.astype(np.float32),
        source_frame_indices=np.asarray(anchors, dtype=np.int32),
        source_fps=np.asarray([source_fps], dtype=np.float32),
        sampled_fps=np.asarray([source_fps], dtype=np.float32),
        image_size=np.asarray(image_size, dtype=np.int32),
        keyframe_index=np.asarray([keyframe_index], dtype=np.int32),
        model_space_center=center.astype(np.float32),
        model_space_scale=np.asarray([world_scale], dtype=np.float32),
        coordinate_mapping=np.asarray([0, 2, 1], dtype=np.int32),
    )

    report: dict[str, Any] = {
        "classification": "PROVEN",
        "route": "FACEVERSE_V4_SHARED_IDENTITY_V2_TRUE_MODEL_SPACE_TO_BLENDER",
        "source_fit": str(coefficient_path),
        "anchors": anchors,
        "keyframe_index": keyframe_index,
        "keyframe_source_frame": anchors[keyframe_index],
        "frame_count": len(anchors),
        "vertex_count": int(vertices_sequence.shape[2]),
        "triangle_count": int(triangles.shape[0]),
        "image_size": list(image_size),
        "source_fps": source_fps,
        "source_sampled_vertex_fraction": float(np.mean(valid)),
        "faceverse_to_blender_mapping": {
            "blender_x": "faceverse_x",
            "blender_y": "faceverse_z",
            "blender_z": "-faceverse_y",
        },
        "model_space_center_faceverse_mapped": center.astype(float).tolist(),
        "model_space_scale": world_scale,
        "target_head_height_world": TARGET_HEAD_HEIGHT,
        "keyframe_world_bounds": {
            "min": key_world_min.astype(float).tolist(),
            "max": key_world_max.astype(float).tolist(),
        },
        "output_npz": str(output_npz),
        "projected_coordinates_used_for_geometry": False,
        "projected_coordinates_used_for_source_color_sampling": True,
        "reference_media_packaged": False,
        "source_frame_plane_used": False,
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
