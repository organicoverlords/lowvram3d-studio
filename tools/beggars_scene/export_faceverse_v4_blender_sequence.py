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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the corrected FaceVerse shared-identity fit as an animated Blender sequence."
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


def bilinear_sample_rgb(image_rgb: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
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
    top = image_rgb[y0, x0].astype(np.float32) * (1.0 - wx) + image_rgb[y0, x1].astype(np.float32) * wx
    bottom = image_rgb[y1, x0].astype(np.float32) * (1.0 - wx) + image_rgb[y1, x1].astype(np.float32) * wx
    sampled[valid] = (top * (1.0 - wy) + bottom * wy) / 255.0
    return sampled, valid


def decode_frames(clip_path: Path, frame_indices: list[int]) -> tuple[dict[int, np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid clip metadata: fps={fps} width={width} height={height}")
    frames: dict[int, np.ndarray] = {}
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode public frame {frame_index}")
        frames[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    capture.release()
    return frames, fps, (width, height)


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

    for required in (coefficient_path, faceverse_root, model_path, checkpoint_path, clip_path):
        if not required.exists():
            raise SystemExit(f"Blender-sequence input is missing: {required}")

    fit = np.load(coefficient_path)
    frame_indices = np.asarray(fit["frame_indices"], dtype=np.int32).tolist()
    optimized = np.asarray(fit["optimized_coefficients"], dtype=np.float32)
    bbox_matrix = np.asarray(fit["bbox_matrix"], dtype=np.float32)
    if optimized.shape[0] != len(frame_indices) or bbox_matrix.shape[0] != len(frame_indices):
        raise RuntimeError("Corrected fit arrays do not share the frame-index dimension")
    lookup = {int(frame): index for index, frame in enumerate(frame_indices)}
    missing = [frame for frame in anchors if frame not in lookup]
    if missing:
        raise RuntimeError(f"Requested Blender anchors are absent from corrected fit: {missing}")
    rows = [lookup[frame] for frame in anchors]
    selected_coefficients = optimized[rows]
    selected_boxes = bbox_matrix[rows]

    frames_rgb, source_fps, image_size = decode_frames(clip_path, anchors)
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    coefficient_tensor = torch.from_numpy(selected_coefficients).to(device)
    vertices_model, projected, _normals, pca_colors = model.from_coeffs(
        coefficient_tensor,
        selected_boxes.astype(np.int32),
    )
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)

    keyframe_index = anchors.index(31) if 31 in anchors else len(anchors) // 2
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

    key_box_height = max(float(selected_boxes[keyframe_index, 3] - selected_boxes[keyframe_index, 1]), 1.0)
    depth_multiplier = key_box_height * 0.42
    base_depth = float(np.median(vertices_model[keyframe_index, :, 2]))
    vertices_image = np.empty((len(anchors), 3, vertices_model.shape[1]), dtype=np.float32)
    for index in range(len(anchors)):
        vertices_image[index, 0] = projected[index, :, 0]
        vertices_image[index, 1] = projected[index, :, 1]
        vertices_image[index, 2] = (
            np.asarray(vertices_model[index, :, 2], dtype=np.float32) - base_depth
        ) * depth_multiplier

    np.savez_compressed(
        output_npz,
        vertices_raw=vertices_image,
        vertices_smoothed=vertices_image,
        triangles=triangles,
        colors_rgb=colors_rgb.astype(np.float32),
        parameters=selected_coefficients.astype(np.float32),
        boxes=selected_boxes.astype(np.float32),
        source_frame_indices=np.asarray(anchors, dtype=np.int32),
        source_fps=np.asarray([source_fps], dtype=np.float32),
        sampled_fps=np.asarray([source_fps], dtype=np.float32),
        image_size=np.asarray(image_size, dtype=np.int32),
        keyframe_index=np.asarray([keyframe_index], dtype=np.int32),
        depth_multiplier=np.asarray([depth_multiplier], dtype=np.float32),
    )

    report: dict[str, Any] = {
        "classification": "PROVEN",
        "route": "FACEVERSE_V4_SHARED_IDENTITY_V2_TO_BLENDER_SEQUENCE",
        "source_fit": str(coefficient_path),
        "anchors": anchors,
        "keyframe_index": keyframe_index,
        "keyframe_source_frame": anchors[keyframe_index],
        "frame_count": len(anchors),
        "vertex_count": int(vertices_image.shape[2]),
        "triangle_count": int(triangles.shape[0]),
        "image_size": list(image_size),
        "source_fps": source_fps,
        "source_sampled_vertex_fraction": float(np.mean(valid)),
        "depth_multiplier": depth_multiplier,
        "output_npz": str(output_npz),
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
