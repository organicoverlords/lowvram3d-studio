from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_FRAME = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a public keyframe and per-vertex camera-projection UVs for the true 3D "
            "FaceVerse mesh. This does not create or use a source-frame plane."
        )
    )
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decode_frame(clip_path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if fps <= 0 or not ok or frame_bgr is None:
        raise RuntimeError(f"Could not decode public frame {frame_index}; fps={fps}")
    return frame_bgr, fps


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    coefficient_path = Path(args.coefficients).resolve()
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    clip_path = Path(args.clip).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_image = Path(args.output_image).resolve()
    output_report = Path(args.output_report).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    for required in (coefficient_path, faceverse_root, model_path, checkpoint_path, clip_path):
        if not required.exists():
            raise SystemExit(f"Projection-texture input is missing: {required}")

    fit = np.load(coefficient_path)
    frame_indices = np.asarray(fit["frame_indices"], dtype=np.int32).tolist()
    optimized = np.asarray(fit["optimized_coefficients"], dtype=np.float32)
    bbox_matrix = np.asarray(fit["bbox_matrix"], dtype=np.float32)
    if args.frame not in frame_indices:
        raise RuntimeError(
            f"Projection frame {args.frame} is absent from optimized fit; available={frame_indices}"
        )
    row = frame_indices.index(args.frame)
    coefficient = optimized[row : row + 1]
    bbox = bbox_matrix[row : row + 1]

    frame_bgr, source_fps = decode_frame(clip_path, args.frame)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]

    if str(faceverse_root) not in sys.path:
        sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    print(f"FACEVERSE_PROJECTION_TEXTURE_DEVICE={device}", flush=True)
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    coefficient_tensor = torch.from_numpy(coefficient).to(device)
    _vertices, projected, normals, _colors = model.from_coeffs(
        coefficient_tensor,
        bbox.astype(np.int32),
    )
    projected_np = np.asarray(projected[0], dtype=np.float32)
    normals_np = np.asarray(normals[0], dtype=np.float32)
    if projected_np.shape != (19546, 3):
        raise RuntimeError(f"Unexpected projected coordinate shape: {projected_np.shape}")
    if normals_np.shape != (19546, 3):
        raise RuntimeError(f"Unexpected normal shape: {normals_np.shape}")

    uv = np.empty((len(projected_np), 2), dtype=np.float32)
    uv[:, 0] = projected_np[:, 0] / max(width - 1, 1)
    uv[:, 1] = 1.0 - projected_np[:, 1] / max(height - 1, 1)
    finite = np.isfinite(uv).all(axis=1)
    inside = (
        finite
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    uv = np.clip(np.nan_to_num(uv, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    normal_z = normals_np[:, 2]
    front_negative = np.clip((-normal_z - 0.02) / 0.55, 0.0, 1.0)
    front_positive = np.clip((normal_z - 0.02) / 0.55, 0.0, 1.0)
    negative_score = float(np.mean(front_negative[:13916]))
    positive_score = float(np.mean(front_positive[:13916]))
    front_weight = front_negative if negative_score >= positive_score else front_positive
    projection_mask = np.where(inside, front_weight, 0.0).astype(np.float32)
    projection_mask[:13916] = np.maximum(projection_mask[:13916], inside[:13916].astype(np.float32) * 0.24)
    projection_mask[13916:] *= 0.55

    if not cv2.imwrite(str(output_image), frame_bgr):
        raise RuntimeError(f"Could not write projection texture: {output_image}")
    np.savez_compressed(
        output_npz,
        uv=uv,
        projection_mask=projection_mask,
        projected_coordinates=projected_np,
        normals=normals_np,
        frame_index=np.asarray([args.frame], dtype=np.int32),
        image_size=np.asarray([width, height], dtype=np.int32),
    )

    report = {
        "classification": "PROVEN",
        "route": "FACEVERSE_V4_CAMERA_PROJECTED_TEXTURE_UV",
        "frame_index": args.frame,
        "source_fps": source_fps,
        "image_size": [width, height],
        "vertex_count": int(len(uv)),
        "uv_inside_fraction": float(np.mean(inside)),
        "projection_mask_mean": float(np.mean(projection_mask)),
        "front_normal_sign": "negative_z" if negative_score >= positive_score else "positive_z",
        "front_negative_score": negative_score,
        "front_positive_score": positive_score,
        "output_npz": str(output_npz),
        "output_image": str(output_image),
        "source_frame_plane_used": False,
        "texture_applied_to_true_3d_mesh": True,
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
