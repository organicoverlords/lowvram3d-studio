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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_faceverse_v4_identity_fusion as helper


DEFAULT_ANCHORS = "21,31,35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a landmark-refined FaceVerse v4 head in canonical model coordinates for "
            "bounded Blender still diagnostics."
        )
    )
    parser.add_argument("--coefficients", required=True)
    parser.add_argument("--variant", default="flexible")
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmarker", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one canonical export anchor is required")
    return indices


def as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        raise RuntimeError(f"Invalid public clip metadata: fps={fps} width={width} height={height}")

    frames: dict[int, np.ndarray] = {}
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode public frame {frame_index}")
        frames[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    capture.release()
    return frames, fps, (width, height)


def source_projected_colors(
    frame_rgb: np.ndarray,
    projected: np.ndarray,
    pca_colors: np.ndarray,
    blend: float = 0.88,
) -> tuple[np.ndarray, float]:
    sampled, valid = helper.bilinear_sample_rgb(
        frame_rgb,
        np.asarray(projected[:, :2], dtype=np.float32),
    )
    result = np.clip(np.asarray(pca_colors, dtype=np.float32), 0.0, 1.0)
    result[valid] = np.clip(
        result[valid] * (1.0 - blend) + sampled[valid] * blend,
        0.0,
        1.0,
    )
    return result, float(np.mean(valid))


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    coefficient_path = Path(args.coefficients).resolve()
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    landmarker_path = Path(args.landmarker).resolve()
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
        landmarker_path,
        clip_path,
    ):
        if not required.exists():
            raise SystemExit(f"Canonical Blender export input is missing: {required}")

    refined = np.load(coefficient_path)
    if args.variant not in refined.files:
        raise RuntimeError(
            f"Requested refinement variant {args.variant!r} is absent; available={refined.files}"
        )
    frame_indices = np.asarray(refined["frame_indices"], dtype=np.int32).tolist()
    coefficient_matrix = np.asarray(refined[args.variant], dtype=np.float32)
    if coefficient_matrix.shape != (len(frame_indices), 621):
        raise RuntimeError(
            f"Unexpected refined coefficient shape: {coefficient_matrix.shape}; "
            f"frame_count={len(frame_indices)}"
        )
    lookup = {int(frame): row for row, frame in enumerate(frame_indices)}
    missing = [frame for frame in anchors if frame not in lookup]
    if missing:
        raise RuntimeError(f"Requested canonical anchors are absent from refinement output: {missing}")
    rows = [lookup[frame] for frame in anchors]
    selected_coefficients = coefficient_matrix[rows]

    frames_rgb, source_fps, image_size = decode_frames(clip_path, anchors)
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    print(f"FACEVERSE_CANONICAL_EXPORT_DEVICE={device}", flush=True)
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)

    crop_boxes: list[np.ndarray] = []
    detector_boxes: list[np.ndarray] = []
    for frame_index in anchors:
        frame_rgb = frames_rgb[frame_index]
        detector_box, _eyes, _landmarks = helper.detect_face_box_and_eyes(
            frame_rgb,
            landmarker_path,
        )
        _network_coefficients, bbox_list = model.process_imgs(
            frame_rgb[np.newaxis, ...],
            detector_box.reshape(1, 1, 4),
        )
        detector_boxes.append(detector_box.astype(np.float32))
        crop_boxes.append(np.asarray(bbox_list[0], dtype=np.int32))

    crop_box_matrix = np.stack(crop_boxes, axis=0)
    coefficient_tensor = torch.from_numpy(selected_coefficients).to(device)
    canonical_vertices, projected, normals, pca_colors = model.from_coeffs(
        coefficient_tensor,
        crop_box_matrix,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    canonical_vertices_np = as_numpy(canonical_vertices, np.float32)
    projected_np = as_numpy(projected, np.float32)
    normals_np = as_numpy(normals, np.float32)
    pca_colors_np = as_numpy(pca_colors, np.float32)
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)
    component_boundaries = np.asarray(model.fvd.get("ver_inds", []), dtype=np.int32).reshape(-1)

    if canonical_vertices_np.shape != (len(anchors), 19546, 3):
        raise RuntimeError(
            f"Unexpected canonical vertex shape: {canonical_vertices_np.shape}"
        )
    if triangles.shape != (38792, 3):
        raise RuntimeError(f"Unexpected FaceVerse triangle shape: {triangles.shape}")
    if component_boundaries.size < 6:
        raise RuntimeError(
            f"FaceVerse component boundaries are incomplete: {component_boundaries.tolist()}"
        )

    keyframe_index = anchors.index(31) if 31 in anchors else len(anchors) // 2
    keyframe_source = anchors[keyframe_index]
    source_colors, sampled_fraction = source_projected_colors(
        frames_rgb[keyframe_source],
        projected_np[keyframe_index],
        pca_colors_np[keyframe_index],
    )

    canonical_min = np.min(canonical_vertices_np, axis=(0, 1))
    canonical_max = np.max(canonical_vertices_np, axis=(0, 1))
    canonical_extent = canonical_max - canonical_min
    if not np.all(np.isfinite(canonical_vertices_np)):
        raise RuntimeError("Canonical FaceVerse vertices contain non-finite values")
    if canonical_extent[0] < 1.5 or canonical_extent[1] < 1.8 or canonical_extent[2] < 1.5:
        raise RuntimeError(
            f"Canonical FaceVerse bounds are implausibly small: extent={canonical_extent.tolist()}"
        )

    np.savez_compressed(
        output_npz,
        canonical_vertices=canonical_vertices_np,
        projected_vertices=projected_np,
        canonical_normals=normals_np,
        triangles=triangles,
        source_colors=source_colors.astype(np.float32),
        pca_colors=pca_colors_np[keyframe_index].astype(np.float32),
        component_boundaries=component_boundaries,
        refined_coefficients=selected_coefficients,
        detector_boxes=np.stack(detector_boxes, axis=0).astype(np.float32),
        crop_boxes=crop_box_matrix.astype(np.int32),
        source_frame_indices=np.asarray(anchors, dtype=np.int32),
        source_fps=np.asarray([source_fps], dtype=np.float32),
        image_size=np.asarray(image_size, dtype=np.int32),
        keyframe_index=np.asarray([keyframe_index], dtype=np.int32),
    )

    report: dict[str, Any] = {
        "classification": "PROVEN",
        "route": "FACEVERSE_V4_REFINED_CANONICAL_TO_BLENDER_DIAGNOSTIC",
        "refinement_variant": args.variant,
        "anchors": anchors,
        "keyframe_index": keyframe_index,
        "keyframe_source_frame": keyframe_source,
        "vertex_count": int(canonical_vertices_np.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "component_boundaries": component_boundaries.astype(int).tolist(),
        "canonical_bounds": {
            "min": canonical_min.astype(float).tolist(),
            "max": canonical_max.astype(float).tolist(),
            "extent": canonical_extent.astype(float).tolist(),
        },
        "source_sampled_vertex_fraction": sampled_fraction,
        "image_size": list(image_size),
        "source_fps": source_fps,
        "coordinate_contract": {
            "faceverse_x": "Blender +X horizontal",
            "faceverse_y": "Blender -Z vertical",
            "faceverse_z": "Blender +Y depth; negative values face the camera",
            "projected_image_rescaling_used": False,
        },
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
