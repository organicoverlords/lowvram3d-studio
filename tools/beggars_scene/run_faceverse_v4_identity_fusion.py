from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch


DEFAULT_FRAMES = "19,21,23,25,27,29,31,33,35"
DEFAULT_ANCHORS = "21,31,35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse FaceVerse v4 identity and texture coefficients across multiple public meme frames, "
            "then render expression/pose anchors with both PCA and source-projected vertex colors."
        )
    )
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmarker", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def stage(name: str) -> None:
    print(f"FACEVERSE_FUSION_STAGE={name}", flush=True)


def norm(vector: np.ndarray) -> np.ndarray:
    length = float(np.sqrt((vector**2).sum()))
    return np.zeros_like(vector) if length <= 1e-8 else vector / length


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(((left - right) ** 2).sum()))


def detect_face_box_and_eyes(
    image_rgb: np.ndarray, landmarker_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(
                mp.ImageFormat.SRGB,
                np.ascontiguousarray(image_rgb.astype(np.uint8)),
            )
        )
    if not result.face_landmarks:
        raise RuntimeError("MediaPipe did not detect a face")
    landmarks = np.asarray(
        [
            (landmark.x * image_rgb.shape[1], landmark.y * image_rgb.shape[0])
            for landmark in result.face_landmarks[0]
        ],
        dtype=np.float32,
    )
    if landmarks.shape[0] < 474:
        raise RuntimeError(f"MediaPipe returned only {landmarks.shape[0]} landmarks")

    left_vector = norm(landmarks[362, :2] - landmarks[263, :2])
    left_distance = max(distance(landmarks[362, :2], landmarks[263, :2]), 1e-6)
    left_center = (landmarks[263, :2] + landmarks[362, :2]) * 0.5
    left_eye_x = float(np.dot(landmarks[473] - left_center, left_vector) / left_distance * 3.0)
    left_eye_y = float(
        np.dot(landmarks[473] - left_center, left_vector[[1, 0]]) / left_distance * -1.5
    )

    right_vector = norm(landmarks[33, :2] - landmarks[133, :2])
    right_distance = max(distance(landmarks[33, :2], landmarks[133, :2]), 1e-6)
    right_center = (landmarks[33, :2] + landmarks[133, :2]) * 0.5
    right_eye_x = float(np.dot(landmarks[468] - right_center, right_vector) / right_distance * 3.0)
    right_eye_y = float(
        np.dot(landmarks[468] - right_center, right_vector[[1, 0]]) / right_distance * -1.5
    )

    bbox = np.asarray(
        [
            float(np.min(landmarks[:, 0])),
            float(np.min(landmarks[:, 1])),
            float(np.max(landmarks[:, 0])),
            float(np.max(landmarks[:, 1])),
        ],
        dtype=np.float32,
    )
    eyes = np.asarray([left_eye_y, left_eye_x, right_eye_y, right_eye_x], dtype=np.float32)
    return bbox, eyes, landmarks


def expanded_square(
    box: np.ndarray, width: int, height: int, margin: float = 0.30
) -> tuple[int, int, int, int]:
    left, top, right, bottom = [float(value) for value in box]
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    side = max(right - left, bottom - top) * (1.0 + 2.0 * margin)
    x0 = max(0, int(round(center_x - side * 0.5)))
    y0 = max(0, int(round(center_y - side * 0.5)))
    x1 = min(width, int(round(center_x + side * 0.5)))
    y1 = min(height, int(round(center_y + side * 0.5)))
    return x0, y0, x1, y1


def crop_square(
    image: np.ndarray, crop: tuple[int, int, int, int], size: int = 320
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise RuntimeError(f"Empty comparison crop: {crop}")
    return cv2.resize(region, (size, size), interpolation=cv2.INTER_LANCZOS4)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one frame index is required")
    return indices


def trimmed_mean(values: np.ndarray, trim_fraction: float = 0.2) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D values, got {values.shape}")
    count = values.shape[0]
    trim = int(math.floor(count * trim_fraction))
    if trim == 0 or count - 2 * trim < 1:
        return np.mean(values, axis=0)
    ordered = np.sort(values, axis=0)
    return np.mean(ordered[trim : count - trim], axis=0)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("Fusion weights do not match coefficient rows")
    total = float(np.sum(weights))
    if total <= 1e-12:
        raise ValueError("Fusion weights sum to zero")
    normalized = weights / total
    return np.sum(values * normalized[:, None], axis=0)


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


def source_projected_colors(
    frame_rgb: np.ndarray,
    projected: torch.Tensor,
    pca_colors: torch.Tensor,
    blend: float = 0.82,
) -> tuple[np.ndarray, float]:
    points = projected.detach().cpu().numpy().astype(np.float32)
    base = pca_colors.detach().cpu().numpy().astype(np.float32)
    sampled, valid = bilinear_sample_rgb(frame_rgb, points[:, :2])
    result = np.clip(base, 0.0, 1.0)
    result[valid] = np.clip(
        base[valid] * (1.0 - blend) + sampled[valid] * blend,
        0.0,
        1.0,
    )
    return result, float(np.mean(valid))


def make_tile(
    reference_rgb: np.ndarray,
    pca_render_rgb: np.ndarray,
    projected_render_rgb: np.ndarray,
    crop: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> np.ndarray:
    reference_crop = crop_square(reference_rgb, crop)
    pca_crop = crop_square(pca_render_rgb, crop)
    projected_crop = crop_square(projected_render_rgb, crop)
    body = np.concatenate((reference_crop, pca_crop, projected_crop), axis=1)
    tile = np.zeros((386, 960, 3), dtype=np.uint8)
    tile[46:366, :, :] = body
    cv2.putText(
        tile,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        subtitle,
        (12, 383),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(tile, "PUBLIC SOURCE", (8, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "FUSED PCA", (334, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "FUSED SOURCE COLOR", (649, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    frame_indices = parse_indices(args.frames)
    anchor_indices = parse_indices(args.anchors)
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    landmarker_path = Path(args.landmarker).resolve()
    clip_path = Path(args.clip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (faceverse_root, model_path, checkpoint_path, landmarker_path, clip_path):
        if not required.exists():
            raise SystemExit(f"Identity-fusion input is missing: {required}")

    stage("LOAD_PUBLIC_FRAMES")
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise SystemExit("Public clip has invalid FPS")

    frame_records: dict[int, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            rejections.append({"frame_index": frame_index, "error": "decode_failed"})
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            bbox, eyes, landmarks = detect_face_box_and_eyes(frame_rgb, landmarker_path)
        except Exception as error:  # bounded frame rejection
            rejections.append({"frame_index": frame_index, "error": str(error)})
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        eye_distance = max(distance(landmarks[33], landmarks[263]), 1e-6)
        mouth_open = float(distance(landmarks[13], landmarks[14]) / eye_distance)
        frame_records[frame_index] = {
            "frame_rgb": frame_rgb,
            "bbox": bbox,
            "eyes": eyes,
            "landmarks": landmarks,
            "sharpness": sharpness,
            "mouth_open": mouth_open,
            "timestamp_seconds": frame_index / fps,
        }
    capture.release()
    if len(frame_records) < 5:
        raise SystemExit(f"Only {len(frame_records)} frames passed identity-fusion detection")
    missing_anchors = [index for index in anchor_indices if index not in frame_records]
    if missing_anchors:
        raise SystemExit(f"Requested fusion anchors failed detection: {missing_anchors}")

    stage("LOAD_FACEVERSE")
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    print(f"FACEVERSE_FUSION_DEVICE={device}", flush=True)
    model_load_start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    model_load_seconds = time.perf_counter() - model_load_start
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)

    stage("INFER_FRAME_COEFFICIENTS")
    ordered_indices = sorted(frame_records)
    coefficient_rows: list[np.ndarray] = []
    for frame_index in ordered_indices:
        record = frame_records[frame_index]
        frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)
        bbox = np.asarray(record["bbox"], dtype=np.float32)
        start = time.perf_counter()
        coefficients, bbox_list = model.process_imgs(
            frame_rgb[np.newaxis, ...], bbox.reshape(1, 1, 4)
        )
        coefficients[:, -4:] = torch.from_numpy(
            np.asarray(record["eyes"], dtype=np.float32).reshape(1, 4)
        ).to(coefficients.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["coefficients"] = coefficients.detach().clone()
        record["bbox_list"] = bbox_list
        record["inference_seconds"] = time.perf_counter() - start
        coefficient_rows.append(coefficients.detach().cpu().numpy()[0].astype(np.float32))
        print(
            f"FACEVERSE_FUSION_FRAME={frame_index} INFERENCE_SECONDS={record['inference_seconds']:.4f}",
            flush=True,
        )

    coefficient_matrix = np.stack(coefficient_rows, axis=0)
    id_end = int(model.id_dims)
    exp_end = id_end + int(model.exp_dims)
    tex_end = exp_end + int(model.tex_dims)
    identity_rows = coefficient_matrix[:, :id_end]
    texture_rows = coefficient_matrix[:, exp_end:tex_end]

    sharpness = np.asarray([float(frame_records[index]["sharpness"]) for index in ordered_indices])
    yaw = np.asarray(
        [abs(float(coefficient_matrix[row_index, tex_end + 27 + 1])) for row_index in range(len(ordered_indices))]
    )
    sharp_scale = np.sqrt(np.maximum(sharpness, 1.0))
    frontal_weight = np.exp(-yaw / 0.70)
    robust_weights = sharp_scale * frontal_weight

    fusion_vectors = {
        "median": {
            "identity": np.median(identity_rows, axis=0),
            "texture": np.median(texture_rows, axis=0),
        },
        "trimmed": {
            "identity": trimmed_mean(identity_rows, trim_fraction=0.2),
            "texture": trimmed_mean(texture_rows, trim_fraction=0.2),
        },
        "sharp_frontal": {
            "identity": weighted_mean(identity_rows, robust_weights),
            "texture": weighted_mean(texture_rows, robust_weights),
        },
    }

    identity_center = np.median(identity_rows, axis=0)
    identity_distance = np.linalg.norm(identity_rows - identity_center[None, :], axis=1)
    texture_center = np.median(texture_rows, axis=0)
    texture_distance = np.linalg.norm(texture_rows - texture_center[None, :], axis=1)

    stage("RENDER_FUSION_VARIANTS")
    tiles: list[np.ndarray] = []
    variants: list[dict[str, Any]] = []
    saved_coefficients: dict[str, np.ndarray] = {}
    for fusion_name, fused in fusion_vectors.items():
        for anchor_index in anchor_indices:
            record = frame_records[anchor_index]
            anchor_coefficients = record["coefficients"].detach().clone()
            fused_coefficients = anchor_coefficients.detach().cpu().numpy().astype(np.float32)
            fused_coefficients[0, :id_end] = fused["identity"]
            fused_coefficients[0, exp_end:tex_end] = fused["texture"]
            fused_tensor = torch.from_numpy(fused_coefficients).to(device)
            bbox_list = record["bbox_list"]
            frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)

            render_start = time.perf_counter()
            vertices, projected, normals, pca_colors = model.from_coeffs(fused_tensor, bbox_list)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pca_render, _ = render_fvr(
                frame_rgb,
                projected[0],
                triangles,
                normals[0],
                pca_colors[0],
            )
            projected_colors, sampled_fraction = source_projected_colors(
                frame_rgb,
                projected[0],
                pca_colors[0],
            )
            projected_render, _ = render_fvr(
                frame_rgb,
                projected[0],
                triangles,
                normals[0],
                projected_colors,
            )
            render_seconds = time.perf_counter() - render_start
            crop = expanded_square(
                np.asarray(record["bbox"], dtype=np.float32),
                frame_rgb.shape[1],
                frame_rgb.shape[0],
            )
            title = (
                f"{fusion_name} identity+texture | anchor {anchor_index:03d} | "
                f"mouth {float(record['mouth_open']):.3f}"
            )
            subtitle = (
                f"sharp {float(record['sharpness']):.0f} | sampled vertices {sampled_fraction:.3f} | "
                f"render {render_seconds:.3f}s"
            )
            tile = make_tile(
                frame_rgb,
                pca_render,
                projected_render,
                crop,
                title,
                subtitle,
            )
            variant_key = f"{fusion_name}_anchor_{anchor_index:03d}"
            compare_path = output_dir / f"fusion_{variant_key}_compare.jpg"
            pca_path = output_dir / f"fusion_{variant_key}_pca.png"
            projected_path = output_dir / f"fusion_{variant_key}_projected.png"
            cv2.imwrite(str(compare_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(pca_path), cv2.cvtColor(pca_render, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(projected_path), cv2.cvtColor(projected_render, cv2.COLOR_RGB2BGR))
            tiles.append(tile)
            saved_coefficients[variant_key] = fused_coefficients[0]
            variants.append(
                {
                    "variant": variant_key,
                    "fusion": fusion_name,
                    "anchor_frame": anchor_index,
                    "anchor_timestamp_seconds": float(record["timestamp_seconds"]),
                    "anchor_sharpness": float(record["sharpness"]),
                    "anchor_mouth_open": float(record["mouth_open"]),
                    "source_sampled_vertex_fraction": sampled_fraction,
                    "render_seconds": render_seconds,
                    "compare": compare_path.name,
                    "pca_render": pca_path.name,
                    "projected_render": projected_path.name,
                }
            )
            print(f"FACEVERSE_FUSION_VARIANT={variant_key}", flush=True)

    columns = 1
    rows = len(tiles)
    sheet = np.zeros((rows * 386, columns * 960, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        sheet[index * 386 : (index + 1) * 386, :960] = tile
    sheet_path = output_dir / "faceverse_identity_fusion.jpg"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        output_dir / "faceverse_identity_fusion_coefficients.npz",
        frame_indices=np.asarray(ordered_indices, dtype=np.int32),
        raw_coefficients=coefficient_matrix,
        identity_median=fusion_vectors["median"]["identity"].astype(np.float32),
        texture_median=fusion_vectors["median"]["texture"].astype(np.float32),
        identity_trimmed=fusion_vectors["trimmed"]["identity"].astype(np.float32),
        texture_trimmed=fusion_vectors["trimmed"]["texture"].astype(np.float32),
        identity_sharp_frontal=fusion_vectors["sharp_frontal"]["identity"].astype(np.float32),
        texture_sharp_frontal=fusion_vectors["sharp_frontal"]["texture"].astype(np.float32),
        robust_weights=robust_weights.astype(np.float32),
        **{key: value.astype(np.float32) for key, value in saved_coefficients.items()},
    )

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V4_MULTI_FRAME_IDENTITY_TEXTURE_FUSION",
        "claim": (
            "Shared identity and texture coefficients are fused across multiple detected public frames; "
            "each variant retains expression, pose, translation and eye coefficients from its anchor frame."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_seconds": model_load_seconds,
        "input_frames_requested": frame_indices,
        "input_frames_used": ordered_indices,
        "anchor_frames": anchor_indices,
        "fusion_methods": list(fusion_vectors),
        "identity_dimensions": id_end,
        "expression_dimensions": int(model.exp_dims),
        "texture_dimensions": int(model.tex_dims),
        "identity_distance_from_median": {
            str(index): float(identity_distance[row]) for row, index in enumerate(ordered_indices)
        },
        "texture_distance_from_median": {
            str(index): float(texture_distance[row]) for row, index in enumerate(ordered_indices)
        },
        "fusion_weights": {
            str(index): float(robust_weights[row]) for row, index in enumerate(ordered_indices)
        },
        "contact_sheet": sheet_path.name,
        "coefficients": "faceverse_identity_fusion_coefficients.npz",
        "variants": variants,
        "rejections": rejections,
        "public_source_frames_packaged": True,
        "source_video_packaged": False,
        "source_frame_plane_used": False,
    }
    report_path = output_dir / "faceverse_identity_fusion.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    stage("COMPLETE")
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
