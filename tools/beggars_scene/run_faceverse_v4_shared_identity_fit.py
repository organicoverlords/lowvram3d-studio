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
import torch.nn.functional as torch_functional


DEFAULT_FRAMES = "19,21,23,25,27,29,31,33,35"
DEFAULT_ANCHORS = "21,31,35"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one FaceVerse v4 identity across multiple public frames using differentiable "
            "MediaPipe landmark reprojection, then compare baseline and optimized 3D renders."
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
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def stage(name: str) -> None:
    print(f"FACEVERSE_SHARED_FIT_STAGE={name}", flush=True)


def parse_indices(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one frame index is required")
    return result


def norm(vector: np.ndarray) -> np.ndarray:
    length = float(np.sqrt((vector**2).sum()))
    return np.zeros_like(vector) if length <= 1e-8 else vector / length


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(((left - right) ** 2).sum()))


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    if landmarks.shape[0] < 478:
        raise RuntimeError(f"MediaPipe returned only {landmarks.shape[0]} landmarks; 478 required")

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
    return bbox, eyes, landmarks[:478, :2]


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


def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def source_projected_colors(
    frame_rgb: np.ndarray,
    projected: np.ndarray | torch.Tensor,
    pca_colors: np.ndarray | torch.Tensor,
    blend: float = 0.82,
) -> tuple[np.ndarray, float]:
    points = as_numpy(projected).astype(np.float32)
    base = as_numpy(pca_colors).astype(np.float32)
    sampled, valid = bilinear_sample_rgb(frame_rgb, points[:, :2])
    result = np.clip(base, 0.0, 1.0)
    result[valid] = np.clip(base[valid] * (1.0 - blend) + sampled[valid] * blend, 0.0, 1.0)
    return result, float(np.mean(valid))


def make_tile(
    reference_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    optimized_rgb: np.ndarray,
    crop: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> np.ndarray:
    body = np.concatenate(
        (
            crop_square(reference_rgb, crop),
            crop_square(baseline_rgb, crop),
            crop_square(optimized_rgb, crop),
        ),
        axis=1,
    )
    tile = np.zeros((386, 960, 3), dtype=np.uint8)
    tile[46:366, :, :] = body
    cv2.putText(tile, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle, (12, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "PUBLIC SOURCE", (8, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "FUSED BASELINE", (332, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "LANDMARK OPTIMIZED", (647, 361), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def build_landmark_weights(device: torch.device) -> torch.Tensor:
    weights = torch.ones(478, dtype=torch.float32, device=device)
    weights[468:478] = 0.35
    weights[0:17] = 1.35
    for index in (1, 2, 4, 5, 6, 9, 10, 13, 14, 33, 61, 152, 168, 263, 291):
        weights[index] = 1.8
    return weights.view(1, 478, 1)


def map_landmarks_to_source(
    model: Any,
    coefficients: torch.Tensor,
    bbox_tensor: torch.Tensor,
) -> torch.Tensor:
    output = model.run(coefficients, only_lms=True)
    projected = output["lms_proj"][:, :478, :2]
    widths = (bbox_tensor[:, 2] - bbox_tensor[:, 0]).view(-1, 1)
    heights = (bbox_tensor[:, 3] - bbox_tensor[:, 1]).view(-1, 1)
    mapped_x = projected[:, :, 0] / float(model.imgsize) * widths + bbox_tensor[:, 0].view(-1, 1)
    mapped_y = projected[:, :, 1] / float(model.imgsize) * heights + bbox_tensor[:, 1].view(-1, 1)
    return torch.stack((mapped_x, mapped_y), dim=2)


def landmark_rmse_pixels(predicted: torch.Tensor, target: torch.Tensor) -> float:
    squared = torch.sum((predicted - target) ** 2, dim=2)
    return float(torch.sqrt(torch.mean(squared)).detach().cpu().item())


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
            raise SystemExit(f"Shared-fit input is missing: {required}")

    stage("LOAD_FRAMES")
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise SystemExit("Public clip has invalid FPS")
    records: dict[int, dict[str, Any]] = {}
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
        except Exception as error:
            rejections.append({"frame_index": frame_index, "error": str(error)})
            continue
        sharpness = float(cv2.Laplacian(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        eye_distance = max(distance(landmarks[33], landmarks[263]), 1e-6)
        mouth_open = float(distance(landmarks[13], landmarks[14]) / eye_distance)
        records[frame_index] = {
            "frame_rgb": frame_rgb,
            "bbox": bbox,
            "eyes": eyes,
            "landmarks": landmarks,
            "sharpness": sharpness,
            "mouth_open": mouth_open,
            "timestamp_seconds": frame_index / fps,
        }
    capture.release()
    ordered_indices = sorted(records)
    if len(ordered_indices) < 5:
        raise SystemExit(f"Only {len(ordered_indices)} frames passed shared-fit detection")
    missing_anchors = [index for index in anchor_indices if index not in records]
    if missing_anchors:
        raise SystemExit(f"Shared-fit anchors failed detection: {missing_anchors}")

    stage("LOAD_FACEVERSE")
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    print(f"FACEVERSE_SHARED_FIT_DEVICE={device}", flush=True)
    model_load_start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    model_load_seconds = time.perf_counter() - model_load_start
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)

    stage("INFER_INITIAL_COEFFICIENTS")
    raw_rows: list[np.ndarray] = []
    bbox_rows: list[np.ndarray] = []
    landmark_rows: list[np.ndarray] = []
    for frame_index in ordered_indices:
        record = records[frame_index]
        frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)
        bbox = np.asarray(record["bbox"], dtype=np.float32)
        coefficients, bbox_list = model.process_imgs(frame_rgb[np.newaxis, ...], bbox.reshape(1, 1, 4))
        coefficients[:, -4:] = torch.from_numpy(np.asarray(record["eyes"], dtype=np.float32).reshape(1, 4)).to(device)
        raw_rows.append(coefficients.detach().cpu().numpy()[0].astype(np.float32))
        bbox_rows.append(np.asarray(bbox_list[0], dtype=np.float32))
        landmark_rows.append(np.asarray(record["landmarks"], dtype=np.float32))
    raw_matrix = np.stack(raw_rows, axis=0)
    bbox_matrix = np.stack(bbox_rows, axis=0)
    target_landmarks_np = np.stack(landmark_rows, axis=0)

    id_end = int(model.id_dims)
    exp_end = id_end + int(model.exp_dims)
    tex_end = exp_end + int(model.tex_dims)
    gamma_end = tex_end + 27
    angle_end = gamma_end + 3
    trans_end = angle_end + 3
    eyes_end = trans_end + 4
    if eyes_end != raw_matrix.shape[1]:
        raise RuntimeError(f"Unexpected coefficient layout: expected {eyes_end}, got {raw_matrix.shape[1]}")

    initial_shared_id_np = np.median(raw_matrix[:, :id_end], axis=0).astype(np.float32)
    shared_texture_np = np.median(raw_matrix[:, exp_end:tex_end], axis=0).astype(np.float32)
    raw_tensor = torch.from_numpy(raw_matrix).to(device)
    bbox_tensor = torch.from_numpy(bbox_matrix).to(device)
    target_landmarks = torch.from_numpy(target_landmarks_np).to(device)
    scale_tensor = torch.maximum(
        bbox_tensor[:, 2] - bbox_tensor[:, 0],
        bbox_tensor[:, 3] - bbox_tensor[:, 1],
    ).clamp_min(1.0).view(-1, 1, 1)
    landmark_weights = build_landmark_weights(device)

    shared_id = torch.nn.Parameter(torch.from_numpy(initial_shared_id_np).to(device).view(1, -1))
    expression = torch.nn.Parameter(raw_tensor[:, id_end:exp_end].detach().clone())
    angles = torch.nn.Parameter(raw_tensor[:, gamma_end:angle_end].detach().clone())
    translation = torch.nn.Parameter(raw_tensor[:, angle_end:trans_end].detach().clone())
    eyes = torch.nn.Parameter(raw_tensor[:, trans_end:eyes_end].detach().clone())
    expression_initial = expression.detach().clone()
    angles_initial = angles.detach().clone()
    translation_initial = translation.detach().clone()
    eyes_initial = eyes.detach().clone()
    shared_id_initial = shared_id.detach().clone()
    fixed_texture = raw_tensor[:, exp_end:tex_end].detach().clone()
    fixed_gamma = raw_tensor[:, tex_end:gamma_end].detach().clone()

    def compose_coefficients() -> torch.Tensor:
        return torch.cat(
            (
                shared_id.expand(len(ordered_indices), -1),
                expression,
                fixed_texture,
                fixed_gamma,
                angles,
                translation,
                eyes,
            ),
            dim=1,
        )

    with torch.no_grad():
        initial_predicted = map_landmarks_to_source(model, compose_coefficients(), bbox_tensor)
        initial_rmse = landmark_rmse_pixels(initial_predicted, target_landmarks)

    stage("OPTIMIZE_SHARED_IDENTITY")
    optimizer = torch.optim.Adam(
        [
            {"params": [shared_id], "lr": 0.025},
            {"params": [expression], "lr": 0.010},
            {"params": [angles], "lr": 0.0025},
            {"params": [translation], "lr": 0.0040},
            {"params": [eyes], "lr": 0.0020},
        ]
    )
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    optimization_start = time.perf_counter()
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        coefficients = compose_coefficients()
        predicted = map_landmarks_to_source(model, coefficients, bbox_tensor)
        normalized_predicted = predicted / scale_tensor
        normalized_target = target_landmarks / scale_tensor
        data_loss = torch_functional.smooth_l1_loss(
            normalized_predicted * landmark_weights,
            normalized_target * landmark_weights,
            beta=0.006,
        )
        id_prior = torch.mean((shared_id - shared_id_initial) ** 2)
        expression_prior = torch.mean((expression - expression_initial) ** 2)
        angle_prior = torch.mean((angles - angles_initial) ** 2)
        translation_prior = torch.mean((translation - translation_initial) ** 2)
        eye_prior = torch.mean((eyes - eyes_initial) ** 2)
        loss = (
            data_loss
            + 0.0010 * id_prior
            + 0.0060 * expression_prior
            + 0.0120 * angle_prior
            + 0.0040 * translation_prior
            + 0.0120 * eye_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([shared_id, expression, angles, translation, eyes], max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            shared_id.clamp_(shared_id_initial - 1.5, shared_id_initial + 1.5)
            expression.clamp_(expression_initial - 1.0, expression_initial + 1.0)
            angles.clamp_(angles_initial - 0.35, angles_initial + 0.35)
            translation.clamp_(translation_initial - 0.45, translation_initial + 0.45)
            eyes.clamp_(-1.0, 1.0)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "shared_id": shared_id.detach().clone(),
                "expression": expression.detach().clone(),
                "angles": angles.detach().clone(),
                "translation": translation.detach().clone(),
                "eyes": eyes.detach().clone(),
            }
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == args.iterations:
            rmse = landmark_rmse_pixels(predicted, target_landmarks)
            row = {
                "iteration": float(iteration + 1),
                "loss": loss_value,
                "data_loss": float(data_loss.detach().cpu().item()),
                "rmse_pixels": rmse,
            }
            history.append(row)
            print(
                f"FACEVERSE_SHARED_FIT_ITER={iteration + 1} LOSS={loss_value:.8f} RMSE_PIXELS={rmse:.4f}",
                flush=True,
            )
    optimization_seconds = time.perf_counter() - optimization_start
    if best_state is None:
        raise RuntimeError("Shared identity optimization produced no state")
    with torch.no_grad():
        shared_id.copy_(best_state["shared_id"])
        expression.copy_(best_state["expression"])
        angles.copy_(best_state["angles"])
        translation.copy_(best_state["translation"])
        eyes.copy_(best_state["eyes"])
        final_coefficients = compose_coefficients().detach().clone()
        final_predicted = map_landmarks_to_source(model, final_coefficients, bbox_tensor)
        final_rmse = landmark_rmse_pixels(final_predicted, target_landmarks)

    stage("RENDER_COMPARISONS")
    baseline_coefficients = raw_tensor.detach().clone()
    baseline_coefficients[:, :id_end] = torch.from_numpy(initial_shared_id_np).to(device).view(1, -1)
    baseline_coefficients[:, exp_end:tex_end] = torch.from_numpy(shared_texture_np).to(device).view(1, -1)
    final_coefficients[:, exp_end:tex_end] = torch.from_numpy(shared_texture_np).to(device).view(1, -1)

    index_lookup = {frame_index: row for row, frame_index in enumerate(ordered_indices)}
    tiles: list[np.ndarray] = []
    render_rows: list[dict[str, Any]] = []
    for anchor_index in anchor_indices:
        row_index = index_lookup[anchor_index]
        record = records[anchor_index]
        frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)
        bbox_list = bbox_matrix[row_index : row_index + 1].astype(np.int32)
        baseline_vs, baseline_projected, baseline_normals, baseline_pca = model.from_coeffs(
            baseline_coefficients[row_index : row_index + 1], bbox_list
        )
        optimized_vs, optimized_projected, optimized_normals, optimized_pca = model.from_coeffs(
            final_coefficients[row_index : row_index + 1], bbox_list
        )
        baseline_colors, baseline_sampled = source_projected_colors(
            frame_rgb, baseline_projected[0], baseline_pca[0]
        )
        optimized_colors, optimized_sampled = source_projected_colors(
            frame_rgb, optimized_projected[0], optimized_pca[0]
        )
        baseline_render, _ = render_fvr(
            frame_rgb,
            baseline_projected[0],
            triangles,
            baseline_normals[0],
            baseline_colors,
        )
        optimized_render, _ = render_fvr(
            frame_rgb,
            optimized_projected[0],
            triangles,
            optimized_normals[0],
            optimized_colors,
        )
        crop = expanded_square(
            np.asarray(record["bbox"], dtype=np.float32),
            frame_rgb.shape[1],
            frame_rgb.shape[0],
        )
        per_frame_initial = float(
            torch.sqrt(
                torch.mean(torch.sum((initial_predicted[row_index] - target_landmarks[row_index]) ** 2, dim=1))
            ).detach().cpu().item()
        )
        per_frame_final = float(
            torch.sqrt(
                torch.mean(torch.sum((final_predicted[row_index] - target_landmarks[row_index]) ** 2, dim=1))
            ).detach().cpu().item()
        )
        title = (
            f"anchor {anchor_index:03d} | mouth {float(record['mouth_open']):.3f} | "
            f"landmark RMSE {per_frame_initial:.2f}->{per_frame_final:.2f}px"
        )
        subtitle = (
            f"shared fit {len(ordered_indices)} frames / {args.iterations} iterations | "
            f"sampled {optimized_sampled:.3f} | source plane false"
        )
        tile = make_tile(frame_rgb, baseline_render, optimized_render, crop, title, subtitle)
        compare_path = output_dir / f"shared_fit_anchor_{anchor_index:03d}_compare.jpg"
        baseline_path = output_dir / f"shared_fit_anchor_{anchor_index:03d}_baseline.png"
        optimized_path = output_dir / f"shared_fit_anchor_{anchor_index:03d}_optimized.png"
        cv2.imwrite(str(compare_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(baseline_path), cv2.cvtColor(baseline_render, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(optimized_path), cv2.cvtColor(optimized_render, cv2.COLOR_RGB2BGR))
        tiles.append(tile)
        render_rows.append(
            {
                "anchor_frame": anchor_index,
                "anchor_timestamp_seconds": float(record["timestamp_seconds"]),
                "anchor_sharpness": float(record["sharpness"]),
                "anchor_mouth_open": float(record["mouth_open"]),
                "initial_landmark_rmse_pixels": per_frame_initial,
                "final_landmark_rmse_pixels": per_frame_final,
                "baseline_source_sampled_fraction": baseline_sampled,
                "optimized_source_sampled_fraction": optimized_sampled,
                "compare": compare_path.name,
                "baseline_render": baseline_path.name,
                "optimized_render": optimized_path.name,
            }
        )

    sheet = np.zeros((len(tiles) * 386, 960, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        sheet[index * 386 : (index + 1) * 386, :, :] = tile
    sheet_path = output_dir / "faceverse_shared_identity_fit.jpg"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        output_dir / "faceverse_shared_identity_fit_coefficients.npz",
        frame_indices=np.asarray(ordered_indices, dtype=np.int32),
        raw_coefficients=raw_matrix,
        bbox_matrix=bbox_matrix,
        target_landmarks=target_landmarks_np,
        initial_shared_identity=initial_shared_id_np,
        optimized_shared_identity=shared_id.detach().cpu().numpy()[0].astype(np.float32),
        optimized_coefficients=final_coefficients.detach().cpu().numpy().astype(np.float32),
        optimized_expression=expression.detach().cpu().numpy().astype(np.float32),
        optimized_angles=angles.detach().cpu().numpy().astype(np.float32),
        optimized_translation=translation.detach().cpu().numpy().astype(np.float32),
        optimized_eyes=eyes.detach().cpu().numpy().astype(np.float32),
        shared_texture=shared_texture_np,
    )

    identity_shift_l2 = float(
        torch.linalg.vector_norm(shared_id.detach() - shared_id_initial).cpu().item()
    )
    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V4_SHARED_MULTI_FRAME_LANDMARK_IDENTITY_FIT",
        "claim": (
            "One shared 156-dimensional identity was optimized across multiple public frames using "
            "FaceVerse landmark-only differentiable projection; per-frame expression, pose, translation "
            "and eyes were regularized around network predictions."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_seconds": model_load_seconds,
        "optimization_seconds": optimization_seconds,
        "iterations": args.iterations,
        "frames_requested": frame_indices,
        "frames_used": ordered_indices,
        "anchor_frames": anchor_indices,
        "initial_global_landmark_rmse_pixels": initial_rmse,
        "final_global_landmark_rmse_pixels": final_rmse,
        "landmark_rmse_improvement_fraction": (
            float((initial_rmse - final_rmse) / initial_rmse) if initial_rmse > 1e-8 else 0.0
        ),
        "identity_shift_l2": identity_shift_l2,
        "best_loss": best_loss,
        "optimization_history": history,
        "renders": render_rows,
        "contact_sheet": sheet_path.name,
        "coefficients": "faceverse_shared_identity_fit_coefficients.npz",
        "rejections": rejections,
        "public_source_frames_packaged": True,
        "source_video_packaged": False,
        "source_frame_plane_used": False,
    }
    report_path = output_dir / "faceverse_shared_identity_fit.json"
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
