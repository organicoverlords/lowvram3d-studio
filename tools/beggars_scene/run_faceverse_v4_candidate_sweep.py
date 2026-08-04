from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit FaceVerse v4 to representative public meme frames and build visual comparisons."
    )
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmarker", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--frames",
        default="12,14,16,18,19,21,23,25,27,29,31,33,35,37,39,41",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


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


def expanded_square(box: np.ndarray, width: int, height: int, margin: float = 0.28) -> tuple[int, int, int, int]:
    left, top, right, bottom = [float(value) for value in box]
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    side = max(right - left, bottom - top) * (1.0 + 2.0 * margin)
    x0 = max(0, int(round(center_x - side * 0.5)))
    y0 = max(0, int(round(center_y - side * 0.5)))
    x1 = min(width, int(round(center_x + side * 0.5)))
    y1 = min(height, int(round(center_y + side * 0.5)))
    return x0, y0, x1, y1


def crop_square(image: np.ndarray, crop: tuple[int, int, int, int], size: int = 360) -> np.ndarray:
    x0, y0, x1, y1 = crop
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise RuntimeError(f"Empty comparison crop: {crop}")
    return cv2.resize(region, (size, size), interpolation=cv2.INTER_LANCZOS4)


def make_tile(
    reference_rgb: np.ndarray,
    render_rgb: np.ndarray,
    crop: tuple[int, int, int, int],
    frame_index: int,
    timestamp: float,
    sharpness: float,
    mouth_open: float,
) -> np.ndarray:
    reference_crop = crop_square(reference_rgb, crop)
    render_crop = crop_square(render_rgb, crop)
    body = np.concatenate((reference_crop, render_crop), axis=1)
    tile = np.zeros((408, 720, 3), dtype=np.uint8)
    tile[48:, :, :] = body
    cv2.putText(
        tile,
        f"frame {frame_index:03d}  {timestamp:.2f}s  sharp {sharpness:.0f}  mouth {mouth_open:.3f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(tile, "PUBLIC SOURCE", (12, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, "FACEVERSE V4", (378, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    landmarker_path = Path(args.landmarker).resolve()
    clip_path = Path(args.clip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = [int(value.strip()) for value in args.frames.split(",") if value.strip()]

    for required in (faceverse_root, model_path, checkpoint_path, landmarker_path, clip_path):
        if not required.exists():
            raise SystemExit(f"Candidate sweep input is missing: {required}")

    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise SystemExit("Public clip has invalid FPS")

    candidates: list[dict[str, object]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            candidates.append({"frame_index": frame_index, "error": "decode_failed"})
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            bbox, eyes, landmarks = detect_face_box_and_eyes(frame_rgb, landmarker_path)
        except Exception as error:  # bounded candidate rejection
            candidates.append({"frame_index": frame_index, "error": str(error)})
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        eye_distance = max(distance(landmarks[33], landmarks[263]), 1e-6)
        mouth_open = float(distance(landmarks[13], landmarks[14]) / eye_distance)
        candidates.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "frame_rgb": frame_rgb,
                "bbox": bbox,
                "eyes": eyes,
                "sharpness": sharpness,
                "mouth_open": mouth_open,
            }
        )
    capture.release()

    valid = [row for row in candidates if "frame_rgb" in row]
    if len(valid) < 4:
        raise SystemExit(f"Only {len(valid)} candidate frames passed face detection")

    device = choose_device(args.device)
    print(f"FACEVERSE_SWEEP_DEVICE={device}", flush=True)
    model_load_start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    model_load_seconds = time.perf_counter() - model_load_start
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)

    tiles: list[np.ndarray] = []
    report_rows: list[dict[str, object]] = []
    for row in valid:
        frame_index = int(row["frame_index"])
        frame_rgb = np.asarray(row["frame_rgb"], dtype=np.uint8)
        bbox = np.asarray(row["bbox"], dtype=np.float32)
        eyes = np.asarray(row["eyes"], dtype=np.float32)
        inference_start = time.perf_counter()
        coefficients, bbox_list = model.process_imgs(
            frame_rgb[np.newaxis, ...], bbox.reshape(1, 1, 4)
        )
        coefficients[:, -4:] = torch.from_numpy(eyes.reshape(1, 4)).to(coefficients.device)
        vertices, projected, normals, colors = model.from_coeffs(coefficients, bbox_list)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        render_rgb, depth = render_fvr(
            frame_rgb,
            projected[0],
            triangles,
            normals[0],
            colors[0],
        )
        crop = expanded_square(bbox, frame_rgb.shape[1], frame_rgb.shape[0])
        tile = make_tile(
            frame_rgb,
            render_rgb,
            crop,
            frame_index,
            float(row["timestamp_seconds"]),
            float(row["sharpness"]),
            float(row["mouth_open"]),
        )
        compare_path = output_dir / f"candidate_{frame_index:03d}_compare.jpg"
        render_path = output_dir / f"candidate_{frame_index:03d}_render.png"
        cv2.imwrite(str(compare_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
        tiles.append(tile)
        coefficient_parts = model.split_coeffs_dict(coefficients)
        report_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "sharpness": float(row["sharpness"]),
                "mouth_open": float(row["mouth_open"]),
                "bbox": np.asarray(bbox_list, dtype=int).tolist(),
                "inference_seconds": inference_seconds,
                "render_nonzero_fraction": float(np.mean(np.any(render_rgb > 0, axis=2))),
                "expression_l2": float(torch.linalg.vector_norm(coefficient_parts["exp"]).item()),
                "rotation": coefficient_parts["angle"].detach().cpu().numpy()[0].astype(float).tolist(),
                "translation": coefficient_parts["trans"].detach().cpu().numpy()[0].astype(float).tolist(),
                "compare": compare_path.name,
                "render": render_path.name,
            }
        )
        print(f"FACEVERSE_SWEEP_FRAME={frame_index} INFERENCE_SECONDS={inference_seconds:.4f}", flush=True)

    columns = 2
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * 408, columns * 720, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * 408
        x = (index % columns) * 720
        sheet[y : y + 408, x : x + 720] = tile
    sheet_path = output_dir / "faceverse_candidate_sweep.jpg"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V4_PUBLIC_FRAME_CANDIDATE_SWEEP",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_seconds": model_load_seconds,
        "candidate_frames_requested": frame_indices,
        "candidate_frames_rendered": len(report_rows),
        "contact_sheet": sheet_path.name,
        "public_source_frames_packaged": True,
        "source_video_packaged": False,
        "candidates": report_rows,
        "rejections": [row for row in candidates if "error" in row],
    }
    (output_dir / "faceverse_candidate_sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
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
