from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a bounded meme clip sequence and reconstruct a temporally smoothed 3D face track."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--third-party-root", required=True)
    parser.add_argument("--yunet-model", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--keyframe-output", required=True)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--smoothing", type=float, default=0.68)
    return parser.parse_args()


def expanded_bbox(raw: np.ndarray, width: int, height: int, margin: float = 0.18) -> list[float]:
    x, y, w, h = [float(value) for value in raw[:4]]
    cx = x + w * 0.5
    cy = y + h * 0.5
    side_w = w * (1.0 + margin * 2.0)
    side_h = h * (1.0 + margin * 2.0)
    left = max(0.0, cx - side_w * 0.5)
    top = max(0.0, cy - side_h * 0.5)
    right = min(float(width - 1), cx + side_w * 0.5)
    bottom = min(float(height - 1), cy + side_h * 0.5)
    return [left, top, right, bottom, float(raw[-1])]


def choose_detection(faces: np.ndarray, previous: list[float] | None) -> np.ndarray:
    if faces is None or len(faces) == 0:
        raise ValueError("No detections supplied")
    if previous is None:
        return max(faces, key=lambda row: float(row[2] * row[3] * max(row[-1], 0.01)))

    px = (previous[0] + previous[2]) * 0.5
    py = (previous[1] + previous[3]) * 0.5

    def score(row: np.ndarray) -> float:
        cx = float(row[0] + row[2] * 0.5)
        cy = float(row[1] + row[3] * 0.5)
        distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        area = float(row[2] * row[3])
        return area * float(max(row[-1], 0.01)) / (1.0 + distance)

    return max(faces, key=score)


def sample_colors(image_bgr: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x = np.clip(np.rint(vertices[0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(vertices[1]).astype(np.int32), 0, height - 1)
    return (image_bgr[y, x][:, ::-1].astype(np.float32) / 255.0).copy()


def face_sharpness(image: np.ndarray, bbox: list[float]) -> float:
    left, top, right, bottom = [int(round(value)) for value in bbox[:4]]
    crop = image[max(top, 0) : max(bottom, top + 1), max(left, 0) : max(right, left + 1)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> int:
    args = parse_args()
    clip_path = Path(args.clip).resolve()
    third_party = Path(args.third_party_root).resolve()
    yunet_model = Path(args.yunet_model).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_report = Path(args.output_report).resolve()
    keyframe_output = Path(args.keyframe_output).resolve()

    if not clip_path.is_file():
        raise SystemExit(f"Reference clip is missing: {clip_path}")
    if not third_party.is_dir():
        raise SystemExit(f"Pinned 3DDFA checkout is missing: {third_party}")
    if not yunet_model.is_file():
        raise SystemExit(f"YuNet model is missing: {yunet_model}")
    if not 0.0 < args.smoothing <= 1.0:
        raise SystemExit("--smoothing must be within (0, 1]")

    frames_dir.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    keyframe_output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(third_party))
    original_cwd = Path.cwd()
    os.chdir(third_party)
    try:
        from TDDFA_ONNX import TDDFA_ONNX  # pylint: disable=import-error,import-outside-toplevel

        config_path = third_party / "configs" / "mb1_120x120.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["checkpoint_fp"] = str((third_party / config["checkpoint_fp"]).resolve())
        config["onnx_fp"] = str((third_party / "weights" / "mb1_120x120.onnx").resolve())
        config["bfm_fp"] = str((third_party / config["bfm_fp"]).resolve())
        config["param_mean_std_fp"] = str(
            (third_party / "configs" / "param_mean_std_62d_120x120.pkl").resolve()
        )
        tddfa = TDDFA_ONNX(**config)
    finally:
        os.chdir(original_cwd)

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open reference clip: {clip_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit(
            f"Invalid clip metadata: fps={source_fps}, width={width}, height={height}, frames={source_frame_count}"
        )

    detector = cv2.FaceDetectorYN.create(
        str(yunet_model),
        "",
        (width, height),
        score_threshold=0.72,
        nms_threshold=0.3,
        top_k=5000,
    )

    sample_stride = max(1, int(round(source_fps / args.target_fps)))
    maximum_source_frames = min(
        source_frame_count if source_frame_count > 0 else int(source_fps * args.max_seconds),
        int(source_fps * args.max_seconds),
    )

    vertices_raw: list[np.ndarray] = []
    parameters: list[np.ndarray] = []
    boxes: list[list[float]] = []
    sampled_source_indices: list[int] = []
    sampled_paths: list[Path] = []
    sharpness_values: list[float] = []
    previous_box: list[float] | None = None
    source_index = 0

    while source_index < maximum_source_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % sample_stride != 0:
            source_index += 1
            continue

        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, detections = detector.detect(frame)
        if detections is None or len(detections) == 0:
            if previous_box is None:
                source_index += 1
                continue
            roi = previous_box
        else:
            selected = choose_detection(detections, previous_box)
            roi = expanded_bbox(selected, frame.shape[1], frame.shape[0])
            previous_box = roi

        os.chdir(third_party)
        try:
            param_list, roi_list = tddfa(frame, [roi])
            vertex_list = tddfa.recon_vers(param_list, roi_list, dense_flag=True)
        finally:
            os.chdir(original_cwd)

        if len(vertex_list) != 1:
            source_index += 1
            continue

        output_frame = frames_dir / f"reference_{len(vertices_raw):04d}.png"
        if not cv2.imwrite(str(output_frame), frame):
            raise SystemExit(f"Could not write extracted frame: {output_frame}")

        vertices_raw.append(vertex_list[0].astype(np.float32))
        parameters.append(np.asarray(param_list[0], dtype=np.float32))
        boxes.append([float(value) for value in roi])
        sampled_source_indices.append(source_index)
        sampled_paths.append(output_frame)
        sharpness_values.append(face_sharpness(frame, roi))
        source_index += 1

    capture.release()

    if len(vertices_raw) < 8:
        raise SystemExit(f"Face tracking produced only {len(vertices_raw)} usable frames; at least 8 are required")

    stack_raw = np.stack(vertices_raw, axis=0)
    stack_smoothed = np.empty_like(stack_raw)
    stack_smoothed[0] = stack_raw[0]
    alpha = float(args.smoothing)
    for index in range(1, stack_raw.shape[0]):
        stack_smoothed[index] = alpha * stack_raw[index] + (1.0 - alpha) * stack_smoothed[index - 1]

    start = int(round(len(sampled_paths) * 0.35))
    stop = max(start + 1, int(round(len(sampled_paths) * 0.88)))
    keyframe_index = max(range(start, stop), key=lambda index: sharpness_values[index])
    keyframe_image = cv2.imread(str(sampled_paths[keyframe_index]), cv2.IMREAD_COLOR)
    if keyframe_image is None:
        raise SystemExit(f"Could not reopen selected keyframe: {sampled_paths[keyframe_index]}")
    shutil.copy2(sampled_paths[keyframe_index], keyframe_output)

    colors_rgb = sample_colors(keyframe_image, stack_raw[keyframe_index])
    triangles = np.asarray(tddfa.tri, dtype=np.int32)

    np.savez_compressed(
        output_npz,
        vertices_raw=stack_raw,
        vertices_smoothed=stack_smoothed,
        triangles=triangles,
        colors_rgb=colors_rgb,
        parameters=np.stack(parameters, axis=0),
        boxes=np.asarray(boxes, dtype=np.float32),
        source_frame_indices=np.asarray(sampled_source_indices, dtype=np.int32),
        source_fps=np.asarray([source_fps], dtype=np.float32),
        sampled_fps=np.asarray([source_fps / sample_stride], dtype=np.float32),
        image_size=np.asarray([width, height], dtype=np.int32),
        keyframe_index=np.asarray([keyframe_index], dtype=np.int32),
    )

    report = {
        "classification": "PROVEN",
        "route": "YUNET_TRACK_PLUS_3DDFA_V2_ONNX",
        "third_party_commit": "1b6c67601abffc1e9f248b291708aef0e43b55ae",
        "source_clip": str(clip_path),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "source_dimensions": [width, height],
        "sample_stride": sample_stride,
        "sampled_frame_count": int(stack_smoothed.shape[0]),
        "sampled_fps": source_fps / sample_stride,
        "keyframe_index": keyframe_index,
        "keyframe_source_frame": sampled_source_indices[keyframe_index],
        "vertex_count": int(stack_smoothed.shape[2]),
        "triangle_count": int(triangles.shape[0]),
        "smoothing": alpha,
        "output_npz": str(output_npz),
        "keyframe_output": str(keyframe_output),
        "reference_media_policy": "Extracted frames and source clip are local temporary inputs and must not be committed or uploaded.",
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
