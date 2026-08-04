from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one packed, face-only RGBA sprite sheet from tracked private reference frames."
    )
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--columns", type=int, default=8)
    return parser.parse_args()


def stabilize_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(f"Expected tracked boxes with at least four columns, got {boxes.shape}")

    centers = np.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
        ],
        axis=1,
    )
    sides = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    valid_sides = sides[np.isfinite(sides) & (sides > 8.0)]
    if valid_sides.size == 0:
        raise RuntimeError("No valid tracked face boxes were available for sprite stabilization")
    global_side = float(np.median(valid_sides))

    # Preserve tracker metadata such as confidence columns while stabilizing
    # only the geometric x1/y1/x2/y2 coordinates.
    stabilized = boxes.copy()
    for index in range(len(boxes)):
        left = max(0, index - 2)
        right = min(len(boxes), index + 3)
        local_centers = centers[left:right]
        local_sides = sides[left:right]
        center = np.median(local_centers, axis=0)
        side = float(np.median(local_sides))
        side = float(np.clip(side, global_side * 0.82, global_side * 1.18))
        stabilized[index, :4] = (
            center[0] - side * 0.5,
            center[1] - side * 0.5,
            center[0] + side * 0.5,
            center[1] + side * 0.5,
        )
    return stabilized


def square_crop_bounds(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    center_x = float((box[0] + box[2]) * 0.5)
    center_y = float((box[1] + box[3]) * 0.5)
    side = max(float(box[2] - box[0]), float(box[3] - box[1]), 8.0) * 1.18
    side = min(side, float(max(width, height)) * 0.92)
    left = int(math.floor(center_x - side * 0.5))
    top = int(math.floor(center_y - side * 0.5))
    right = int(math.ceil(center_x + side * 0.5))
    bottom = int(math.ceil(center_y + side * 0.5))
    return left, top, right, bottom


def crop_with_transparent_border(
    image_bgr: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    left, top, right, bottom = bounds
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    canvas = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_bgr.shape[1], right)
    source_bottom = min(image_bgr.shape[0], bottom)
    if source_right <= source_left or source_bottom <= source_top:
        raise RuntimeError(f"Face crop does not intersect the source image: {bounds}")

    destination_left = source_left - left
    destination_top = source_top - top
    destination_right = destination_left + (source_right - source_left)
    destination_bottom = destination_top + (source_bottom - source_top)
    canvas[destination_top:destination_bottom, destination_left:destination_right] = image_bgr[
        source_top:source_bottom,
        source_left:source_right,
    ]
    return canvas, (left, top)


def neutralize_warm_cast(image_bgr: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, list[float]]:
    corrected = image_bgr.astype(np.float32) / 255.0
    valid = alpha > 96
    if int(np.count_nonzero(valid)) < 64:
        return image_bgr, [1.0, 1.0, 1.0]

    means = np.mean(corrected[valid], axis=0)
    target = float(np.exp(np.mean(np.log(np.maximum(means, 1e-4)))))
    gains = np.clip(target / np.maximum(means, 1e-4), 0.72, 1.30)
    corrected *= gains.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0.0, 1.0)

    hsv = cv2.cvtColor(np.rint(corrected * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 1] *= 0.82
    hsv_float[:, :, 2] = np.clip(hsv_float[:, :, 2] * 1.03, 0.0, 255.0)
    corrected = cv2.cvtColor(np.rint(hsv_float).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return corrected, [float(value) for value in gains]


def make_face_cell(
    image_bgr: np.ndarray,
    vertices: np.ndarray,
    stabilized_box: np.ndarray,
    cell_size: int,
) -> tuple[np.ndarray, dict]:
    points_xy = np.stack([vertices[0], vertices[1]], axis=1).astype(np.float32)
    bounds = square_crop_bounds(stabilized_box, image_bgr.shape[1], image_bgr.shape[0])
    crop, origin = crop_with_transparent_border(image_bgr, bounds)

    expansion = max(float(stabilized_box[2] - stabilized_box[0]), 8.0) * 0.12
    in_box = (
        (points_xy[:, 0] >= stabilized_box[0] - expansion)
        & (points_xy[:, 0] <= stabilized_box[2] + expansion)
        & (points_xy[:, 1] >= stabilized_box[1] - expansion)
        & (points_xy[:, 1] <= stabilized_box[3] + expansion)
    )
    filtered = points_xy[in_box]
    if filtered.shape[0] < 256:
        filtered = points_xy

    local_points = np.rint(filtered - np.asarray(origin, dtype=np.float32)).astype(np.int32)
    local_points[:, 0] = np.clip(local_points[:, 0], 0, crop.shape[1] - 1)
    local_points[:, 1] = np.clip(local_points[:, 1], 0, crop.shape[0] - 1)

    hull = cv2.convexHull(local_points)
    alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(alpha, hull, 255, lineType=cv2.LINE_AA)
    dilation = max(1, int(round(max(crop.shape[:2]) * 0.008)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    alpha = cv2.dilate(alpha, kernel, iterations=1)
    sigma = max(0.8, float(max(crop.shape[:2])) * 0.005)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)

    corrected, gains = neutralize_warm_cast(crop, alpha)
    rgba = cv2.cvtColor(corrected, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    resized = cv2.resize(rgba, (cell_size, cell_size), interpolation=cv2.INTER_LANCZOS4)
    return resized, {
        "crop_bounds": [int(value) for value in bounds],
        "stabilized_box": [float(value) for value in stabilized_box],
        "mask_coverage_fraction": float(np.mean(alpha > 8)),
        "color_balance_gains_bgr": gains,
    }


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames_dir).resolve()
    sequence_path = Path(args.sequence).resolve()
    output_image = Path(args.output_image).resolve()
    output_report = Path(args.output_report).resolve()

    if args.cell_size < 128 or args.cell_size > 1024:
        raise SystemExit("--cell-size must be within [128, 1024]")
    if args.columns < 1 or args.columns > 16:
        raise SystemExit("--columns must be within [1, 16]")
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory is missing: {frames_dir}")
    if not sequence_path.is_file():
        raise SystemExit(f"Reconstruction sequence is missing: {sequence_path}")

    frame_paths = sorted(frames_dir.glob("reference_*.png"))
    if not frame_paths:
        raise SystemExit(f"No extracted reference frames were found in {frames_dir}")

    data = np.load(sequence_path)
    vertices = np.asarray(data["vertices_raw"], dtype=np.float32)
    boxes = np.asarray(data["boxes"], dtype=np.float32)
    if len(frame_paths) != int(vertices.shape[0]) or len(frame_paths) != int(boxes.shape[0]):
        raise SystemExit(
            "Frame/track count mismatch: "
            f"{len(frame_paths)} frames, {vertices.shape[0]} meshes, {boxes.shape[0]} boxes"
        )
    stabilized_boxes = stabilize_boxes(boxes)

    frame_count = len(frame_paths)
    columns = min(args.columns, frame_count)
    rows = int(math.ceil(frame_count / columns))
    sheet = np.zeros((rows * args.cell_size, columns * args.cell_size, 4), dtype=np.uint8)
    frame_reports = []

    for index, (frame_path, frame_vertices, stabilized_box) in enumerate(
        zip(frame_paths, vertices, stabilized_boxes)
    ):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"OpenCV could not read extracted frame: {frame_path}")
        cell, cell_report = make_face_cell(image, frame_vertices, stabilized_box, args.cell_size)
        row = index // columns
        column = index % columns
        top = row * args.cell_size
        left = column * args.cell_size
        sheet[top : top + args.cell_size, left : left + args.cell_size] = cell
        frame_reports.append(
            {
                "frame_index": index,
                "row": row,
                "column": column,
                **cell_report,
            }
        )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image), sheet):
        raise SystemExit(f"Could not write derived face sprite sheet: {output_image}")

    crop_sides = [report["crop_bounds"][2] - report["crop_bounds"][0] for report in frame_reports]
    report = {
        "classification": "PROVEN",
        "policy": "DERIVED_FACE_ONLY_RGBA_SPRITE_SHEET_RAW_REFERENCE_EXCLUDED",
        "crop_route": "SMOOTHED_DETECTOR_BOX_WITH_DENSE_FACE_ALPHA",
        "color_route": "MASKED_GRAY_WORLD_BALANCE_AND_SATURATION_REDUCTION",
        "frame_count": frame_count,
        "columns": columns,
        "rows": rows,
        "cell_size": args.cell_size,
        "sheet_dimensions": [int(sheet.shape[1]), int(sheet.shape[0])],
        "alpha_coverage_fraction": float(np.mean(sheet[:, :, 3] > 8)),
        "crop_side_min": int(min(crop_sides)),
        "crop_side_max": int(max(crop_sides)),
        "raw_frames_packaged": False,
        "source_clip_packaged": False,
        "frames": frame_reports,
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
