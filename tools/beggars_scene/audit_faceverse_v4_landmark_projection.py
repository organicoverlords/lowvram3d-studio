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

from run_faceverse_v4_shared_identity_fit import (
    detect_face_box_and_eyes,
    parse_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FaceVerse lightweight and full-mesh landmark projections against MediaPipe targets."
    )
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmarker", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", default="19,21,23,25,27,29,31,33,35")
    parser.add_argument("--overlay-frames", default="21,31,35")
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


def map_crop_to_source(projected: torch.Tensor, bbox: torch.Tensor, image_size: int) -> torch.Tensor:
    widths = (bbox[:, 2] - bbox[:, 0]).view(-1, 1)
    heights = (bbox[:, 3] - bbox[:, 1]).view(-1, 1)
    x = projected[:, :, 0] / float(image_size) * widths + bbox[:, 0].view(-1, 1)
    y = projected[:, :, 1] / float(image_size) * heights + bbox[:, 1].view(-1, 1)
    return torch.stack((x, y), dim=2)


def describe_errors(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    errors = torch.linalg.vector_norm(predicted - target, dim=2)
    flat = errors.flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], device=flat.device),
    )
    per_point = torch.mean(errors, dim=0)
    worst_values, worst_indices = torch.topk(per_point, k=20)
    return {
        "rmse_pixels": float(torch.sqrt(torch.mean(errors**2)).detach().cpu().item()),
        "mean_pixels": float(torch.mean(errors).detach().cpu().item()),
        "quantiles_pixels": {
            key: float(value)
            for key, value in zip(
                ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
                quantiles.detach().cpu().tolist(),
            )
        },
        "worst_mean_point_errors": [
            {"index": int(index), "mean_error_pixels": float(value)}
            for value, index in zip(
                worst_values.detach().cpu().tolist(),
                worst_indices.detach().cpu().tolist(),
            )
        ],
    }


def range_report(points: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(points)
    return {
        "finite_fraction": float(torch.mean(finite.float()).detach().cpu().item()),
        "x_min": float(torch.nan_to_num(points[:, :, 0], nan=0.0).min().detach().cpu().item()),
        "x_max": float(torch.nan_to_num(points[:, :, 0], nan=0.0).max().detach().cpu().item()),
        "y_min": float(torch.nan_to_num(points[:, :, 1], nan=0.0).min().detach().cpu().item()),
        "y_max": float(torch.nan_to_num(points[:, :, 1], nan=0.0).max().detach().cpu().item()),
    }


def draw_overlay(
    image_rgb: np.ndarray,
    target: np.ndarray,
    full: np.ndarray,
    lightweight: np.ndarray,
    label: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for points, color in (
        (target, (0, 255, 0)),
        (full, (255, 128, 0)),
        (lightweight, (0, 0, 255)),
    ):
        for x, y in points:
            if np.isfinite(x) and np.isfinite(y) and -1000 <= x <= 2000 and -1000 <= y <= 2000:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{label} | green target | orange full | red lightweight",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    faulthandler.enable()
    args = parse_args()
    frames = parse_indices(args.frames)
    overlay_frames = parse_indices(args.overlay_frames)
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    landmarker_path = Path(args.landmarker).resolve()
    clip_path = Path(args.clip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (faceverse_root, model_path, checkpoint_path, landmarker_path, clip_path):
        if not required.exists():
            raise SystemExit(f"Projection-audit input missing: {required}")

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open clip: {clip_path}")
    records: dict[int, dict[str, Any]] = {}
    for frame_index in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not decode frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bbox, eyes, landmarks = detect_face_box_and_eyes(frame_rgb, landmarker_path)
        records[frame_index] = {
            "frame_rgb": frame_rgb,
            "bbox": bbox,
            "eyes": eyes,
            "target": landmarks[:478, :2],
        }
    capture.release()

    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel

    device = choose_device(args.device)
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    coefficients_rows: list[torch.Tensor] = []
    bbox_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for frame_index in frames:
        record = records[frame_index]
        coefficients, bbox_list = model.process_imgs(
            np.asarray(record["frame_rgb"], dtype=np.uint8)[None, ...],
            np.asarray(record["bbox"], dtype=np.float32).reshape(1, 1, 4),
        )
        coefficients[:, -4:] = torch.from_numpy(
            np.asarray(record["eyes"], dtype=np.float32).reshape(1, 4)
        ).to(device)
        coefficients_rows.append(coefficients.detach())
        bbox_rows.append(np.asarray(bbox_list[0], dtype=np.float32))
        target_rows.append(np.asarray(record["target"], dtype=np.float32))

    coefficients = torch.cat(coefficients_rows, dim=0)
    bbox_tensor = torch.from_numpy(np.stack(bbox_rows)).to(device)
    target = torch.from_numpy(np.stack(target_rows)).to(device)

    with torch.no_grad():
        lightweight_crop = model.run(coefficients, only_lms=True)["lms_proj"][:, :478, :2]
        _, full_crop, _, _ = model.compute_for_final(coefficients, compute_color=False)
        media_indices = model.kp_inds[:478]
        full_media_crop = full_crop[:, media_indices, :2]
        lightweight_source = map_crop_to_source(lightweight_crop, bbox_tensor, model.imgsize)
        full_source = map_crop_to_source(full_media_crop, bbox_tensor, model.imgsize)

    direct_difference = describe_errors(lightweight_source, full_source)
    report = {
        "classification": "LANDMARK_PROJECTION_AUDIT_COMPLETE",
        "route": "FACEVERSE_V4_LIGHTWEIGHT_VS_FULL_LANDMARK_PROJECTION",
        "device": str(device),
        "frames": frames,
        "target_range": range_report(target),
        "lightweight_crop_range": range_report(lightweight_crop),
        "full_crop_range": range_report(full_media_crop),
        "lightweight_source_range": range_report(lightweight_source),
        "full_source_range": range_report(full_source),
        "lightweight_vs_target": describe_errors(lightweight_source, target),
        "full_vs_target": describe_errors(full_source, target),
        "lightweight_vs_full": direct_difference,
        "bbox_rows": np.stack(bbox_rows).astype(float).tolist(),
        "overlay_files": [],
    }

    index_lookup = {frame: row for row, frame in enumerate(frames)}
    overlays: list[np.ndarray] = []
    for frame_index in overlay_frames:
        row = index_lookup[frame_index]
        overlay = draw_overlay(
            np.asarray(records[frame_index]["frame_rgb"], dtype=np.uint8),
            target[row].detach().cpu().numpy(),
            full_source[row].detach().cpu().numpy(),
            lightweight_source[row].detach().cpu().numpy(),
            f"frame {frame_index:03d}",
        )
        path = output_dir / f"projection_overlay_{frame_index:03d}.png"
        cv2.imwrite(str(path), overlay)
        report["overlay_files"].append(path.name)
        overlays.append(overlay)
    if overlays:
        sheet = np.concatenate(overlays, axis=0)
        sheet_path = output_dir / "faceverse_landmark_projection_audit.png"
        cv2.imwrite(str(sheet_path), sheet)
        report["contact_sheet"] = sheet_path.name

    report_path = output_dir / "faceverse_landmark_projection_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
