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
import numpy as np
import torch
import torch.nn.functional as F

import run_faceverse_v4_identity_fusion as base


DEFAULT_FRAMES = "19,21,23,25,27,29,31,33,35"
DEFAULT_ANCHORS = "21,31,35"
PRIOR_VARIANTS = {
    "flexible": 0.002,
    "balanced": 0.010,
    "conservative": 0.050,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine one shared FaceVerse identity against dense MediaPipe landmarks from multiple "
            "public frames, while retaining per-frame expression and pose."
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
    print(f"FACEVERSE_REFINE_STAGE={name}", flush=True)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def source_projected_colors(
    frame_rgb: np.ndarray,
    projected: Any,
    pca_colors: Any,
    blend: float = 0.86,
) -> tuple[np.ndarray, float]:
    points = as_numpy(projected).astype(np.float32)
    pca = as_numpy(pca_colors).astype(np.float32)
    sampled, valid = base.bilinear_sample_rgb(frame_rgb, points[:, :2])
    result = np.clip(pca, 0.0, 1.0)
    result[valid] = np.clip(
        result[valid] * (1.0 - blend) + sampled[valid] * blend,
        0.0,
        1.0,
    )
    return result, float(np.mean(valid))


def make_tile(
    source_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    refined_pca_rgb: np.ndarray,
    refined_projected_rgb: np.ndarray,
    crop: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> np.ndarray:
    panels = [
        base.crop_square(source_rgb, crop, 300),
        base.crop_square(baseline_rgb, crop, 300),
        base.crop_square(refined_pca_rgb, crop, 300),
        base.crop_square(refined_projected_rgb, crop, 300),
    ]
    tile = np.zeros((372, 1200, 3), dtype=np.uint8)
    tile[42:342] = np.concatenate(panels, axis=1)
    cv2.putText(tile, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle, (12, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    labels = ("PUBLIC SOURCE", "FUSED BASELINE", "REFINED PCA", "REFINED SOURCE COLOR")
    for index, label in enumerate(labels):
        cv2.putText(tile, label, (index * 300 + 8, 337), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("At least one frame index is required")
    return indices


def load_frames(
    clip_path: Path,
    frame_indices: list[int],
    landmarker_path: Path,
) -> tuple[dict[int, dict[str, Any]], float, list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open public clip: {clip_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise RuntimeError("Public clip has invalid FPS")

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
            bbox, eyes, landmarks = base.detect_face_box_and_eyes(frame_rgb, landmarker_path)
        except Exception as error:  # bounded public-frame rejection
            rejections.append({"frame_index": frame_index, "error": str(error)})
            continue
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        eye_distance = max(base.distance(landmarks[33], landmarks[263]), 1e-6)
        mouth_open = float(base.distance(landmarks[13], landmarks[14]) / eye_distance)
        records[frame_index] = {
            "frame_rgb": frame_rgb,
            "bbox": bbox.astype(np.float32),
            "eyes": eyes.astype(np.float32),
            "landmarks": landmarks.astype(np.float32),
            "sharpness": sharpness,
            "mouth_open": mouth_open,
            "timestamp_seconds": frame_index / fps,
        }
    capture.release()
    return records, fps, rejections


def build_full_coefficients(
    identity: torch.Tensor,
    expression: torch.Tensor,
    texture: torch.Tensor,
    gamma: torch.Tensor,
    angle: torch.Tensor,
    translation: torch.Tensor,
    eyes: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((identity.expand(expression.shape[0], -1), expression, texture, gamma, angle, translation, eyes), dim=1)


def landmark_rmse_pixels(
    model: Any,
    coefficients: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    with torch.no_grad():
        predicted = model.run(coefficients, only_lms=True)["lms_proj"][:, :478, :2]
        return float(torch.sqrt(torch.mean((predicted - targets) ** 2)).item())


def optimize_variant(
    model: Any,
    initial_identity: torch.Tensor,
    raw_coefficients: torch.Tensor,
    target_landmarks: torch.Tensor,
    point_weights: torch.Tensor,
    frame_weights: torch.Tensor,
    prior_strength: float,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    parts = model.split_coeffs_dict(raw_coefficients)
    identity = torch.nn.Parameter(initial_identity.detach().clone())
    expression = torch.nn.Parameter(parts["exp"].detach().clone())
    angle = torch.nn.Parameter(parts["angle"].detach().clone())
    translation = torch.nn.Parameter(parts["trans"].detach().clone())
    eyes = torch.nn.Parameter(parts["eyes"].detach().clone())
    texture = parts["tex"].detach().clone()
    gamma = parts["gamma"].detach().clone()

    initial_expression = expression.detach().clone()
    initial_angle = angle.detach().clone()
    initial_translation = translation.detach().clone()
    initial_eyes = eyes.detach().clone()
    initial_identity_copy = initial_identity.detach().clone()

    optimizer = torch.optim.Adam(
        [
            {"params": [identity], "lr": 0.030},
            {"params": [expression], "lr": 0.012},
            {"params": [angle], "lr": 0.003},
            {"params": [translation], "lr": 0.006},
            {"params": [eyes], "lr": 0.004},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(iterations, 1), eta_min=0.08)
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        coefficients = build_full_coefficients(identity, expression, texture, gamma, angle, translation, eyes)
        predicted = model.run(coefficients, only_lms=True)["lms_proj"][:, :478, :2]
        diff_loss = F.smooth_l1_loss(
            predicted / float(model.imgsize),
            target_landmarks / float(model.imgsize),
            beta=0.010,
            reduction="none",
        ).sum(dim=2)
        weighted = diff_loss * point_weights * frame_weights[:, None]
        landmark_loss = weighted.sum() / torch.clamp((point_weights * frame_weights[:, None]).sum(), min=1.0)
        id_prior = torch.mean((identity - initial_identity_copy) ** 2)
        id_zero = torch.mean(identity**2)
        expression_prior = torch.mean((expression - initial_expression) ** 2)
        pose_prior = torch.mean((angle - initial_angle) ** 2)
        translation_prior = torch.mean((translation - initial_translation) ** 2)
        eye_prior = torch.mean((eyes - initial_eyes) ** 2)
        loss = (
            landmark_loss
            + prior_strength * id_prior
            + 0.0004 * id_zero
            + 0.025 * expression_prior
            + 0.050 * pose_prior
            + 0.025 * translation_prior
            + 0.050 * eye_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([identity, expression, angle, translation, eyes], max_norm=8.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            identity.clamp_(-5.0, 5.0)
            expression.clamp_(-5.0, 5.0)
            angle.clamp_(-1.5, 1.5)
            eyes.clamp_(-1.0, 1.0)
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration + 1 == iterations:
            record = {
                "iteration": float(iteration + 1),
                "loss": float(loss.detach().item()),
                "landmark_loss": float(landmark_loss.detach().item()),
                "id_prior": float(id_prior.detach().item()),
            }
            history.append(record)
            print(
                f"FACEVERSE_REFINE_ITER={iteration + 1} LOSS={record['loss']:.7f} "
                f"LANDMARK={record['landmark_loss']:.7f} ID_PRIOR={record['id_prior']:.7f}",
                flush=True,
            )

    final_coefficients = build_full_coefficients(identity, expression, texture, gamma, angle, translation, eyes).detach()
    return final_coefficients, {"history": history, "prior_strength": prior_strength}


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
            raise SystemExit(f"Landmark-refinement input is missing: {required}")

    stage("LOAD_PUBLIC_FRAMES")
    frame_records, fps, rejections = load_frames(clip_path, frame_indices, landmarker_path)
    ordered_indices = sorted(frame_records)
    if len(ordered_indices) < 5:
        raise SystemExit(f"Only {len(ordered_indices)} frames passed landmark detection")
    missing_anchors = [index for index in anchor_indices if index not in frame_records]
    if missing_anchors:
        raise SystemExit(f"Requested refinement anchors failed detection: {missing_anchors}")

    stage("LOAD_FACEVERSE")
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    device = base.choose_device(args.device)
    print(f"FACEVERSE_REFINE_DEVICE={device}", flush=True)
    model_load_start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    model_load_seconds = time.perf_counter() - model_load_start
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)

    stage("INFER_INITIAL_COEFFICIENTS")
    raw_rows: list[torch.Tensor] = []
    targets: list[np.ndarray] = []
    for frame_index in ordered_indices:
        record = frame_records[frame_index]
        frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)
        coefficients, bbox_list = model.process_imgs(
            frame_rgb[np.newaxis, ...],
            np.asarray(record["bbox"], dtype=np.float32).reshape(1, 1, 4),
        )
        coefficients[:, -4:] = torch.from_numpy(np.asarray(record["eyes"], dtype=np.float32).reshape(1, 4)).to(device)
        record["raw_coefficients"] = coefficients.detach().clone()
        record["bbox_list"] = bbox_list
        raw_rows.append(coefficients.detach().clone())
        bbox = bbox_list[0].astype(np.float32)
        scale = np.maximum(bbox[2:] - bbox[:2], 1.0)
        target = (np.asarray(record["landmarks"], dtype=np.float32) - bbox[:2]) / scale * float(model.imgsize)
        targets.append(target)

    raw_coefficients = torch.cat(raw_rows, dim=0)
    target_landmarks = torch.from_numpy(np.stack(targets).astype(np.float32)).to(device)
    parts = model.split_coeffs_dict(raw_coefficients)
    yaw = torch.abs(parts["angle"][:, 1]).detach().cpu().numpy()
    sharpness = np.asarray([float(frame_records[index]["sharpness"]) for index in ordered_indices], dtype=np.float64)
    robust_weights_np = np.sqrt(np.maximum(sharpness, 1.0)) * np.exp(-yaw / 0.70)
    robust_weights_np = robust_weights_np / np.maximum(np.mean(robust_weights_np), 1e-6)
    frame_weights = torch.from_numpy(robust_weights_np.astype(np.float32)).to(device)
    initial_identity = torch.sum(parts["id"] * frame_weights[:, None], dim=0, keepdim=True) / torch.sum(frame_weights)
    initial_texture = torch.sum(parts["tex"] * frame_weights[:, None], dim=0, keepdim=True) / torch.sum(frame_weights)

    baseline_coefficients = raw_coefficients.detach().clone()
    baseline_coefficients[:, : model.id_dims] = initial_identity.expand(len(ordered_indices), -1)
    tex_start = model.id_dims + model.exp_dims
    baseline_coefficients[:, tex_start : tex_start + model.tex_dims] = initial_texture.expand(len(ordered_indices), -1)

    with torch.no_grad():
        initial_predicted = model.run(baseline_coefficients, only_lms=True)["lms_proj"][:, :478, :2]
        initial_error = torch.linalg.vector_norm((initial_predicted - target_landmarks) / float(model.imgsize), dim=2)
        point_weights = torch.exp(-initial_error / 0.035).clamp(0.12, 1.0)
        initial_rmse = landmark_rmse_pixels(model, baseline_coefficients, target_landmarks)
    print(f"FACEVERSE_REFINE_INITIAL_RMSE_PX={initial_rmse:.6f}", flush=True)

    stage("OPTIMIZE_SHARED_IDENTITY")
    variants: dict[str, dict[str, Any]] = {}
    for variant_name, prior_strength in PRIOR_VARIANTS.items():
        print(f"FACEVERSE_REFINE_VARIANT_BEGIN={variant_name}", flush=True)
        refined_coefficients, metadata = optimize_variant(
            model,
            initial_identity,
            baseline_coefficients,
            target_landmarks,
            point_weights,
            frame_weights,
            prior_strength,
            args.iterations,
        )
        final_rmse = landmark_rmse_pixels(model, refined_coefficients, target_landmarks)
        metadata["coefficients"] = refined_coefficients
        metadata["initial_rmse_pixels"] = initial_rmse
        metadata["final_rmse_pixels"] = final_rmse
        metadata["improvement_fraction"] = (initial_rmse - final_rmse) / max(initial_rmse, 1e-6)
        variants[variant_name] = metadata
        print(
            f"FACEVERSE_REFINE_VARIANT_DONE={variant_name} FINAL_RMSE_PX={final_rmse:.6f} "
            f"IMPROVEMENT={metadata['improvement_fraction']:.6f}",
            flush=True,
        )

    stage("RENDER_COMPARISONS")
    tiles: list[np.ndarray] = []
    visual_rows: list[dict[str, Any]] = []
    saved_coefficients: dict[str, np.ndarray] = {}
    index_lookup = {frame_index: row for row, frame_index in enumerate(ordered_indices)}
    for variant_name, metadata in variants.items():
        refined_all = metadata["coefficients"]
        saved_coefficients[variant_name] = refined_all.detach().cpu().numpy().astype(np.float32)
        for anchor_index in anchor_indices:
            row = index_lookup[anchor_index]
            record = frame_records[anchor_index]
            frame_rgb = np.asarray(record["frame_rgb"], dtype=np.uint8)
            bbox_list = np.asarray(record["bbox_list"], dtype=np.int32)
            baseline = baseline_coefficients[row : row + 1]
            refined = refined_all[row : row + 1]

            _, baseline_projected, _, baseline_colors = model.from_coeffs(baseline, bbox_list)
            baseline_source_colors, _ = source_projected_colors(frame_rgb, baseline_projected[0], baseline_colors[0])
            baseline_render, _ = render_fvr(frame_rgb, baseline_projected[0], triangles, np.zeros_like(baseline_projected[0]), baseline_source_colors)

            _, refined_projected, refined_normals, refined_colors = model.from_coeffs(refined, bbox_list)
            refined_pca_render, _ = render_fvr(frame_rgb, refined_projected[0], triangles, refined_normals[0], refined_colors[0])
            refined_source_colors, sampled_fraction = source_projected_colors(frame_rgb, refined_projected[0], refined_colors[0])
            refined_projected_render, _ = render_fvr(frame_rgb, refined_projected[0], triangles, refined_normals[0], refined_source_colors)

            crop = base.expanded_square(np.asarray(record["bbox"], dtype=np.float32), frame_rgb.shape[1], frame_rgb.shape[0])
            title = (
                f"{variant_name} landmark-refined identity | anchor {anchor_index:03d} | "
                f"mouth {float(record['mouth_open']):.3f}"
            )
            subtitle = (
                f"RMSE {initial_rmse:.2f}px -> {float(metadata['final_rmse_pixels']):.2f}px | "
                f"improvement {float(metadata['improvement_fraction']) * 100.0:.1f}% | sampled {sampled_fraction:.3f}"
            )
            tile = make_tile(
                frame_rgb,
                baseline_render,
                refined_pca_render,
                refined_projected_render,
                crop,
                title,
                subtitle,
            )
            key = f"{variant_name}_anchor_{anchor_index:03d}"
            compare_path = output_dir / f"landmark_refine_{key}_compare.jpg"
            render_path = output_dir / f"landmark_refine_{key}_projected.png"
            cv2.imwrite(str(compare_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(render_path), cv2.cvtColor(refined_projected_render, cv2.COLOR_RGB2BGR))
            tiles.append(tile)
            visual_rows.append(
                {
                    "variant": variant_name,
                    "anchor_frame": anchor_index,
                    "compare": compare_path.name,
                    "projected_render": render_path.name,
                    "sampled_vertex_fraction": sampled_fraction,
                }
            )

    sheet = np.zeros((len(tiles) * 372, 1200, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        sheet[index * 372 : (index + 1) * 372] = tile
    sheet_path = output_dir / "faceverse_landmark_refine.jpg"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        output_dir / "faceverse_landmark_refine_coefficients.npz",
        frame_indices=np.asarray(ordered_indices, dtype=np.int32),
        baseline=baseline_coefficients.detach().cpu().numpy().astype(np.float32),
        initial_identity=initial_identity.detach().cpu().numpy().astype(np.float32),
        point_weights=point_weights.detach().cpu().numpy().astype(np.float32),
        frame_weights=frame_weights.detach().cpu().numpy().astype(np.float32),
        **saved_coefficients,
    )

    report_variants: dict[str, Any] = {}
    for name, metadata in variants.items():
        report_variants[name] = {
            key: value
            for key, value in metadata.items()
            if key != "coefficients"
        }
    best_variant = min(report_variants, key=lambda name: float(report_variants[name]["final_rmse_pixels"]))
    best_improvement = float(report_variants[best_variant]["improvement_fraction"])
    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED" if best_improvement > 0.02 else "LANDMARK_REFINEMENT_REJECTED_NO_MATERIAL_GAIN",
        "route": "FACEVERSE_V4_MULTI_FRAME_DENSE_LANDMARK_REFINEMENT",
        "claim": (
            "One shared identity is optimized against 478 MediaPipe-corresponding landmarks across all accepted frames; "
            "per-frame expression, pose, translation and eyes remain frame-specific with priors to network estimates."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_seconds": model_load_seconds,
        "iterations": args.iterations,
        "frames_requested": frame_indices,
        "frames_used": ordered_indices,
        "anchors": anchor_indices,
        "initial_rmse_pixels": initial_rmse,
        "best_variant": best_variant,
        "variants": report_variants,
        "visual_rows": visual_rows,
        "contact_sheet": sheet_path.name,
        "coefficients": "faceverse_landmark_refine_coefficients.npz",
        "rejections": rejections,
        "public_source_frames_packaged": True,
        "source_video_packaged": False,
        "source_frame_plane_used": False,
        "fps": fps,
    }
    (output_dir / "faceverse_landmark_refine.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
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
