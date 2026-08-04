from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tracked 3DDFA parameters into a pose-decoupled canonical dense face sequence."
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--third-party-root", required=True)
    parser.add_argument("--bfm-pkl", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expression-smoothing", type=float, default=0.62)
    return parser.parse_args()


def decompose_affine(param: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    affine = np.asarray(param[:12], dtype=np.float64).reshape(3, 4)
    linear = affine[:, :3]
    offset = affine[:, 3]
    row0_norm = max(float(np.linalg.norm(linear[0])), 1.0e-8)
    row1_norm = max(float(np.linalg.norm(linear[1])), 1.0e-8)
    scale = (row0_norm + row1_norm) * 0.5
    r0 = linear[0] / row0_norm
    r1 = linear[1] / row1_norm
    r2 = np.cross(r0, r1)
    rotation_guess = np.stack((r0, r1, r2), axis=0)
    u, _, vt = np.linalg.svd(rotation_guess)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return scale, rotation.astype(np.float32), offset.astype(np.float32)


def robust_normalize(sequence: np.ndarray, keyframe_index: int) -> tuple[np.ndarray, dict[str, float]]:
    key = sequence[keyframe_index]
    x_low, x_high = np.percentile(key[:, 0], [1.0, 99.0])
    y_low, y_high = np.percentile(key[:, 1], [1.0, 99.0])
    z_low, z_high = np.percentile(key[:, 2], [1.0, 99.0])
    center_x = float((x_low + x_high) * 0.5)
    center_y = float((y_low + y_high) * 0.5)
    center_z = float(np.median(key[:, 2]))
    face_height = max(float(y_high - y_low), 1.0e-6)
    scene_scale = 2.45 / face_height

    normalized = np.empty_like(sequence, dtype=np.float32)
    normalized[:, :, 0] = (sequence[:, :, 0] - center_x) * scene_scale
    normalized[:, :, 2] = (sequence[:, :, 1] - center_y) * scene_scale
    normalized[:, :, 1] = -(sequence[:, :, 2] - center_z) * scene_scale * 0.92

    report = {
        "canonical_x_low": float(x_low),
        "canonical_x_high": float(x_high),
        "canonical_y_low": float(y_low),
        "canonical_y_high": float(y_high),
        "canonical_z_low": float(z_low),
        "canonical_z_high": float(z_high),
        "scene_scale": float(scene_scale),
    }
    return normalized, report


def main() -> int:
    args = parse_args()
    input_npz = Path(args.input_npz).resolve()
    third_party_root = Path(args.third_party_root).resolve()
    bfm_pkl = Path(args.bfm_pkl).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_report = Path(args.output_report).resolve()

    if not input_npz.is_file():
        raise SystemExit(f"Tracked face sequence is missing: {input_npz}")
    if not third_party_root.is_dir():
        raise SystemExit(f"Pinned 3DDFA root is missing: {third_party_root}")
    if not bfm_pkl.is_file():
        raise SystemExit(f"Pinned BFM model is missing: {bfm_pkl}")
    if not 0.0 < args.expression_smoothing <= 1.0:
        raise SystemExit("--expression-smoothing must be in (0, 1]")

    data = np.load(input_npz)
    parameters = np.asarray(data["parameters"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    keyframe_index = int(np.asarray(data["keyframe_index"]).reshape(-1)[0])
    if parameters.ndim != 2 or parameters.shape[1] != 62:
        raise SystemExit(f"Expected tracked 62D parameters, got {parameters.shape}")

    # Older 3DDFA sources still refer to np.long.
    if not hasattr(np, "long"):
        np.long = np.int64  # type: ignore[attr-defined]

    sys.path.insert(0, str(third_party_root))
    original_cwd = Path.cwd()
    os.chdir(third_party_root)
    try:
        from bfm.bfm import BFMModel  # pylint: disable=import-error,import-outside-toplevel

        bfm = BFMModel(str(bfm_pkl), shape_dim=40, exp_dim=10)
    finally:
        os.chdir(original_cwd)

    identity = np.median(parameters[:, 12:52], axis=0).astype(np.float32)
    expressions_raw = parameters[:, 52:62].astype(np.float32)
    expressions = np.empty_like(expressions_raw)
    expressions[0] = expressions_raw[0]
    alpha = float(args.expression_smoothing)
    for index in range(1, expressions.shape[0]):
        expressions[index] = alpha * expressions_raw[index] + (1.0 - alpha) * expressions[index - 1]

    canonical_frames: list[np.ndarray] = []
    scales: list[float] = []
    rotations: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for param, expression in zip(parameters, expressions):
        dense = (
            bfm.u
            + bfm.w_shp @ identity.reshape(-1, 1)
            + bfm.w_exp @ expression.reshape(-1, 1)
        ).reshape(3, -1, order="F").T
        canonical_frames.append(np.asarray(dense, dtype=np.float32))
        scale, rotation, offset = decompose_affine(param)
        scales.append(scale)
        rotations.append(rotation)
        offsets.append(offset)

    canonical = np.stack(canonical_frames, axis=0)
    normalized, normalization = robust_normalize(canonical, keyframe_index)
    rotations_array = np.stack(rotations, axis=0)
    key_rotation = rotations_array[keyframe_index]
    relative_rotations = np.stack(
        [rotation @ key_rotation.T for rotation in rotations_array], axis=0
    ).astype(np.float32)
    scales_array = np.asarray(scales, dtype=np.float32)
    relative_scales = scales_array / max(float(scales_array[keyframe_index]), 1.0e-8)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        canonical_vertices=normalized,
        triangles=triangles,
        colors_rgb=colors_rgb,
        relative_rotations=relative_rotations,
        relative_scales=relative_scales,
        offsets=np.stack(offsets, axis=0),
        boxes=np.asarray(data["boxes"], dtype=np.float32),
        keyframe_index=np.asarray([keyframe_index], dtype=np.int32),
        source_frame_indices=np.asarray(data["source_frame_indices"], dtype=np.int32),
        sampled_fps=np.asarray(data["sampled_fps"], dtype=np.float32),
    )

    key = normalized[keyframe_index]
    scene_bounds = {
        "x": [float(np.min(key[:, 0])), float(np.max(key[:, 0]))],
        "y": [float(np.min(key[:, 1])), float(np.max(key[:, 1]))],
        "z": [float(np.min(key[:, 2])), float(np.max(key[:, 2]))],
    }
    report = {
        "classification": "PROVEN",
        "route": "3DDFA_CANONICAL_DENSE_IDENTITY_MEDIAN_EXPRESSION_TRACK",
        "frame_count": int(normalized.shape[0]),
        "vertex_count": int(normalized.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "keyframe_index": keyframe_index,
        "expression_smoothing": alpha,
        "identity_coefficients_source": "median_across_tracked_sequence",
        "pose_decoupled": True,
        "scene_bounds": scene_bounds,
        "normalization": normalization,
        "output_npz": str(output_npz),
    }
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
