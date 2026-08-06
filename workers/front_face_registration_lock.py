"""Bounded front-face registration and chart-local atlas finishing.

The worker is intentionally texture-only.  It accepts a source face patch and
an atlas frame, estimates a bounded similarity transform from explicit
landmarks, and writes only the caller supplied face mask.  Direct texel owners
are immutable; conservative support and gutter pixels are handled as separate
layers after the lock.  A small luminance-only grade is applied last and is
reported separately from ownership/provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FaceRegistrationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _similarity_from_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 2:
        raise FaceRegistrationError("FACE_LOCK_LANDMARK_SHAPE_INVALID")
    source_mean, target_mean = source.mean(0), target.mean(0)
    a = source - source_mean
    b = target - target_mean
    denom = float(np.sum(a * a))
    if denom <= 1e-12:
        raise FaceRegistrationError("FACE_LOCK_DEGENERATE_LANDMARKS")
    # Orthogonal Procrustes with a uniform scale; no shear/non-rigid warp.
    u, singular, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / denom)
    linear = scale * rotation
    translation = target_mean - source_mean @ linear
    return np.asarray([[linear[0, 0], linear[1, 0], translation[0]],
                       [linear[0, 1], linear[1, 1], translation[1]]], dtype=np.float64)


def estimate_bounded_similarity(source_points: np.ndarray, target_points: np.ndarray,
                                *, max_translation: float = 24.0,
                                max_scale_delta: float = 0.08,
                                max_rotation_degrees: float = 8.0,
                                max_rmse: float = 8.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate and gate a front lock from explicit landmark correspondences."""
    matrix = _similarity_from_points(source_points, target_points)
    linear = matrix[:, :2]
    scale = float(np.sqrt(max(abs(np.linalg.det(linear)), 1e-12)))
    angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    source = np.asarray(source_points, np.float64)
    target = np.asarray(target_points, np.float64)
    predicted = np.column_stack((source, np.ones(len(source)))) @ matrix.T
    residuals = np.linalg.norm(predicted - target, axis=1)
    translation = matrix[:, 2]
    report = {
        "transform_type": "bounded_similarity",
        "scale": scale,
        "rotation_degrees": angle,
        "translation": translation.tolist(),
        "landmark_rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "landmark_max_error": float(np.max(residuals)),
        "limits": {"max_translation": max_translation, "max_scale_delta": max_scale_delta,
                   "max_rotation_degrees": max_rotation_degrees, "max_rmse": max_rmse},
    }
    if float(np.linalg.norm(translation)) > max_translation + 1e-6:
        raise FaceRegistrationError("FACE_LOCK_TRANSLATION_LIMIT")
    if abs(scale - 1.0) > max_scale_delta + 1e-6:
        raise FaceRegistrationError("FACE_LOCK_SCALE_LIMIT")
    if abs(angle) > max_rotation_degrees + 1e-6:
        raise FaceRegistrationError("FACE_LOCK_ROTATION_LIMIT")
    if report["landmark_rmse"] > max_rmse + 1e-6:
        raise FaceRegistrationError("FACE_LOCK_RESIDUAL_LIMIT")
    report["passed"] = True
    return matrix, report


def _warp_source(source: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.warpAffine(source, matrix.astype(np.float32), (width, height),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def chart_local_support(atlas: np.ndarray, direct_owner: np.ndarray,
                        conservative_owner: np.ndarray, chart_id: np.ndarray,
                        *, radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fill only conservative owners and chart-local gutters, never direct texels.

    Returns ``(atlas, gutter_mask)``.  Conservative pixels are copied from the
    nearest direct texel in the same chart.  Gutter pixels are similarly copied
    but remain unowned and are therefore excluded from provenance counts.
    """
    out = np.asarray(atlas).copy()
    direct_owner = np.asarray(direct_owner)
    conservative_owner = np.asarray(conservative_owner)
    chart_id = np.asarray(chart_id)
    if not (direct_owner.shape == conservative_owner.shape == chart_id.shape == out.shape[:2]):
        raise ValueError("FACE_LOCK_SUPPORT_SHAPE_MISMATCH")
    direct = direct_owner >= 0
    support = (conservative_owner >= 0) & ~direct
    source_mask = direct.astype(np.uint8)
    # OpenCV's pixel labels give a deterministic nearest direct texel in one
    # pass.  A chart check keeps the fast lookup from crossing UV islands.
    _distance, labels = cv2.distanceTransformWithLabels(
        (1 - source_mask) * 255, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    direct_coords = np.argwhere(direct)
    support_coords = np.argwhere(support)
    if len(direct_coords):
        nearest = direct_coords[np.clip(labels[support] - 1, 0, len(direct_coords) - 1)]
        same_chart = chart_id[support] == chart_id[nearest[:, 0], nearest[:, 1]]
        out[support_coords[same_chart, 0], support_coords[same_chart, 1]] = out[
            nearest[same_chart, 0], nearest[same_chart, 1]]
    occupied = direct | support
    gutter = np.zeros_like(direct)
    if radius > 0:
        expanded = cv2.dilate(direct.astype(np.uint8),
                              np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
        ring = expanded & ~occupied
        ring_coords = np.argwhere(ring)
        if len(direct_coords) and len(ring_coords):
            nearest = direct_coords[np.clip(labels[ring] - 1, 0, len(direct_coords) - 1)]
            same_chart = chart_id[ring] == chart_id[nearest[:, 0], nearest[:, 1]]
            out[ring_coords[same_chart, 0], ring_coords[same_chart, 1]] = out[
                nearest[same_chart, 0], nearest[same_chart, 1]]
            gutter[ring_coords[same_chart, 0], ring_coords[same_chart, 1]] = True
    return out, gutter


def luma_preserving_grade(atlas: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                          *, max_gain: float = 0.10) -> tuple[np.ndarray, dict[str, float]]:
    """Match masked luma mean/std with bounded gain, preserving chroma ratios."""
    out = np.asarray(atlas).copy().astype(np.float32)
    ref = np.asarray(reference).astype(np.float32)
    mask = np.asarray(mask, bool)
    if out.shape != ref.shape or out.shape[:2] != mask.shape:
        raise ValueError("FACE_LOCK_GRADE_SHAPE_MISMATCH")
    def lum(values: np.ndarray) -> np.ndarray:
        return 0.2126 * values[:, 0] + 0.7152 * values[:, 1] + 0.0722 * values[:, 2]
    current = lum(out[mask])
    desired = lum(ref[mask])
    if not len(current) or float(current.std()) < 1e-6:
        return np.asarray(atlas).copy(), {"gain": 1.0, "offset": 0.0, "luma_mean_before": float(current.mean()) if len(current) else 0.0,
                                         "luma_mean_after": float(current.mean()) if len(current) else 0.0}
    gain = float(np.clip(desired.std() / max(current.std(), 1e-6), 1.0 - max_gain, 1.0 + max_gain))
    offset = float(np.clip(desired.mean() - gain * current.mean(), -255.0 * max_gain, 255.0 * max_gain))
    out[mask] = np.clip(out[mask] * gain + offset, 0.0, 255.0)
    after = lum(out[mask])
    report = {"gain": gain, "offset": offset, "luma_mean_before": float(current.mean()),
              "luma_mean_after": float(after.mean()), "luma_std_before": float(current.std()),
              "luma_std_after": float(after.std()), "max_gain": max_gain}
    return np.rint(out).astype(np.uint8), report


def build_candidate(atlas: np.ndarray, source_face: np.ndarray, reference: np.ndarray,
                    source_points: np.ndarray, target_points: np.ndarray, face_mask: np.ndarray,
                    direct_owner: np.ndarray, conservative_owner: np.ndarray, chart_id: np.ndarray,
                    *, radius: int = 1, limits: dict[str, float] | None = None,
                    max_luma_gain: float = 0.10) -> tuple[np.ndarray, dict[str, Any]]:
    limits = limits or {}
    matrix, lock_report = estimate_bounded_similarity(source_points, target_points, **limits)
    warped = _warp_source(source_face, matrix, atlas.shape[:2])
    out = np.asarray(atlas).copy()
    face_mask = np.asarray(face_mask, bool)
    direct = np.asarray(direct_owner) >= 0
    # Lock only the explicitly selected face footprint; direct ownership is not altered.
    writable = face_mask & ~direct
    out[writable] = warped[writable]
    out, gutter = chart_local_support(out, direct_owner, conservative_owner, chart_id, radius=radius)
    out, grade_report = luma_preserving_grade(out, reference, face_mask, max_gain=max_luma_gain)
    report = {"schema": "panda_front_face_registration_lock_v1", "classification": "DIAGNOSTIC_ONLY",
              "lock": lock_report, "direct_texels_preserved": True,
              "face_writable_texels": int(writable.sum()), "gutter_texels": int(gutter.sum()),
              "conservative_texels": int(((np.asarray(conservative_owner) >= 0) & ~direct).sum()),
              "grade": grade_report, "ordering": ["face_lock", "chart_local_support", "chart_local_gutter", "luma_grade"]}
    return out, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--source-face", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--face-mask", type=Path, required=True)
    parser.add_argument("--direct-owner", type=Path, required=True)
    parser.add_argument("--conservative-owner", type=Path, required=True)
    parser.add_argument("--chart-id", type=Path, required=True)
    parser.add_argument("--source-landmarks", type=Path, required=True)
    parser.add_argument("--target-landmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gutter-radius", type=int, default=1)
    parser.add_argument("--max-luma-gain", type=float, default=0.10)
    args = parser.parse_args()
    read = lambda p: cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    atlas, source, reference = read(args.atlas), read(args.source_face), read(args.reference)
    if atlas is None or source is None or reference is None:
        raise SystemExit("FACE_LOCK_IMAGE_MISSING")
    face_mask = np.load(args.face_mask).astype(bool)
    direct_owner, conservative_owner, chart_id = (np.load(p) for p in (args.direct_owner, args.conservative_owner, args.chart_id))
    source_points = np.asarray(json.loads(args.source_landmarks.read_text(encoding="utf-8")), dtype=np.float64)
    target_points = np.asarray(json.loads(args.target_landmarks.read_text(encoding="utf-8")), dtype=np.float64)
    result, report = build_candidate(atlas, source, reference, source_points, target_points, face_mask,
                                     direct_owner, conservative_owner, chart_id,
                                     radius=args.gutter_radius, max_luma_gain=args.max_luma_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    report.update({"output": str(args.output), "output_sha256": _sha256(args.output),
                   "input_atlas": str(args.atlas), "input_atlas_sha256": _sha256(args.atlas),
                   "resolution": int(result.shape[0]), "promotion_authorized": False})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
