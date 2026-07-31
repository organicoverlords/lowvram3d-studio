from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np
from PIL import Image


def bbox_from_alpha(alpha: np.ndarray, threshold: float = 0.2) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(alpha) >= threshold)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def component_report(alpha: np.ndarray, threshold: float = 0.28) -> dict[str, Any]:
    binary = (np.asarray(alpha) >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    areas = [int(value) for value in stats[1:, cv2.CC_STAT_AREA]] if count > 1 else []
    areas.sort(reverse=True)
    significant = [area for area in areas if area >= max(64, int(binary.size * 0.002))]
    return {
        "component_count": len(areas),
        "significant_component_count": len(significant),
        "areas": areas[:8],
        "second_to_first_ratio": round(significant[1] / significant[0], 6) if len(significant) > 1 else 0.0,
        "labels": labels,
        "stats": stats,
    }


def keep_largest_component(alpha: np.ndarray, threshold: float = 0.28) -> tuple[np.ndarray, dict[str, Any]]:
    report = component_report(alpha, threshold)
    stats = report.pop("stats")
    labels = report.pop("labels")
    if len(stats) <= 1:
        return np.asarray(alpha, dtype=np.float32), report
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    hard = (labels == keep).astype(np.uint8)
    halo = max(5, int(round(min(alpha.shape) * 0.012)))
    if halo % 2 == 0:
        halo += 1
    dilated = cv2.dilate(hard, np.ones((halo, halo), np.uint8), iterations=1)
    return np.asarray(alpha, dtype=np.float32) * dilated.astype(np.float32), report


def _guided_filter(guidance: np.ndarray, target: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    guide = np.asarray(guidance, dtype=np.float32)
    value = np.asarray(target, dtype=np.float32)
    size = (radius * 2 + 1, radius * 2 + 1)
    mean_i = cv2.boxFilter(guide, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(value, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * value, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + epsilon)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)
    return mean_a * guide + mean_b


def refine_alpha(alpha: np.ndarray, rgb: np.ndarray, pose_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    matte = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    height, width = matte.shape
    if pose_mask is not None:
        pose = cv2.resize(np.asarray(pose_mask, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        pose = np.clip(pose, 0.0, 1.0)
        core = (matte >= 0.55).astype(np.uint8)
        envelope_size = max(17, int(round(min(height, width) * 0.045)))
        if envelope_size % 2 == 0:
            envelope_size += 1
        envelope_kernel = np.ones((envelope_size, envelope_size), np.uint8)
        envelope = cv2.morphologyEx(core, cv2.MORPH_CLOSE, envelope_kernel, iterations=1)
        envelope = cv2.dilate(envelope, envelope_kernel, iterations=1)
        # Pose segmentation fills holes and missing limbs only near the trusted BiRefNet subject.
        matte = np.maximum(matte, pose * envelope.astype(np.float32) * 0.84)

    matte, components = keep_largest_component(matte)
    strong = (matte >= 0.54).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    soft = cv2.GaussianBlur(matte, (0, 0), 0.72)
    matte = np.maximum(soft, strong.astype(np.float32) * 0.965)

    guidance = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    radius = max(3, min(12, int(round(min(height, width) / 180))))
    guided = _guided_filter(guidance, matte, radius=radius, epsilon=0.0025)
    uncertain = (matte > 0.015) & (matte < 0.985)
    matte[uncertain] = matte[uncertain] * 0.58 + guided[uncertain] * 0.42
    matte = cv2.bilateralFilter(matte.astype(np.float32), 7, 0.08, 4.0)
    matte[matte < 0.012] = 0.0
    matte[matte > 0.988] = 1.0
    return np.clip(matte, 0.0, 1.0), components


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    matte = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    sigma = max(1.2, min(4.0, min(matte.shape) / 420.0))
    weight = cv2.GaussianBlur(matte, (0, 0), sigma)
    estimate = np.empty_like(image)
    for channel in range(3):
        weighted = cv2.GaussianBlur(image[..., channel] * matte, (0, 0), sigma)
        estimate[..., channel] = weighted / np.maximum(weight, 1e-4)
    edge = np.clip((1.0 - matte) * 1.35, 0.0, 1.0)
    edge *= ((matte > 0.015) & (matte < 0.985)).astype(np.float32)
    cleaned = image * (1.0 - edge[..., None]) + estimate * edge[..., None]
    return np.clip(cleaned * 255.0, 0, 255).astype(np.uint8)


def framing_report(alpha: np.ndarray, pose: dict[str, Any]) -> dict[str, Any]:
    height, width = alpha.shape
    box = bbox_from_alpha(alpha)
    warnings: list[str] = []
    if box is None:
        return {"ready": False, "rig_ready": False, "warnings": ["No foreground subject remained after masking"]}
    x0, y0, x1, y1 = box
    margins = {
        "left": x0 / width,
        "right": (width - x1) / width,
        "top": y0 / height,
        "bottom": (height - y1) / height,
    }
    coverage = float(np.mean(alpha >= 0.2))
    if coverage < 0.08:
        warnings.append("Subject is too small in the frame")
    if coverage > 0.78:
        warnings.append("Foreground occupies almost the whole frame; body may be cropped")
    for side, value in margins.items():
        if value < 0.003:
            warnings.append(f"Subject touches the {side} frame edge")

    full_body = False
    if pose.get("detected") and len(pose.get("landmarks", [])) >= 33:
        landmarks = pose["landmarks"]
        required = (11, 12, 23, 24, 25, 26, 27, 28)
        full_body = all(float(landmarks[index].get("visibility", 0.0)) >= 0.45 for index in required)
        if not full_body:
            warnings.append("Full shoulders, hips, knees and ankles were not confidently visible")
    else:
        warnings.append("Human pose landmarks were not detected")
    rig_ready = full_body and coverage >= 0.06
    quality_ready = rig_ready and all(value >= 0.003 for value in margins.values()) and coverage <= 0.78
    return {
        "ready": quality_ready,
        "rig_ready": rig_ready,
        "full_body_detected": full_body,
        "coverage": round(coverage, 6),
        "bbox_pixels": [x0, y0, x1, y1],
        "margins": {key: round(value, 6) for key, value in margins.items()},
        "warnings": warnings,
    }


def normalize_subject(
    rgb: np.ndarray,
    alpha: np.ndarray,
    pose: dict[str, Any],
    canvas_size: int = 1024,
    subject_fill: float = 0.86,
) -> tuple[Image.Image, np.ndarray, dict[str, Any], dict[str, Any]]:
    box = bbox_from_alpha(alpha, threshold=0.08)
    if box is None:
        raise RuntimeError("Cannot normalize an empty foreground mask")
    height, width = alpha.shape
    x0, y0, x1, y1 = box
    pad = int(round(max(x1 - x0, y1 - y0) * 0.065))
    crop_x0, crop_y0 = max(0, x0 - pad), max(0, y0 - pad)
    crop_x1, crop_y1 = min(width, x1 + pad), min(height, y1 + pad)
    crop_rgb = np.asarray(rgb, dtype=np.uint8)[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_alpha = np.asarray(alpha, dtype=np.float32)[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_h, crop_w = crop_alpha.shape
    scale = min((canvas_size * subject_fill) / max(crop_w, 1), (canvas_size * subject_fill) / max(crop_h, 1))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    offset_x = (canvas_size - resized_w) // 2
    offset_y = (canvas_size - resized_h) // 2

    rgb_image = Image.fromarray(crop_rgb, "RGB").resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    alpha_image = Image.fromarray(np.round(crop_alpha * 255.0).astype(np.uint8), "L").resize(
        (resized_w, resized_h), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    patch = rgb_image.convert("RGBA")
    patch.putalpha(alpha_image)
    canvas.alpha_composite(patch, (offset_x, offset_y))
    normalized_alpha = np.asarray(canvas.getchannel("A"), dtype=np.float32) / 255.0

    transformed_pose = deepcopy(pose)
    original_landmarks = deepcopy(pose.get("landmarks", []))
    transformed_pose["source_landmarks"] = original_landmarks
    for landmark in transformed_pose.get("landmarks", []):
        source_x = float(landmark.get("x", 0.5)) * width
        source_y = float(landmark.get("y", 0.5)) * height
        landmark["x"] = ((source_x - crop_x0) * scale + offset_x) / canvas_size
        landmark["y"] = ((source_y - crop_y0) * scale + offset_y) / canvas_size
        landmark["z"] = float(landmark.get("z", 0.0)) * scale * max(width, height) / canvas_size

    transform = {
        "canvas_size": canvas_size,
        "subject_fill": subject_fill,
        "source_size": [width, height],
        "crop_box": [crop_x0, crop_y0, crop_x1, crop_y1],
        "resized_size": [resized_w, resized_h],
        "offset": [offset_x, offset_y],
        "scale": round(scale, 8),
    }
    return canvas, normalized_alpha, transformed_pose, transform
