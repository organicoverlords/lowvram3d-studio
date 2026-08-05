"""Recover the deterministic transform between the authoritative and legacy panda sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask(image: np.ndarray) -> np.ndarray:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    raw = (distance > 18.0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return raw.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def affine_from_sift(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, dict]:
    sift = cv2.SIFT_create()
    key_high, desc_high = sift.detectAndCompute(cv2.cvtColor(high, cv2.COLOR_BGR2GRAY), None)
    key_low, desc_low = sift.detectAndCompute(cv2.cvtColor(low, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(desc_high, desc_low, k=2)
    good = [first for first, second in matches if first.distance < 0.75 * second.distance]
    if len(good) < 3:
        raise RuntimeError("PAN_SOURCE_TRANSFORM_INSUFFICIENT_FEATURE_MATCHES")
    source = np.float32([key_high[m.queryIdx].pt for m in good])
    target = np.float32([key_low[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=5000, confidence=0.999, refineIters=20)
    if matrix is None or inliers is None:
        raise RuntimeError("PAN_SOURCE_TRANSFORM_RANSAC_FAILED")
    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(predicted - target, axis=1)
    inlier = inliers.ravel().astype(bool)
    report = {
        "detector": "SIFT",
        "ratio_test": 0.75,
        "ransac_reprojection_threshold_px": 2.0,
        "candidate_match_count": len(good),
        "inlier_count": int(inlier.sum()),
        "inlier_fraction": float(inlier.mean()),
        "median_inlier_error_px": float(np.median(errors[inlier])),
        "p95_inlier_error_px": float(np.percentile(errors[inlier], 95)),
    }
    return matrix.astype(np.float64), report


def compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose source->middle and middle->target 2x3 affine matrices."""
    a = np.vstack([first, [0.0, 0.0, 1.0]])
    b = np.vstack([second, [0.0, 0.0, 1.0]])
    return (b @ a)[:2]


def render_overlay(low: np.ndarray, warped_high: np.ndarray, low_mask: np.ndarray,
                   warped_mask: np.ndarray, path: Path) -> None:
    # Green = legacy source, magenta = mapped authoritative source; aligned detail appears white.
    a = cv2.cvtColor(low, cv2.COLOR_BGR2RGB).astype(np.float32)
    b = cv2.cvtColor(warped_high, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = np.clip(0.5 * a + 0.5 * b, 0, 255).astype(np.uint8)
    edge_low = cv2.Canny((low_mask * 255).astype(np.uint8), 50, 150) > 0
    edge_high = cv2.Canny((warped_mask * 255).astype(np.uint8), 50, 150) > 0
    overlay[edge_low & ~edge_high] = (0, 255, 0)
    overlay[edge_high & ~edge_low] = (255, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)


def contact_sheet(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, array in images:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((416, 525), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (416, 555), (20, 20, 20))
        tile.paste(image, (0, 30))
        ImageDraw.Draw(tile).text((8, 8), label, fill=(240, 240, 240))
        tiles.append(tile)
    sheet = Image.new("RGB", (416 * 2, 555 * 2), (20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 416, (index // 2) * 555))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--conditioning", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    high_path, low_path, out = Path(args.high), Path(args.low), Path(args.output_dir)
    high = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
    low = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
    if high is None or low is None:
        raise RuntimeError("PAN_SOURCE_IMAGE_UNREADABLE")
    matrix, fit = affine_from_sift(high, low)
    height, width = low.shape[:2]
    warped = cv2.warpAffine(high, matrix, (width, height), flags=cv2.INTER_AREA,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    high_mask = mask(high)
    low_mask = mask(low)
    warped_mask = cv2.warpAffine(high_mask.astype(np.uint8), matrix, (width, height),
                                 flags=cv2.INTER_NEAREST) > 0
    uniform = cv2.resize(high, (width, height), interpolation=cv2.INTER_AREA)
    uniform_mask = mask(uniform)
    fit["mask_iou"] = iou(low_mask, warped_mask)
    fit["uniform_resize_mask_iou"] = iou(low_mask, uniform_mask)
    fit["source_matrix_high_to_low"] = matrix.tolist()
    fit["source_matrix_convention"] = "x_low = A @ x_high_homogeneous; origin top-left; rows down"
    fit["high_resolution"] = [int(high.shape[1]), int(high.shape[0])]
    fit["low_resolution"] = [int(low.shape[1]), int(low.shape[0])]
    identity = {
        "schema": "panda_source_identity_v1",
        "authoritative_source": {
            "path": str(high_path), "sha256": sha256(high_path),
            "dimensions": [int(high.shape[1]), int(high.shape[0])],
            "classification": "AUTHORITATIVE_ORIGINAL_UPSCALED",
        },
        "legacy_conditioning_source": {
            "path": str(low_path), "sha256": sha256(low_path),
            "dimensions": [int(low.shape[1]), int(low.shape[0])],
            "classification": "LEGACY_LOW_RES_CONDITIONING",
        },
        "same_design_not_alternate_character": True,
        "proof": fit,
    }
    transform = {
        "schema": "panda_highres_to_lowres_transform_v1",
        "method": "SIFT_ratio_test_then_partial_affine_RANSAC",
        "source_identity": identity["authoritative_source"],
        "target_identity": identity["legacy_conditioning_source"],
        "transform": fit,
        "candidate_baseline": {
            "uniform_resize_matrix": [[width / high.shape[1], 0.0, 0.0],
                                       [0.0, height / high.shape[0], 0.0]],
            "uniform_resize_mask_iou": fit["uniform_resize_mask_iou"],
        },
        "proven": bool(fit["inlier_count"] >= 100 and fit["median_inlier_error_px"] <= 1.0
                       and fit["mask_iou"] > fit["uniform_resize_mask_iou"]),
    }
    (out / "forensics").mkdir(parents=True, exist_ok=True)
    (out / "forensics" / "panda_source_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8")
    (out / "forensics" / "panda_highres_to_lowres_transform.json").write_text(
        json.dumps(transform, indent=2), encoding="utf-8")
    render_overlay(low, warped, low_mask, warped_mask,
                   out / "forensics" / "panda_source_alignment_overlay.png")
    contact_sheet([
        ("legacy 416x525", low),
        ("uniform resize", uniform),
        ("mapped authoritative", warped),
        ("mapped overlay", cv2.cvtColor(cv2.imread(
            str(out / "forensics" / "panda_source_alignment_overlay.png")), cv2.COLOR_BGR2RGB)),
    ], out / "forensics" / "panda_source_alignment_contact_sheet.png")
    if args.conditioning:
        conditioning_path = Path(args.conditioning)
        conditioning = cv2.imread(str(conditioning_path), cv2.IMREAD_COLOR)
        if conditioning is None:
            raise RuntimeError("PAN_CONDITIONING_IMAGE_UNREADABLE")
        low_to_conditioning, conditioning_fit = affine_from_sift(low, conditioning)
        high_to_conditioning = compose(matrix, low_to_conditioning)
        mapped_conditioning = cv2.warpAffine(
            high, high_to_conditioning, (conditioning.shape[1], conditioning.shape[0]),
            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127))
        # The conditioning image is a canvas with a deliberately uniform
        # background, so mask(conditioning) describes the canvas rectangle,
        # not the subject.  Use the legacy subject matte transformed through
        # the measured low->conditioning affine as the independent target.
        conditioning_mask = cv2.warpAffine(
            low_mask.astype(np.uint8), low_to_conditioning,
            (conditioning.shape[1], conditioning.shape[0]),
            flags=cv2.INTER_NEAREST) > 0
        mapped_conditioning_mask = cv2.warpAffine(
            high_mask.astype(np.uint8), high_to_conditioning,
            (conditioning.shape[1], conditioning.shape[0]), flags=cv2.INTER_NEAREST) > 0
        conditioning_transform = {
            "schema": "panda_highres_to_conditioning_transform_v1",
            "source": identity["authoritative_source"],
            "conditioning": {"path": str(conditioning_path),
                              "dimensions": [int(conditioning.shape[1]), int(conditioning.shape[0])]},
            "low_to_conditioning_fit": conditioning_fit,
            "source_matrix_high_to_conditioning": high_to_conditioning.tolist(),
            "matrix_convention": "x_conditioning = A @ x_high_homogeneous; origin top-left; rows down",
            "subject_mask_iou": iou(conditioning_mask, mapped_conditioning_mask),
            "proven": bool(conditioning_fit["inlier_count"] >= 100
                           and conditioning_fit["median_inlier_error_px"] <= 1.0
                           and iou(conditioning_mask, mapped_conditioning_mask) > 0.95),
        }
        (out / "forensics" / "panda_highres_to_conditioning_transform.json").write_text(
            json.dumps(conditioning_transform, indent=2), encoding="utf-8")
        Image.fromarray(cv2.cvtColor(mapped_conditioning, cv2.COLOR_BGR2RGB)).save(
            out / "forensics" / "panda_highres_in_conditioning_384.png")
        render_overlay(conditioning, mapped_conditioning, conditioning_mask,
                       mapped_conditioning_mask,
                       out / "forensics" / "panda_conditioning_alignment_overlay.png")
    print(json.dumps(transform, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
