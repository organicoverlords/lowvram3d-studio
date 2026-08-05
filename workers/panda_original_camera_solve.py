"""Bounded original-front image registration against the canonical front control mask."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_mask(image: np.ndarray) -> np.ndarray:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    raw = (distance > 18.0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return raw.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def apply_registration(mask: np.ndarray, scale: float, dx: int, dy: int) -> np.ndarray:
    size = mask.shape[0]
    scaled = max(1, int(round(size * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (scaled, scaled), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros_like(mask, dtype=np.uint8)
    x = (size - scaled) // 2 + int(dx)
    y = (size - scaled) // 2 + int(dy)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(size, x + scaled), min(size, y + scaled)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[y0 - y:y1 - y, x0 - x:x1 - x]
    return canvas.astype(bool)


def contour_distance(a: np.ndarray, b: np.ndarray) -> float:
    edge_a = cv2.Canny((a * 255).astype(np.uint8), 50, 150) > 0
    edge_b = cv2.Canny((b * 255).astype(np.uint8), 50, 150) > 0
    if not edge_a.any() or not edge_b.any():
        return float("inf")
    dist_a = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, 3)
    dist_b = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * (dist_b[edge_a].mean() + dist_a[edge_b].mean()))


def score(candidate: np.ndarray, target: np.ndarray) -> dict:
    union = np.count_nonzero(candidate | target)
    intersection = np.count_nonzero(candidate & target)
    iou = float(intersection / max(union, 1))
    contour = contour_distance(candidate, target)
    return {
        "foreground_mask_iou": iou,
        "contour_distance_px": contour,
        "source_visible_mesh_coverage": float(intersection / max(np.count_nonzero(target), 1)),
        "objective": float(iou - 0.002 * contour),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True)
    parser.add_argument("--conditioning-transform", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    high_path = Path(args.high)
    transform_path = Path(args.conditioning_transform)
    bundle = Path(args.bundle)
    out = Path(args.output_dir)
    high = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
    if high is None:
        raise RuntimeError("PAN_CAMERA_SOURCE_UNREADABLE")
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    matrix = np.asarray(transform["source_matrix_high_to_conditioning"], dtype=np.float64)
    base = cv2.warpAffine(high, matrix, (384, 384), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    # Derive the foreground before compositing into the conditioning canvas.
    # The historical conditioning image contains the source's white rectangular
    # background inside a gray canvas; treating that rectangle as foreground
    # would solve the wrong camera.
    high_mask = source_mask(high)
    candidate_mask = cv2.warpAffine(high_mask.astype(np.uint8), matrix, (384, 384),
                                    flags=cv2.INTER_NEAREST) > 0
    target_mask = cv2.imread(str(bundle / "horizontal_1_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    if candidate_mask.shape != target_mask.shape:
        raise RuntimeError("PAN_CAMERA_MASK_DIMENSION_MISMATCH")
    scores = []
    best = None
    # Same bounded search envelope as the established registration helper, with
    # a finer deterministic refinement around it.  No camera or mesh deformation.
    for scale in np.arange(0.94, 1.061, 0.01):
        for dx in range(-24, 25, 2):
            for dy in range(-24, 25, 2):
                transformed = apply_registration(candidate_mask, float(scale), dx, dy)
                record = {"scale": round(float(scale), 4), "dx": dx, "dy": dy}
                record.update(score(transformed, target_mask))
                scores.append(record)
                if best is None or record["objective"] > best["objective"]:
                    best = record
    assert best is not None
    # One-pixel local refinement around the best coarse candidate.
    refined = []
    for scale in np.arange(best["scale"] - 0.01, best["scale"] + 0.0101, 0.0025):
        for dx in range(best["dx"] - 2, best["dx"] + 3):
            for dy in range(best["dy"] - 2, best["dy"] + 3):
                transformed = apply_registration(candidate_mask, float(scale), dx, dy)
                record = {"scale": round(float(scale), 4), "dx": dx, "dy": dy}
                record.update(score(transformed, target_mask))
                refined.append(record)
                if record["objective"] > best["objective"]:
                    best = record
    selected = apply_registration(candidate_mask, best["scale"], best["dx"], best["dy"])
    overlay = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    edge_source = cv2.Canny((selected * 255).astype(np.uint8), 50, 150) > 0
    edge_target = cv2.Canny((target_mask * 255).astype(np.uint8), 50, 150) > 0
    overlay[edge_source & ~edge_target] = (255, 0, 255)
    overlay[edge_target & ~edge_source] = (0, 255, 0)
    (out / "forensics").mkdir(parents=True, exist_ok=True)
    solution = {
        "schema": "panda_original_camera_solution_v1",
        "mesh_unchanged": True,
        "uv_unchanged": True,
        "source_image": {"path": str(high_path), "sha256": sha256(high_path),
                          "dimensions": [int(high.shape[1]), int(high.shape[0])]},
        "conditioning_transform": str(transform_path),
        "front_control": {
            "mask": str(bundle / "horizontal_1_mask.png"),
            "mask_sha256": sha256(bundle / "horizontal_1_mask.png"),
            "resolution": [384, 384],
        },
        "registration": best,
        "camera_parameters": {
            "semantic": "front",
            "camera_basis_source": str(bundle / "camera_contract.json"),
            "projection": "existing orthographic front control; no camera basis change",
        },
        "source_visible_mesh_coverage": best["source_visible_mesh_coverage"],
        "foreground_mask_iou": best["foreground_mask_iou"],
        "contour_distance_px": best["contour_distance_px"],
        "bounded_search": {"coarse_count": len(scores), "refinement_count": len(refined),
                           "scale_range": [0.94, 1.06], "translation_range_px": [-24, 24]},
        "proven": bool(best["foreground_mask_iou"] >= 0.85 and best["contour_distance_px"] <= 8.0),
    }
    (out / "forensics" / "panda_original_camera_solution.json").write_text(
        json.dumps(solution, indent=2), encoding="utf-8")
    (out / "forensics" / "panda_camera_candidate_scores.json").write_text(
        json.dumps({"coarse": scores, "refined": refined, "selected": best}, indent=2),
        encoding="utf-8")
    cv2.imwrite(str(out / "forensics" / "panda_camera_alignment_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(json.dumps(solution, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
