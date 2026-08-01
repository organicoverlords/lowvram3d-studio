"""Deterministic local image comparison for localized repairs.

This is the authority in the hybrid gate. It measures a candidate crop against the baseline and
the canonical source with fixed, explainable metrics - no model, no VRAM, no timeout risk.

It is generic: the region of interest and any feature masks arrive in the manifest. Nothing about
a staff, a ring or a hole is encoded here.

Requires only numpy and Pillow, so it runs in the ordinary pipeline environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lowvram3d.visual_delta_policy import DeltaThresholds, decide  # noqa: E402

MAX_SIDE = 384


# ---------------------------------------------------------------- image helpers


def load_image(path: str, size: int = MAX_SIDE) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scale = size / float(max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def to_common_shape(*images: np.ndarray) -> list[np.ndarray]:
    height = min(i.shape[0] for i in images)
    width = min(i.shape[1] for i in images)
    out = []
    for image in images:
        pil = Image.fromarray((image * 255).astype(np.uint8))
        out.append(np.asarray(pil.resize((width, height), Image.LANCZOS), np.float32) / 255.0)
    return out


def grayscale(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Subject vs background, using the median border colour as the background reference."""
    border = np.concatenate([
        image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :],
    ])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(image - background, axis=2)
    # Otsu-style split on the distance field keeps this free of magic colour constants.
    return distance > max(0.12, float(np.median(distance)) * 0.9)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Simple iterative flood fill; adequate for <=384 px crops and dependency-free."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    height, width = mask.shape
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x]:
                continue
            current += 1
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def enclosed_opening_diameter(mask: np.ndarray) -> float:
    """Equivalent diameter of the largest background region fully enclosed by the subject.

    This is what makes an opening measurable without knowing what the feature is: a real hole is
    background that cannot reach the image border.
    """
    background = ~mask
    labels, count = label_components(background)
    if count == 0:
        return 0.0
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    best = 0
    for index in range(1, count + 1):
        if index in border_labels:
            continue
        area = int((labels == index).sum())
        best = max(best, area)
    return float(2.0 * np.sqrt(best / np.pi)) if best else 0.0


def subject_diameter(mask: np.ndarray) -> float:
    area = float(mask.sum())
    return float(2.0 * np.sqrt(area / np.pi)) if area else 0.0


def edge_map(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    peak = float(magnitude.max())
    return magnitude / peak if peak > 1e-8 else magnitude


def normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.ravel() - a.mean(), b.ravel() - b.mean()
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denominator) if denominator > 1e-8 else 0.0


def roi_mask(shape, roi) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not roi:
        mask[:] = True
        return mask
    height, width = shape
    x1, y1, x2, y2 = roi
    # Accept normalised or pixel ROI.
    if max(roi) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
    return mask


# ---------------------------------------------------------------- metrics


def compute_metrics(manifest: dict) -> dict:
    source = load_image(manifest["source_crop"])
    before = load_image(manifest["before_crop"])
    candidate = load_image(manifest["candidate_crop"])
    source, before, candidate = to_common_shape(source, before, candidate)

    before_gray, candidate_gray, source_gray = (
        grayscale(before), grayscale(candidate), grayscale(source)
    )
    before_mask = foreground_mask(before)
    candidate_mask = foreground_mask(candidate)
    source_mask = foreground_mask(source)

    region = roi_mask(before_gray.shape, manifest.get("repair_roi"))
    outside = ~region

    difference = np.abs(before_gray - candidate_gray)
    outside_change = (
        float((difference[outside] > 0.06).mean()) if outside.any() else 0.0
    )

    before_area = float(before_mask.sum())
    silhouette_ratio = (
        float(candidate_mask.sum()) / before_area if before_area > 0 else 1.0
    )

    source_open = enclosed_opening_diameter(source_mask)
    source_outer = subject_diameter(source_mask)
    candidate_open = enclosed_opening_diameter(candidate_mask)
    candidate_outer = subject_diameter(candidate_mask)

    source_fraction = source_open / source_outer if source_outer > 0 else 0.0
    candidate_fraction = candidate_open / candidate_outer if candidate_outer > 0 else 0.0
    # Ratio of the candidate's opening proportion to the source's opening proportion. Scale
    # invariant, so a render and a piece of concept art can be compared directly.
    feature_scale_ratio = (
        candidate_fraction / source_fraction if source_fraction > 1e-6
        else (0.0 if candidate_fraction <= 1e-6 else 99.0)
    )

    # Compare silhouette edges, not raw pixels. The source is textured concept art and the
    # candidate is an untextured clay render, so a greyscale edge correlation between them is
    # meaningless (measured ~0 even for a correct repair). Mask boundaries are domain-invariant.
    source_edges = edge_map(source_mask.astype(np.float32))
    edge_similarity = normalised_correlation(
        source_edges, edge_map(candidate_mask.astype(np.float32))
    )
    alignment = normalised_correlation(source_edges, edge_map(before_mask.astype(np.float32)))

    return {
        "outside_region_change": outside_change,
        "silhouette_area_ratio": silhouette_ratio,
        "feature_scale_ratio": feature_scale_ratio,
        "edge_similarity": edge_similarity,
        "source_candidate_distance": float(np.abs(source_gray - candidate_gray).mean()),
        "before_candidate_distance": float(np.abs(before_gray - candidate_gray).mean()),
    }, {
        "alignment_confidence": alignment,
        "source_opening_fraction": source_fraction,
        "candidate_opening_fraction": candidate_fraction,
        "before_opening_fraction": (
            enclosed_opening_diameter(before_mask) / subject_diameter(before_mask)
            if subject_diameter(before_mask) > 0 else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic local visual gate")
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--receipt-dir", required=True)
    args = parser.parse_args()

    receipt_root = Path(args.receipt_dir)
    receipt_root.mkdir(parents=True, exist_ok=True)
    failures = 0

    for manifest_path in args.manifest:
        # utf-8-sig: PowerShell's Set-Content -Encoding utf8 writes a BOM.
        manifest = json.loads(Path(manifest_path).read_text("utf-8-sig"))
        for key in ("source_crop", "before_crop", "candidate_crop"):
            if not Path(manifest[key]).exists():
                print(json.dumps({
                    "manifest": manifest_path, "passed": False,
                    "reason_codes": ["VISUAL_INPUT_MISSING"],
                    "missing": manifest[key],
                }))
                failures += 1
                break
        else:
            metrics, extra = compute_metrics(manifest)
            overrides = manifest.get("thresholds") or {}
            thresholds = DeltaThresholds(**overrides) if overrides else DeltaThresholds()
            verdict = decide(
                metrics, thresholds,
                require_change=bool(manifest.get("require_change", True)),
                alignment_confidence=extra["alignment_confidence"],
            )
            verdict["extra"] = {k: round(float(v), 6) for k, v in extra.items()}
            verdict["manifest"] = manifest_path
            name = Path(manifest_path).stem
            (receipt_root / f"{name}.json").write_text(
                json.dumps(verdict, indent=2), encoding="utf-8"
            )
            print(json.dumps(verdict))
            if not verdict["passed"]:
                failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
