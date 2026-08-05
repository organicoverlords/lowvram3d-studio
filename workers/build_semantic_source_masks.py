"""Build asset-declared source semantic masks without embedding asset coordinates in code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sidecar", required=True)
    args = parser.parse_args()
    image = cv2.imread(args.source_image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("SOURCE_IMAGE_UNREADABLE")
    h, w = image.shape[:2]
    cfg = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    detail = cfg.get("texture", {}).get("face_detail", cfg.get("face_detail", {}))
    roi = detail.get("source_roi_normalized") or detail.get("source_roi")
    if roi is None or len(roi) != 4:
        raise RuntimeError("FACE_ROI_NOT_DECLARED")
    x0, y0, x1, y1 = [float(v) for v in roi]
    roi_px = np.array([[round(x0*w), round(y0*h)], [round(x1*w), round(y0*h)], [round(x1*w), round(y1*h)], [round(x0*w), round(y1*h)]], np.int32)
    face = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face, roi_px, 255)
    # Landmarks are explicitly declared in the asset manifest. A small ellipse around
    # them includes eyes/beak/plate/cheek boundary without expanding into props.
    landmarks = detail.get("source_landmarks_normalized", detail.get("landmarks", {}))
    points = []
    for value in landmarks.values():
        if len(value) == 2:
            points.append((round(float(value[0])*w), round(float(value[1])*h)))
    if points:
        pts = np.asarray(points, np.int32)
        centre = pts.mean(axis=0)
        radius = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))) * 0.32
        ellipse = np.zeros_like(face)
        cv2.ellipse(ellipse, tuple(np.round(centre).astype(int)), (max(1, round(radius*1.2)), max(1, round(radius))), 0, 0, 360, 255, -1)
        face = cv2.bitwise_or(face, ellipse)
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] >= 4 else np.full((h, w), 255, np.uint8)
    background = (alpha == 0).astype(np.uint8) * 255
    nonface = ((alpha > 0) & (face == 0)).astype(np.uint8) * 255
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, mask in (("original_face_mask", face), ("original_nonface_mask", nonface), ("source_background_mask", background)):
        path = out / f"{name}.png"
        cv2.imwrite(str(path), mask)
        outputs[name] = str(path)
    sidecar = {"schema": "semantic_source_masks_v1", "source_image": args.source_image, "roi_normalized": [x0, y0, x1, y1], "landmark_count": len(points), "outputs": outputs, "face_policy": "ROI plus bounded landmark ellipse; props and rear geometry are excluded by source-image scope"}
    Path(args.sidecar).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sidecar).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"SEMANTIC_MASKS face={int((face>0).sum())} nonface={int((nonface>0).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
