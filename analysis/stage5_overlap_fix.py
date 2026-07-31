"""Recompute UV overlap honestly.

The first metric rasterised every triangle and counted pixels covered more than once, which
double-counts the shared edge between any two adjacent triangles in the same chart -- that is
normal connectivity, not overlap. Each triangle is inset slightly toward its centroid here so
only genuine surface-on-surface overlap is counted.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

NPZ = sys.argv[1]
RESOLUTION = int(sys.argv[2])
INSET_PIXELS = 0.75

data = np.load(NPZ)
uvs = data["uvs"]
indices = data["indices"]

pixel_uv = uvs.copy()
pixel_uv[:, 0] *= RESOLUTION - 1
pixel_uv[:, 1] = (1.0 - pixel_uv[:, 1]) * (RESOLUTION - 1)
tri_uv = pixel_uv[indices]

counter = np.zeros((RESOLUTION, RESOLUTION), np.uint16)
for triangle in tri_uv:
    centroid = triangle.mean(axis=0)
    offset = triangle - centroid
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    inset = triangle - offset / np.maximum(length, 1e-9) * INSET_PIXELS
    x_lo = max(int(np.floor(inset[:, 0].min())), 0)
    x_hi = min(int(np.ceil(inset[:, 0].max())), RESOLUTION - 1)
    y_lo = max(int(np.floor(inset[:, 1].min())), 0)
    y_hi = min(int(np.ceil(inset[:, 1].max())), RESOLUTION - 1)
    if x_hi < x_lo or y_hi < y_lo:
        continue
    local = np.zeros((y_hi - y_lo + 1, x_hi - x_lo + 1), np.uint8)
    cv2.fillPoly(local, [(inset - np.array([x_lo, y_lo])).astype(np.int32)], 1)
    counter[y_lo:y_hi + 1, x_lo:x_hi + 1] += local

covered = int((counter > 0).sum())
overlapped = int((counter > 1).sum())
print(json.dumps({
    "inset_pixels": INSET_PIXELS,
    "covered_pixels": covered,
    "overlapping_pixels": overlapped,
    "overlap_percent": round(overlapped / max(covered, 1) * 100, 4),
}, indent=2))
